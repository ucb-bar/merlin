"""Independent C instruction machine and actual LLVM codec for a neutral target."""

from dataclasses import replace

from device_staged_dense_fixture import attributes, canonical
from xdsl.dialects import llvm
from xdsl.parser import Parser

from merlin.compile.scheduling.block_schedule import Compute, Load, Preload, Store
from merlin.compile.scheduling.compact_dense import DenseLoopAdapter, DenseLoopStorage, DenseMemoryView
from merlin.llvmlower.counted_llvm_cfg import CountedPointer, project_counted_function
from merlin.sched.ir import Call, Const, Loop, Scaled, Select, Sum, Var, as_expr
from merlin.targetgen.oot_starterkit.llvm_context import make_llvm_context
from merlin.transforms.dense_planning import DenseProblem, DenseResources

FIELDS = {
    "lhs_load": ("a", "b", "src_row", "src_col", "rows", "cols", "dst", "step"),
    "weight_load": ("a", "b", "src_row", "src_col", "rows", "cols", "dst", "step"),
    "preload": ("weight", "k", "n", "acc", "continue", "rows", "cols", "depth"),
    "preload_keep": ("k", "n", "acc", "continue", "rows", "cols", "depth"),
    "compute": ("lhs", "m", "k", "rows", "depth"),
    "compute_keep": ("lhs", "m", "k", "rows", "depth"),
    "store": ("row", "col", "rows", "cols", "acc"),
}


def encode(op):
    if isinstance(op, Load):
        name = op.role + "_load"
        values = (*op.block, op.dram_row, op.dram_col, op.rows, op.cols, op.row, op.step)
    elif isinstance(op, Preload):
        values = (*op.weight_block, op.accumulator_row, op.accumulate, op.rows, op.cols, op.weight_rows)
        name = "preload_keep" if op.weight_row is None else "preload"
        if op.weight_row is not None:
            values = (op.weight_row, *values)
    elif isinstance(op, Compute):
        name = "compute" if op.fresh_weights else "compute_keep"
        values = (op.input_row, *op.input_block, op.rows, op.cols)
    elif isinstance(op, Store):
        name, values = "store", (op.dram_row, op.dram_col, op.rows, op.cols, op.accumulator_row)
    else:
        raise ValueError("unknown neutral instruction")
    return Call(name, tuple(zip(FIELDS[name], map(as_expr, values), strict=True)))


def project(call):
    v = dict(call.args)
    if tuple(v) != FIELDS.get(call.instr):
        raise ValueError("unknown neutral instruction or fields")
    if call.instr.endswith("_load"):
        return Load(
            call.instr.removesuffix("_load"),
            (v["a"], v["b"]),
            v["src_row"],
            v["src_col"],
            v["rows"],
            v["cols"],
            v["dst"],
            v["step"],
        )
    if call.instr.startswith("preload"):
        return Preload(v.get("weight"), (v["k"], v["n"]), v["acc"], v["continue"], v["rows"], v["cols"], v["depth"])
    if call.instr.startswith("compute"):
        return Compute(v["lhs"], (v["m"], v["k"]), v["rows"], v["depth"], call.instr == "compute")
    return Store(v["row"], v["col"], v["rows"], v["cols"], v["acc"])


ADAPTER = DenseLoopAdapter(
    encode,
    project,
    "rectangular-c-machine-v1",
    "exact-small-integer-v1",
    True,
    True,
    64,
    qualification_identity="test-runtime-only",
    retained_weights=True,
)


def fixture():
    p = DenseProblem(5, 7, 9, 2, 3, 4, 1, 1, 4)
    s = DenseLoopStorage(
        DenseMemoryView("left", "left_sram", 3, 0, 12),
        DenseMemoryView("right", "right_sram", 4, 0, 12),
        DenseMemoryView("acc", "acc_sram", 16, 0, 12),
    )
    r = DenseResources(84, 192, 3, 3, 2, partial_output_tiles=True, partial_output_blocks=True)
    return p, r, s


def module(kernel):
    """Emit compact control flow from actual Kernel operands, never instances()."""
    lines, serial = [], 0

    def fresh():
        nonlocal serial
        serial += 1
        return f"%v{serial}"

    def expr(value, env):
        if type(value) is Var:
            return env[value.name]
        if type(value) is Const:
            name = fresh()
            lines.append(f"{name} = llvm.mlir.constant({value.value} : i64) : i64")
            return name
        if type(value) is Sum:
            result = expr(value.terms[0], env)
            for term in value.terms[1:]:
                rhs, name = expr(term, env), fresh()
                lines.append(f"{name} = llvm.add {result}, {rhs} : i64")
                result = name
            return result
        if type(value) is Scaled:
            left, right, name = expr(value.expr, env), expr(Const(value.factor), env), fresh()
            lines.append(f"{name} = llvm.mul {left}, {right} : i64")
            return name
        if type(value) is Select:
            right, yes, no = expr(Const(value.equals), env), expr(value.then, env), expr(value.other, env)
            condition, name = fresh(), fresh()
            lines.append(f'{condition} = llvm.icmp "eq" {env[value.var]}, {right} : i64')
            lines.append(f"{name} = llvm.select {condition}, {yes}, {no} : i1, i64")
            return name
        raise ValueError("unsupported neutral expression")

    def body(statements, env):
        for statement in statements:
            if type(statement) is Loop:
                zero = expr(Const(0), env)
                label = fresh()[1:]
                variable = fresh()
                lines.extend([f"llvm.br ^{label}({zero} : i64)", f"^{label}({variable}: i64):"])
                body(statement.body, {**env, statement.var: variable})
                one, end = expr(Const(1), env), expr(Const(statement.extent), env)
                next_value, condition = fresh(), fresh()
                exit_label = fresh()[1:]
                lines.extend(
                    [
                        f"{next_value} = llvm.add {variable}, {one} : i64",
                        f'{condition} = llvm.icmp "ult" {next_value}, {end} : i64',
                        f"llvm.cond_br {condition}, ^{label}({next_value} : i64), ^{exit_label}",
                        f"^{exit_label}:",
                    ]
                )
            else:
                assert tuple(k for k, _ in statement.args) == FIELDS[statement.instr]
                values = [expr(value, env) for _, value in statement.args]
                types = ", ".join(["!llvm.ptr"] * 3 + ["i64"] * len(values))
                lines.append(f"llvm.call @{statement.instr}(%a, %b, %c, {', '.join(values)}) : ({types}) -> ()")

    declarations = [
        f"llvm.func @{name}({', '.join(['!llvm.ptr'] * 3 + ['i64'] * len(fields))})" for name, fields in FIELDS.items()
    ]
    body(kernel.body, {})
    text = "\n".join(
        [
            "builtin.module {",
            *declarations,
            "llvm.func @kernel(%a: !llvm.ptr, %b: !llvm.ptr, %c: !llvm.ptr) {",
            *lines,
            "llvm.return",
            "}",
            "}",
        ]
    )
    result = Parser(make_llvm_context(), text).parse_module()
    result.verify()
    return result


def frame(actual):
    functions = [op for op in actual.body.block.ops if isinstance(op, llvm.FuncOp)]
    if [op.sym_name.data for op in functions] != [*FIELDS, "kernel"]:
        raise ValueError("neutral module roster changed")
    function = functions[-1]
    return (
        attributes(actual),
        tuple(canonical(op) for op in actual.body.block.ops if op is not function),
        tuple((key, canonical(value, attribute=True)) for key, value in sorted(function.properties.items())),
        tuple((key, canonical(value, attribute=True)) for key, value in sorted(function.attributes.items())),
    )


class Owner:
    def __init__(self, original, kernel, mode=None):
        self.frame = frame(original)
        self.kernel = kernel
        self.mode = mode

    def project(self, actual):
        if self.mode == "unknown" or frame(actual) != self.frame:
            return None
        function = list(actual.body.block.ops)[-1]
        pointers = tuple(CountedPointer(arg.name) for arg in self.kernel.args)

        def observe(op, inputs):
            if not isinstance(op, llvm.CallOp) or op.callee is None:
                return None
            name = op.callee.root_reference.data
            if name not in FIELDS or inputs[:3] != pointers or len(inputs) != len(FIELDS[name]) + 3:
                raise ValueError("neutral actual instruction pointer or ABI differs")
            return Call(name, tuple(zip(FIELDS[name], inputs[3:], strict=True)))

        return replace(self.kernel, body=project_counted_function(function, pointers, observe=observe))

    def emit(self, original, selected):
        if self.mode == "ignore":
            return original.clone()
        emitted = module(selected)
        result = original.clone()
        old = result.body.block.last_op
        new = emitted.body.block.last_op
        new.detach()
        new.properties.clear()
        new.properties.update(old.properties)
        new.attributes.clear()
        new.attributes.update(old.attributes)
        result.body.block.insert_op_before(new, old)
        result.body.block.erase_op(old)
        if self.mode == "wrong_pointer":
            call = next(op for op in new.walk() if isinstance(op, llvm.CallOp))
            call.operands = (new.body.blocks.first.args[1], *call.operands[1:])
        return result


RUNTIME = """#include <stdint.h>
#include <stdlib.h>
static int8_t left[12][3],right[12][4],pe[3][4];
static int32_t acc[12][4];
static long wr,ar,cont,rr,cc,dd,pending,keep,pe_valid;
long counts[7];
static void check(int ok){if(!ok)abort();}
void lhs_load(int8_t*a,int8_t*b,int32_t*c,long m,long k,long sr,long sc,long r,long n,long dst,long step){
 check(sr==m*2&&sc==k*3&&r>0&&r<=2&&n>0&&n<=3&&sr+r<=5&&sc+n<=7&&dst>=0&&dst+r<=12&&step==1);
 for(long i=0;i<r;i++)for(long j=0;j<n;j++)left[dst+i][j]=a[(sr+i)*7+sc+j];counts[0]++;}
void weight_load(int8_t*a,int8_t*b,int32_t*c,long k,long n,long sr,long sc,long r,long cols,long dst,long step){
 check(sr==k*3&&sc==n*4&&r>0&&r<=3&&cols>0&&cols<=4&&sr+r<=7&&sc+cols<=9&&dst>=0&&dst+r<=12&&step==1);
 for(long i=0;i<r;i++)for(long j=0;j<cols;j++)right[dst+i][j]=b[(sr+i)*9+sc+j];counts[1]++;}
static void desc(long row,long dest,long continuation,long rows,long cols,long depth,long retained){
 check(!pending&&dest>=0&&dest+rows<=12&&rows>0&&rows<=2&&cols>0&&cols<=4&&depth>0&&depth<=3);
 wr=row;ar=dest;cont=continuation;rr=rows;cc=cols;dd=depth;keep=retained;pending=1;counts[2]++;}
void preload(int8_t*a,int8_t*b,int32_t*c,long row,long k,long n,long dest,
 long continuation,long rows,long cols,long depth){
 check(row>=0&&row+depth<=12);desc(row,dest,continuation,rows,cols,depth,0);}
void preload_keep(int8_t*a,int8_t*b,int32_t*c,long k,long n,long dest,long continuation,long rows,long cols,long depth){
 desc(0,dest,continuation,rows,cols,depth,1);}
static void compute_common(long row,long rows,long depth,long retained){
 check(pending&&rows==rr&&depth==dd&&retained==keep&&row>=0&&row+rows<=12);
 if(!retained){for(long d=0;d<depth;d++)for(long j=0;j<cc;j++)pe[d][j]=right[wr+d][j];pe_valid=1;counts[5]++;}
 check(pe_valid);
 for(long i=0;i<rows;i++)for(long j=0;j<cc;j++){
  int32_t value=cont?acc[ar+i][j]:0;
  for(long d=0;d<depth;d++)value+=left[row+i][d]*pe[d][j];acc[ar+i][j]=value;
 }pending=0;counts[3]++;if(retained)counts[6]++;}
void compute(int8_t*a,int8_t*b,int32_t*c,long row,long m,long k,long rows,long depth){compute_common(row,rows,depth,0);}
void compute_keep(int8_t*a,int8_t*b,int32_t*c,long row,long m,long k,long rows,long depth){
 compute_common(row,rows,depth,1);}
void store(int8_t*a,int8_t*b,int32_t*c,long row,long col,long rows,long cols,long src){
 check(!pending&&row>=0&&row+rows<=5&&col>=0&&col+cols<=9&&src>=0&&src+rows<=12);
 for(long i=0;i<rows;i++)for(long j=0;j<cols;j++)c[(row+i)*9+col+j]=acc[src+i][j];counts[4]++;}
"""
