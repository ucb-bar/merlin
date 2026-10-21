"""Neutral compact-loop provider; instruction meanings are this C fixture only."""

from dataclasses import replace
from io import StringIO

from test_target_staged_dense import Owner, operations, parse, values
from xdsl.dialects import builtin, llvm

from merlin.common.dense_float_ir import DenseFloatPrinter
from merlin.compile.scheduling import staged_dense as S

ROWS = 3


def canonical(item, *, attribute=False):
    text = StringIO()
    printer = DenseFloatPrinter(stream=text, print_generic_format=True)
    if attribute:
        printer.print_attribute(item)
    else:
        printer.print_op(item)
    return text.getvalue()


def attributes(module):
    # Unregistered metadata attribute classes are context-local. Compare the
    # complete generic representations, not Python class identity across imports.
    return tuple((name, canonical(value, attribute=True)) for name, value in sorted(module.attributes.items()))


def loop_module(*, retain_rhs=False):
    lines = [
        "builtin.module {",
        "llvm.func @event(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64)",
        "llvm.func @kernel(%a: !llvm.ptr, %b: !llvm.ptr, %c: !llvm.ptr) {",
        "%zero = llvm.mlir.constant(0 : i64) : i64",
        "%one = llvm.mlir.constant(1 : i64) : i64",
        f"%end = llvm.mlir.constant({ROWS} : i64) : i64",
        "llvm.br ^loop(%zero : i64)",
        "^loop(%row: i64):",
    ]
    for index, operation in enumerate(operations()[:9]):
        if index == 2 and retain_rhs:
            lines.extend(
                [
                    '%first = llvm.icmp "eq" %row, %zero : i64',
                    "llvm.cond_br %first, ^rhs, ^rest",
                    "^rhs:",
                ]
            )
        operands = []
        for position, value in enumerate(values(operation)):
            if position == 2 and (index == 1 or type(operation) is S.AccumulatorReadout):
                operands.append("%row")
            else:
                name = f"%v{index}_{position}"
                lines.append(f"{name} = llvm.mlir.constant({value} : i64) : i64")
                operands.append(name)
        lines.append(
            "llvm.call @event(%a, %b, %c, " + ", ".join(operands) + ") : "
            "(!llvm.ptr,!llvm.ptr,!llvm.ptr,i64,i64,i64,i64,i64,i64,i64) -> ()"
        )
        if index == 2 and retain_rhs:
            lines.extend(["llvm.br ^rest", "^rest:"])
    lines.extend(
        [
            "%next = llvm.add %row, %one : i64",
            '%more = llvm.icmp "ult" %next, %end : i64',
            "llvm.cond_br %more, ^loop(%next : i64), ^exit",
            "^exit:",
            "llvm.return",
            "}",
            "}",
        ]
    )
    return parse("\n".join(lines))


def expected_operations():
    result = []
    for row in range(ROWS):
        for operation in operations()[:9]:
            if type(operation) is S.OperandLoad and operation.role == "lhs":
                operation = replace(operation, source_row=row)
            if type(operation) is S.AccumulatorReadout:
                operation = replace(operation, output_row=row)
            result.append(operation)
    return tuple(result)


class LoopOwner(Owner):
    def __init__(self, original, mode=None):
        super().__init__(mode)
        self.module_attributes = dict(original.attributes)
        self.module_attribute_record = attributes(original)
        self.module_metadata = tuple(op.clone() for op in original.body.block.ops if not isinstance(op, llvm.FuncOp))
        for op in self.module_metadata:
            if (
                op.name != "builtin.unregistered"
                or op.attributes != {"op_name__": builtin.StringAttr("llvm.module_flags")}
                or set(op.properties) != {"flags"}
                or op.operands
                or op.results
                or op.regions
                or op.successors
            ):
                raise ValueError("unknown top-level neutral LLVM metadata")

    def project(self, actual):
        if self.mode == "unknown" or attributes(actual) != self.module_attribute_record or actual.properties:
            return None
        functions = [op for op in actual.body.block.ops if isinstance(op, llvm.FuncOp)]
        metadata = [op for op in actual.body.block.ops if not isinstance(op, llvm.FuncOp)]
        if len(functions) != 2 or len(metadata) != len(self.module_metadata):
            return None
        if not all(canonical(a) == canonical(b) for a, b in zip(metadata, self.module_metadata, strict=True)):
            return None
        if [fn.sym_name.data for fn in functions] != ["event", "kernel"]:
            return None
        # Delegate the same exhaustive actual-call parser after exact attribute
        # equality and this test ABI's sole function-name correspondence.
        clone = actual.clone()
        clone.attributes.clear()
        for op in list(clone.body.block.ops):
            if not isinstance(op, llvm.FuncOp):
                clone.body.block.erase_op(op)
        clone.body.block.last_op.properties["sym_name"] = builtin.StringAttr("entry")
        # Imported typed metadata wrappers subclass the regular LLVM operations.
        # Reparse every property generically in the original fixture's closed
        # LLVM schema; unsupported properties refuse instead of disappearing.
        text = StringIO()
        DenseFloatPrinter(stream=text, print_generic_format=True).print_op(clone)
        return super().project(parse(text.getvalue()))

    def emit(self, source, before, after):
        if self.mode == "ignore":
            return source.clone()
        if before != expected_operations():
            raise ValueError("unexpected neutral compact-loop source effects")
        rhs = [op for op in after if type(op) is S.OperandLoad and op.role == "rhs"]
        if len(rhs) not in (1, ROWS):
            raise ValueError("unsupported compact retained-load pattern")
        result = loop_module(retain_rhs=len(rhs) == 1)
        result.attributes.update(self.module_attributes)
        for metadata in self.module_metadata:
            result.body.block.insert_op_before(metadata.clone(), result.body.block.first_op)
        if self.mode == "wrong_pointer":
            function = result.body.block.last_op
            first = next(op for op in function.walk() if type(op) is llvm.CallOp)
            operands = list(first.operands)
            operands[1] = function.body.blocks.first.args[0]
            first.operands = operands
        return result


RUNTIME = """#include <stdint.h>
static int32_t cells[2], acc; static int rhs,pending;
long counts[2];
void event(int8_t*a,int8_t*b,int32_t*c,long tag,long x,long y,long z,long u,long v,long w){
 switch(tag){
 case 0:
  if(x<0||x>1||y<0||y>2||z||u!=x||v!=1||w!=1||(x&&y))__builtin_trap();
  cells[u]=x?b[y]:a[y];counts[x]++;break;
 case 1:if(x||y!=1||z!=1||u||v||w)__builtin_trap();acc=0;break;
 case 2:if(x!=1||y||z!=1||u!=1||v!=1||w!=1||pending)__builtin_trap();rhs=x;pending=1;break;
 case 3:if(x||y!=1||z!=1||u||v||w||!pending)__builtin_trap();acc+=cells[x]*cells[rhs];pending=0;break;
 case 4:if(x||y<0||y>2||z||u!=1||v!=1||w||pending)__builtin_trap();c[y]=acc;break;
 case 5:if(x||y||z||u||v||w)__builtin_trap();break;
 default:__builtin_trap();}
}
"""
