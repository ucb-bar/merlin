"""Run consumed LLVM loops against independent neutral memory arithmetic.

No real target ISA, simulator, performance result or implicit effect qualification.
"""
from dataclasses import replace
import importlib.util
import io
from pathlib import Path
import shutil
import subprocess

import pytest
from xdsl.dialects import llvm
from xdsl.dialects.builtin import ModuleOp, StringAttr, i32, i64
from xdsl.printer import Printer

from merlin.compile.scheduling.llvm_program import FnBuilderIndexAdapter
from merlin.common.paths import repo_root
from merlin.llvmlower.host_builder import FnBuilder, PTR

from test_dense_source import compile_source, source
from test_selected_dense_contracts import inputs

ROOT = repo_root()
SPEC = importlib.util.spec_from_file_location("dense_llvm_example", ROOT / "examples/compiler/source_dense_llvm.py")
EXAMPLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXAMPLE)

ARGUMENTS = {
    "lhs_load": ("a", "b", "src_row", "src_col", "rows", "cols", "dst", "step"),
    "weight_load": ("a", "b", "src_row", "src_col", "rows", "cols", "dst", "step"),
    "preload": ("weight", "k", "n", "acc", "continue", "rows", "cols", "depth"),
    "compute": ("lhs", "m", "k", "rows", "depth"),
    "store": ("row", "col", "rows", "cols", "acc"),
}


def tools():
    candidates = [Path("/work/merlin/toolchains/llvm-a47bddcc/install/bin")]
    executable = shutil.which("mlir-translate")
    if executable:
        candidates.insert(0, Path(executable).parent)
    for directory in candidates:
        if all((directory / tool).is_file() for tool in ("mlir-translate", "clang")):
            return directory
    pytest.skip("native LLVM translation/clang unavailable")


def native_program(function, work, shape):
    directory = tools()
    declarations = [llvm.FuncOp(name, llvm.LLVMFunctionType([PTR, PTR, PTR] + [i64] * len(args), llvm.LLVMVoidType()),
                               linkage=llvm.LinkageAttr("external")) for name, args in ARGUMENTS.items()]
    module = ModuleOp([*declarations, function])
    module.verify()
    stream = io.StringIO()
    Printer(stream=stream).print_op(module)
    mlir = work / "program.mlir"
    mlir.write_text(stream.getvalue())
    ll = work / "program.ll"
    subprocess.run([str(directory / "mlir-translate"), "--mlir-to-llvmir", str(mlir), "-o", str(ll)],
                   check=True, capture_output=True, text=True)
    m, k, n = shape
    # Runtime is deliberately independent of schedule projection/instance expansion.
    # It consumes the compiled callback ABI, catches early store/reseed mistakes,
    # and compares every output against a separate scalar source recurrence.
    runtime = work / "runtime.c"
    runtime.write_text(f"""
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define M {m}
#define K {k}
#define N {n}
#define P void *ap, void *bp, void *yp
#define L int64_t
static int8_t left[128][64], right[128][64];
static int32_t sums[128][64];
static L wr, ar, cont, nc, depth;
static int writes[M][N];
void lhs_load(P,L a,L b,L r,L c,L nr,L cols,L dst,L step) {{
  if(nr<=0||cols<=0||r+nr>M||c+cols>K) abort();
  for(L i=0;i<nr;i++) for(L j=0;j<cols;j++) left[dst+i][j]=((int8_t*)ap)[(r+i)*K+c+j];
}}
void weight_load(P,L a,L b,L r,L c,L nr,L cols,L dst,L step) {{
  if(nr<=0||cols<=0||r+nr>K||c+cols>N) abort();
  for(L i=0;i<nr;i++) for(L j=0;j<cols;j++) right[dst+i][j]=((int8_t*)bp)[(r+i)*N+c+j];
}}
void preload(P,L weight,L k,L n,L acc,L keep,L rows,L cols,L d) {{
  if(keep!=0&&keep!=1) abort(); wr=weight; ar=acc; cont=keep; nc=cols; depth=d;
}}
void compute(P,L lhs,L m,L k,L rows,L d) {{
  if(d!=depth) abort();
  for(L i=0;i<rows;i++) for(L j=0;j<nc;j++) {{
    int32_t value=cont?sums[ar+i][j]:0;
    for(L t=0;t<d;t++) value+=(int32_t)left[lhs+i][t]*(int32_t)right[wr+t][j];
    sums[ar+i][j]=value;
  }}
}}
void store(P,L row,L col,L rows,L cols,L acc) {{
  if(rows<=0||cols<=0||row+rows>M||col+cols>N) abort();
  for(L i=0;i<rows;i++) for(L j=0;j<cols;j++) {{
    ((int32_t*)yp)[(row+i)*N+col+j]=sums[acc+i][j]; writes[row+i][col+j]++;
  }}
}}
extern void compiled(void*,void*,void*);
int main(void) {{
 int8_t a[M*K],b[K*N]; int32_t y[M*N];
 for(int i=0;i<M*K;i++) a[i]=i%7-3;
 for(int i=0;i<K*N;i++) b[i]=i%5-2;
 for(int i=0;i<M*N;i++) y[i]=-987654;
 compiled(a,b,y);
 for(int i=0;i<M;i++) for(int j=0;j<N;j++) {{
   int32_t expected=0; for(int k=0;k<K;k++) expected+=(int32_t)a[i*K+k]*b[k*N+j];
   if(y[i*N+j]!=expected||writes[i][j]!=1) return 2;
 }}
 puts("all outputs exact; each stored once"); return 0;
}}
""")
    binary = work / "program"
    subprocess.run([str(directory / "clang"), "-O1", str(ll), str(runtime), "-o", str(binary)],
                   check=True, capture_output=True, text=True)
    return subprocess.run([str(binary)], capture_output=True, text=True), stream.getvalue()


@pytest.mark.parametrize("shape,corrupt_callback", [((5, 14, 13), False), ((1, 7, 9), False), ((5, 14, 13), True)])
def test_source_bound_schedule_is_consumed_by_native_llvm_with_all_dimension_tails(tmp_path, shape, corrupt_callback):
    tools()
    model = inputs(tmp_path, tile=(2, 3, 4), shape=shape)
    module, operation = source(shape, model[0].dense.dtypes)
    compiled = compile_source(model, module, operation)
    observed = []

    def lower(builder, pointers, instruction, operands):
        assert set(operands) == set(ARGUMENTS[instruction])
        assert all(value.type == i64 for value in operands.values())
        observed.append(instruction)
        if corrupt_callback and instruction == "preload":
            # Structurally valid LLVM can still violate target semantics: erase
            # accumulation continuation. Independent output comparison catches it.
            operands = {**operands, "continue": builder.const(0)}
        builder.add(llvm.CallOp(instruction, *[pointers[name] for name in ("lhs", "rhs", "output")],
                               *[operands[name] for name in ARGUMENTS[instruction]]))

    function, receipt = EXAMPLE.emit_dense_function(compiled, lower, symbol="compiled")
    result, text = native_program(function, tmp_path, shape)
    assert result.returncode == (2 if corrupt_callback else 0), result.stderr
    if not corrupt_callback:
        assert "all outputs exact" in result.stdout
    assert "llvm.cond_br" in text and "llvm.select" in text
    assert set(observed) == set(ARGUMENTS)
    assert receipt["source_sha256"] == compiled.source_stamp.source_sha256
    assert receipt["kernel_sha256"] == compiled.compiled.kernel.digest()
    assert receipt["machine_emission"] == "requires_independent_target_qualification"


@pytest.mark.parametrize("mutation", ["source", "contract"])
def test_source_or_contract_mutation_during_emission_refuses(tmp_path, mutation):
    model = inputs(tmp_path)
    module, operation = source((5, 14, 13), model[0].dense.dtypes)
    compiled = compile_source(model, module, operation)
    calls = []

    def change(builder, pointers, instruction, operands):
        calls.append(instruction)
        if mutation == "source":
            module.attributes["changed"] = StringAttr("different source")
        else:
            model[0].software_path.write_bytes(model[0].software_bytes + b"\n")

    with pytest.raises(ValueError, match="changed"):
        EXAMPLE.emit_dense_function(compiled, change, symbol="invalid")
    assert calls


def test_selected_32_bit_index_mode_is_not_silently_widened(tmp_path):
    selected, problem, storage, adapter = inputs(tmp_path)
    module, operation = source((5, 14, 13), selected.dense.dtypes)
    compiled = compile_source((selected, problem, storage, replace(adapter, index_bits=32)), module, operation)
    assert compiled.compiled.index_bits == 32
    observed = []
    with pytest.raises(ValueError, match="64-bit indices"):
        EXAMPLE.emit_dense_function(compiled, lambda *args: observed.append(args), symbol="unsupported")
    assert not observed


def test_fn_builder_adapter_checks_index_width_direction_and_signed_loop_step():
    builder = FnBuilder([])
    adapter = FnBuilderIndexAdapter(builder)
    for bounds in ((2, 0, 1), (0, 2, 0), (0, 2, -1), (-1, 2, 1), (0, 1 << 63, 1), (0, (1 << 63)-1, 2)):
        with pytest.raises(ValueError, match="ascending"):
            adapter.loop(*bounds, lambda value: None)
    with pytest.raises(ValueError, match="i64"):
        adapter.value(builder.const(7, i32))
    with pytest.raises(ValueError, match="signed i64"):
        adapter.value(1 << 63)
    with pytest.raises(ValueError, match="equality"):
        adapter.cmp("ult", 0, 1)
    observed = []
    adapter.loop(2, 9, 3, lambda value: observed.append(value.type))
    assert observed == [i64]  # Body compiled once, never Python-expanded.
    function = llvm.FuncOp("steps", llvm.LLVMFunctionType([], llvm.LLVMVoidType()), body=builder.finish())
    function.verify()
    predicates = [op.predicate.value.data for op in function.walk() if isinstance(op, llvm.ICmpOp)]
    assert predicates == [llvm.ICmpPredicateFlag.SLT.to_int()]
