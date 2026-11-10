"""Native f32 lowering derives numeric and borrowing facts from complete bodies."""

import os
import subprocess

import pytest

from merlin.llvmlower.atomic_compare_exchange import SUPPORT
from merlin.llvmlower.toolchain import available, compiler_python

pytestmark = pytest.mark.skipif(not available(), reason="existing native MLIR required")


def source(change=None):
    body = """module {
      func.func @forward(%p:memref<?xf32>,%i:index,%value:f32)->f32 {
        %old=memref.generic_atomic_rmw %p[%i]:memref<?xf32> {
        ^bb0(%current:f32):
          %exponent=arith.constant 2139095040:i32
          %sign=arith.constant -2147483648:i32
          %zero=arith.constant 0:i32
          %word=arith.bitcast %current:f32 to i32
          %field=arith.andi %word,%exponent:i32
          %small=arith.cmpi eq,%field,%zero:i32
          %signedzero=arith.andi %word,%sign:i32
          %flushed=arith.select %small,%signedzero,%word:i32
          %floating=arith.bitcast %flushed:i32 to f32
          %sum=arith.addf %floating,%value:f32
          %sumword=arith.bitcast %sum:f32 to i32
          %sumfield=arith.andi %sumword,%exponent:i32
          %sumsmall=arith.cmpi eq,%sumfield,%zero:i32
          %sumsignedzero=arith.andi %sumword,%sign:i32
          %desired=arith.select %sumsmall,%sumsignedzero,%sumword:i32
          %fraction=arith.constant 8388607:i32
          %quiet=arith.constant 4194304:i32
          %nanfield=arith.andi %desired,%exponent:i32
          %all_exponent=arith.cmpi eq,%nanfield,%exponent:i32
          %mantissa=arith.andi %desired,%fraction:i32
          %nonzero=arith.cmpi ne,%mantissa,%zero:i32
          %nan=arith.andi %all_exponent,%nonzero:i1
          %quietword=arith.ori %desired,%quiet:i32
          %stored=arith.select %nan,%quietword,%desired:i32
          %result=arith.bitcast %stored:i32 to f32
          memref.atomic_yield %result:f32
        }
        return %old:f32
      }
    }"""
    if change == "sign": body = body.replace("-2147483648", "1073741824")
    elif change == "quiet": body = body.replace("4194304", "2097152")
    elif change == "exponent": body = body.replace("2139095040", "2130706432")
    elif change == "input": body = body.replace("%floating,%value", "%current,%value")
    elif change == "output": body = body.replace("memref.atomic_yield %result", "memref.atomic_yield %sum")
    elif change == "extra": body = body.replace("%result=", "%unused=arith.addf %current,%value:f32\n%result=")
    elif change == "fast": body = body.replace("%floating,%value:f32", "%floating,%value fastmath<fast>:f32")
    elif change == "dtype": body = body.replace("f32", "f64").replace("i32", "i64")
    elif change == "space": body = body.replace("memref<?xf32>", "memref<?xf32, 3>")
    elif change == "layout": body = body.replace("memref<?xf32>", "memref<?xf32, strided<[2], offset: ?>>")
    return body


def run(tmp_path, program, body):
    original, script = tmp_path / "source.mlir", tmp_path / "run.py"
    original.write_text(program)
    script.write_text("from torch_mlir import ir,passmanager\nimport sys\n" + SUPPORT +
        "\nwith ir.Context() as ctx:\n    module=ir.Module.parse(open(sys.argv[1]).read())\n" +
        "    module.operation.verify()\n" + "\n".join("    " + line for line in body.splitlines()))
    process = subprocess.run([str(compiler_python()), str(script), str(original)], capture_output=True,
        text=True, timeout=20, env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(tmp_path)))
    assert process.returncode == 0, process.stderr


@pytest.mark.parametrize("canonical", [False, True])
@pytest.mark.parametrize("identity", [False, True])
@pytest.mark.parametrize("ieee", [False, True])
def test_f32_region_keeps_atomic_load_retry_and_returned_old_word(tmp_path, canonical, identity, ieee):
    before = "passmanager.PassManager.parse('builtin.module(canonicalize,cse)').run(module.operation)\n" if canonical else ""
    program = source()
    if ieee:
        omitted = ('%sign=', '%word=', '%field=', '%small=', '%signedzero=', '%flushed=', '%floating=',
                   '%sumfield=', '%sumsmall=', '%sumsignedzero=', '%desired=')
        program = '\n'.join(line for line in program.splitlines() if not line.strip().startswith(omitted))
        program = program.replace('%floating,%value', '%current,%value').replace('%desired', '%sumword')
    if identity:
        operand = '%current' if ieee else '%floating'
        program = program.replace(f'%sum=arith.addf {operand},%value:f32', '').replace('%sumword=arith.bitcast %sum:f32 to i32', f'%sumword=arith.bitcast {operand}:f32 to i32')
    run(tmp_path, program, before + """assert _cas_rewrite(ctx,module,word_views=True)==1
passmanager.PassManager.parse('builtin.module(canonicalize,cse)').run(module.operation)
assert len(_cas_checked_calls(module))==1
assert sum(op.name=='llvm.cmpxchg' for op in _cas_walk(module.operation))==1
assert sum(op.name=='llvm.load' for op in _cas_walk(module.operation))==1
assert sum(op.name=='scf.while' for op in _cas_walk(module.operation))==1
_cas_suspend(module)
assert len(_CAS_PENDING)==1 and not _CAS_PENDING[0][1].regions[0].blocks
passmanager.PassManager.parse('builtin.module(ownership-based-buffer-deallocation,buffer-deallocation-simplification,canonicalize,bufferization-lower-deallocations,canonicalize)').run(module.operation)
_cas_restore(module)
passmanager.PassManager.parse('builtin.module(canonicalize,cse)').run(module.operation)
assert len(_cas_checked_calls(module))==1
passmanager.PassManager.parse('builtin.module(expand-strided-metadata,lower-affine,convert-scf-to-cf,finalize-memref-to-llvm,convert-func-to-llvm,convert-arith-to-llvm,convert-cf-to-llvm,reconcile-unrealized-casts)').run(module.operation)
module.operation.verify()
assert not any(op.name.startswith(('scf.','memref.','arith.')) for op in _cas_walk(module.operation))
""")


@pytest.mark.parametrize("change", ["sign", "quiet", "exponent", "input", "output", "extra", "fast", "dtype", "space", "layout"])
def test_changed_ftz_arithmetic_is_refused_before_mutating_ir(tmp_path, change):
    run(tmp_path, source(change), """before=str(module)
try:
    _cas_rewrite(ctx,module,word_views=True)
except ValueError:
    assert str(module)==before
else:
    raise AssertionError('unqualified numeric body accepted')
""")


@pytest.mark.parametrize("change", [None, "order", "load", "bytes", "linkage", "arithmetic", "capture", "return"])
@pytest.mark.parametrize("ieee", [False, True])
def test_f32_borrowing_grant_checks_the_actual_complete_current_helper(tmp_path, change, ieee):
    replacements = {"order": ("acq_rel acquire", "seq_cst acquire"),
        "load": (" atomic acquire", ""), "bytes": ("arith.constant 4 : index", "arith.constant 8 : index"),
        "linkage": ("#llvm.linkage<internal>", "#llvm.linkage<external>"),
        "arithmetic": ("%sum = arith.addf", "%sum = arith.subf"),
        "return": ("%result = arith.bitcast %old", "%result = arith.bitcast %start")}
    mutate = f"text=text.replace({replacements[change][0]!r},{replacements[change][1]!r})\n" if change in replacements else ""
    if change == "capture":
        mutate = "text=text.replace('module {','module {func.func private @capture(memref<?xf32>)').replace('%bytes =','func.call @capture(%buffer):(memref<?xf32>)->()\\n%bytes =')\n"
    run(tmp_path, source(), f"text=_f32_add_source('arbitrary_name',ieee={ieee})\n" + mutate + """module=ir.Module.parse(text.rsplit('}',1)[0]+'func.func @forward(%p:memref<?xf32>,%i:index,%v:f32)->f32 {%r=func.call @arbitrary_name(%p,%i,%v):(memref<?xf32>,index,f32)->f32 return %r:f32}}')
module.operation.verify()
passmanager.PassManager.parse('builtin.module(canonicalize,cse)').run(module.operation)
""" + f"assert len(_cas_checked_calls(module))=={0 if change else 1}\n")
