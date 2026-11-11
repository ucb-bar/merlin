"""Native BF16 transport keeps word joins without changing arithmetic/ABI."""

import os
import subprocess

import pytest

from merlin.llvmlower.bfloat_transport import MARKER, SUPPORT, bind_runner, edit_pipeline
from merlin.llvmlower.toolchain import available, compiler_python

SOURCE = '''module {
  func.func @transport(%storage: memref<4xbf16>, %choose: i1, %count: index) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %value = memref.load %storage[%zero] : memref<4xbf16>
    cf.br ^loop(%value, %count : bf16, index)
  ^loop(%current: bf16, %remaining: index):
    %continue = arith.cmpi sgt, %remaining, %zero : index
    cf.cond_br %continue, ^body, ^done(%current : bf16)
  ^body:
    %other = memref.load %storage[%one] : memref<4xbf16>
    %selected = arith.select %choose, %current, %other : bf16
    %next = arith.subi %remaining, %one : index
    cf.br ^loop(%selected, %next : bf16, index)
  ^done(%result: bf16):
    memref.store %result, %storage[%zero] : memref<4xbf16>
    return
  }
}'''


@pytest.mark.skipif(not available(), reason="existing native compiler environment required")
@pytest.mark.parametrize("change", [None, "constant", "arithmetic", "fastmath", "scalar-input", "scalar-result", "unlowered-region"])
def test_native_transport_and_refusals_leave_complete_ir(tmp_path, change):
    additions = {
        "fastmath": "",
        "scalar-input": "func.func private @unqualified(bf16)",
        "scalar-result": "func.func private @unqualified() -> bf16",
        "constant": '''func.func @constant(%p:memref<1xbf16>) {
          %z=arith.constant 0:index
          %v=arith.constant 0x7F81:bf16
          memref.store %v,%p[%z]:memref<1xbf16>
          return
        }''',
        "arithmetic": '''func.func @arithmetic(%p:memref<1xbf16>) {
          %z=arith.constant 0:index
          %v=memref.load %p[%z]:memref<1xbf16>
          %r=arith.addf %v,%v:bf16
          memref.store %r,%p[%z]:memref<1xbf16>
          return
        }''',
        "unlowered-region": '''func.func @unqualified(%x:memref<4xbf16>,%n:index) {
          %z=arith.constant 0:index
          %o=arith.constant 1:index
          %v=memref.load %x[%z]:memref<4xbf16>
          %r=scf.for %i=%z to %n step %o iter_args(%a=%v)->bf16 {scf.yield %a:bf16}
          memref.store %r,%x[%z]:memref<4xbf16>
          return
        }''',
    }
    program = SOURCE if change is None else SOURCE[:-1] + additions[change] + "}"
    source, runner = tmp_path / "source.mlir", tmp_path / "run.py"
    source.write_text(program)
    runner.write_text("from torch_mlir import ir,passmanager\nimport sys\n" + SUPPORT + '''
with ir.Context():
    module=ir.Module.parse(open(sys.argv[1]).read())
    if sys.argv[2]!='unlowered-region':
        passmanager.PassManager.parse('builtin.module(convert-scf-to-cf,finalize-memref-to-llvm,convert-func-to-llvm,convert-arith-to-llvm,convert-cf-to-llvm,reconcile-unrealized-casts)').run(module.operation)
    if sys.argv[2]=='fastmath':
        selected=next(op for op in _bf_walk(module.operation) if op.name=='llvm.select')
        selected.attributes['fastmathFlags']=ir.Attribute.parse('#llvm.fastmath<nnan>')
    before=str(module)
    if sys.argv[2] in ('positive','constant'):
        assert _bf_transport(module)>0
        assert _bf_transport(module)==0
        for op in _bf_walk(module.operation):
            assert all(str(value.type)!='bf16' for value in (*op.operands,*op.results))
            if op.name=='llvm.select':
                assert str(op.results[0].type)=='i16'
        assert 'llvm.load' in str(module) and 'llvm.store' in str(module)
        if sys.argv[2]=='constant':
            assert any(op.name=='llvm.mlir.constant' and 'value' in op.attributes
                and str(op.results[0].type)=='i16' and ir.IntegerAttr(op.attributes['value']).value==0x7f81
                for op in _bf_walk(module.operation))
        print('TRANSPORT VERIFIED')
    else:
        try:
            _bf_transport(module)
        except ValueError:
            assert str(module)==before
            print('REFUSED unchanged')
        else:
            raise AssertionError('unqualified ABI/region accepted')
''')
    result = subprocess.run([str(compiler_python()), str(runner), str(source), "positive" if change is None else change],
        capture_output=True, text=True, timeout=30,
        env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"))
    assert result.returncode == 0, result.stdout + result.stderr
    assert ("TRANSPORT VERIFIED" if change in (None, "constant") else "REFUSED unchanged") in result.stdout


@pytest.mark.parametrize("passes", [[], ["convert-scf-to-cf"] * 2, [MARKER, "convert-scf-to-cf"]])
def test_pipeline_requires_one_seam(passes):
    with pytest.raises(ValueError, match="one unmodified"):
        edit_pipeline(passes)


def test_pipeline_and_runner_keep_checked_boundary():
    original = ["canonicalize", "convert-cf-to-llvm", "convert-arith-to-llvm"]
    assert edit_pipeline(original) == [*original[:-1], "arith-expand{include-bf16=true}", original[-1], MARKER]
    source = "_run_stages(ctx, module, pipeline, 0)\n"
    assert bind_runner(source, selected=False) == source
    assert "BF16 transport marker was not consumed" in bind_runner(source, selected=True)
    with pytest.raises(ValueError, match="one native stage runner"):
        bind_runner("pass\n", selected=True)


def test_existing_cast_expansion_is_retained_once_and_late_expansion_refuses():
    expansion = "arith-expand{include-bf16=true}"
    original = [expansion, "convert-arith-to-llvm", "convert-cf-to-llvm"]
    assert edit_pipeline(original) == [*original, MARKER]
    for invalid in ([*original, expansion], ["convert-arith-to-llvm", expansion, "convert-cf-to-llvm"]):
        with pytest.raises(ValueError, match="must precede"):
            edit_pipeline(invalid)


@pytest.mark.skipif(not available(), reason="existing native MLIR required")
def test_default_native_feature_expands_arithmetic_before_word_transport(tmp_path):
    from merlin.llvmlower.bfloat_transport import FEATURE
    from merlin.llvmlower.pipeline import lower_to_llvm_ir
    source = '''module {func.func @forward(%input:memref<1xbf16>,%output:memref<1xf32>,%raw:memref<1xi16>) {
      %index=arith.constant 0:index
      %value=memref.load %input[%index]:memref<1xbf16>
      %expanded=arith.extf %value:bf16 to f32
      %word=arith.bitcast %value:bf16 to i16
      memref.store %expanded,%output[%index]:memref<1xf32>
      memref.store %word,%raw[%index]:memref<1xi16>
      return
    }}'''
    llvm = lower_to_llvm_ir(source, workdir=tmp_path, features=frozenset({FEATURE}), timeout=30)
    assert "fpext bfloat" not in llvm
    assert "define " in llvm
