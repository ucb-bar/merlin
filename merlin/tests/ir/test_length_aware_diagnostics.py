"""Literal diagnostics preserve bytes and refuse incompatible runtime ABIs."""

import os
import subprocess

import pytest

from merlin.llvmlower.length_aware_diagnostics import MARKER, SUPPORT, bind_runner, edit_pipeline
from merlin.llvmlower.toolchain import available, compiler_python


@pytest.mark.skipif(not available(), reason="existing native compiler environment required")
@pytest.mark.parametrize("change", ["bytes", "putchar-type", "abort-body", "early", "collision"])
def test_native_literal_lowering_and_unchanged_refusals(tmp_path, change):
    declarations = {
        "putchar-type": "llvm.func @putchar(i64) -> i64",
        "abort-body": "llvm.func @abort() { llvm.return }",
        "collision": "llvm.func @__merlin_assert_bytes_0() { llvm.return }",
    }
    source = (r'module { func.func @entry(%c:i1) { cf.assert %c,"before\00after λ" '
              r'{prov.source_ids=["source.assert"]} return } '
              + declarations.get(change, "") + "}")
    program = tmp_path / "program.mlir"
    program.write_text(source)
    runner = tmp_path / "run.py"
    runner.write_text("from torch_mlir import ir,passmanager\nimport sys\n" + SUPPORT + '''
with ir.Context():
    module=ir.Module.parse(open(sys.argv[1]).read())
    if sys.argv[2]!='early':
        passmanager.PassManager.parse('builtin.module(convert-func-to-llvm,convert-arith-to-llvm)').run(module.operation)
    before=str(module)
    if sys.argv[2] in ('bytes','collision'):
        assert _diag_lower(module)==1
        assert _diag_lower(module)==0
        module.operation.verify()
        assert not any(op.name=='cf.assert' for op in _diag_walk(module.operation))
        assert any(op.name=='llvm.call' and 'prov.source_ids' in op.attributes for op in _diag_walk(module.operation))
        globals=[op for op in module.body.operations if op.name=='llvm.mlir.global']
        assert len(globals)==1
        value=ir.StringAttr(globals[0].attributes['value']).value
        assert value=='before\\x00after λ\\n'
        assert str(ir.TypeAttr(globals[0].attributes['global_type']).value)=='!llvm.array<16 x i8>'
        if sys.argv[2]=='collision':
            assert 'callee = @__merlin_assert_bytes_1' in module.operation.get_asm(print_generic_op_form=True)
        passmanager.PassManager.parse('builtin.module(convert-cf-to-llvm,reconcile-unrealized-casts)').run(module.operation)
        module.operation.verify()
        print('VERIFIED')
    else:
        try:
            _diag_lower(module)
        except ValueError:
            assert str(module)==before
            print('REFUSED unchanged')
        else:
            raise AssertionError('incompatible runtime/boundary accepted')
''')
    result = subprocess.run([str(compiler_python()), str(runner), str(program), change], capture_output=True,
        text=True, timeout=30, env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"))
    assert result.returncode == 0, result.stdout + result.stderr
    assert ("VERIFIED" if change in ("bytes", "collision") else "REFUSED unchanged") in result.stdout


@pytest.mark.parametrize("passes", [[], ["convert-cf-to-llvm"] * 2, [MARKER, "convert-cf-to-llvm"]])
def test_requires_one_unmodified_native_seam(passes):
    with pytest.raises(ValueError, match="one unmodified"):
        edit_pipeline(passes)


def test_pipeline_and_runner_bind_the_explicit_feature():
    original = ["convert-arith-to-llvm", "convert-cf-to-llvm", "reconcile-unrealized-casts"]
    assert edit_pipeline(original) == [original[0], MARKER, *original[1:]]
    source = "_run_stages(ctx, module, pipeline, 0)\n"
    assert bind_runner(source, selected=False) == source
    assert "literal diagnostic marker was not consumed" in bind_runner(source, selected=True)
    with pytest.raises(ValueError, match="one native stage runner"):
        bind_runner("pass\n", selected=True)
