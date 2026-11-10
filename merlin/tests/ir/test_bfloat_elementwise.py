"""Scalar BF16 arithmetic promotion preserves storage and fails closed."""

import subprocess

import pytest

from merlin.llvmlower.bfloat_elementwise import MARKER,SUPPORT,PRELUDE,bind_runner,edit_pipeline
from merlin.llvmlower.toolchain import available,compiler_python


@pytest.mark.skipif(not available(),reason="existing native compiler environment required")
@pytest.mark.parametrize("change",["positive","sqrt","log","mixed-log","add","vector","call","fastmath"])
def test_native_emulation_and_unchanged_refusals(tmp_path,change):
    expressions = {
        "positive":"%r=arith.divf %x,%y:bf16",
        "sqrt":"%r=math.sqrt %x:bf16",
        "log":"%r=math.log %x:bf16",
        "mixed-log":"%good=arith.divf %x,%y:bf16 %r=math.log %good:bf16",
        "add":"%r=arith.addf %x,%y:bf16",
        "vector":"%r=arith.divf %x,%y:vector<2xbf16>",
        "call":"%r=func.call @unknown(%x):(bf16)->bf16",
        "fastmath":"%r=arith.divf %x,%y fastmath<nnan>:bf16",
    }
    dtype = "vector<2xbf16>" if change == "vector" else "bf16"
    source = f'''module {{func.func @f(%x:{dtype},%y:{dtype})->{dtype} {{
        {expressions[change]}
        return %r:{dtype}
    }}''' + ("func.func private @unknown(bf16)->bf16" if change == "call" else "") + "}"
    runner,program = tmp_path / "run.py",tmp_path / "source.mlir"
    program.write_text(source)
    runner.write_text("from torch_mlir import ir\nimport sys\n"+SUPPORT+'''
with ir.Context():
    module=ir.Module.parse(open(sys.argv[1]).read())
    before=str(module)
    if sys.argv[2] in ('positive','sqrt'):
        assert _bf_elementwise(module)==1
        assert _bf_elementwise(module)==0
        assert 'arith.extf' in str(module) and 'arith.truncf' in str(module)
        assert '-> bf16' in str(module)
        for op in _bfe_walk(module.operation):
            if op.name in ('arith.divf','math.sqrt'):
                assert all(str(value.type)=='f32' for value in (*op.operands,*op.results))
        module.operation.verify()
    else:
        try: _bf_elementwise(module)
        except ValueError: assert str(module)==before
        else: raise AssertionError('unqualified BF16 arithmetic was accepted')
    print('VERIFIED')
''')
    process = subprocess.run([str(compiler_python()),str(runner),str(program),change],capture_output=True,text=True,timeout=30)
    assert process.returncode == 0,process.stdout+process.stderr
    assert process.stdout.strip() == "VERIFIED"


def test_feature_seam_and_runner_are_explicit():
    passes = ["canonicalize","one-shot-bufferize","convert-arith-to-llvm"]
    assert edit_pipeline(passes) == [MARKER,*passes]
    for invalid in ([MARKER,*passes],["arith-emulate-unsupported-floats{source-types=bf16}",*passes]):
        with pytest.raises(ValueError): edit_pipeline(invalid)
    source = "def _run_stages(*args): pass\n_run_stages(ctx,module,pipeline,0)\n"
    assert bind_runner(source,selected=False) == source
    assert "if not _BFE_CONSUMED" in bind_runner(source,selected=True)
    with pytest.raises(ValueError): bind_runner("",selected=True)


def test_source_seam_forwards_every_ownership_and_runner_callback():
    calls,emulations = [],[]
    scope = {"_run_stages":lambda *args:calls.append(args)}
    exec(PRELUDE,scope)
    scope["_bf_elementwise"] = lambda module:emulations.append(module)
    ctx,module = object(),object()
    callbacks = (17,("mid",),("late",),("post",),("pre",))
    scope["_run_stages"](ctx,module,MARKER+",canonicalize",*callbacks)
    assert emulations == [module] and calls == [(ctx,module,"canonicalize",*callbacks)]
    assert scope["_BFE_CONSUMED"] is True
    with pytest.raises(ValueError):
        scope["_run_stages"](ctx,module,"canonicalize,"+MARKER,*callbacks)
