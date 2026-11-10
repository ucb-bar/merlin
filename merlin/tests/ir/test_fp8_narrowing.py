"""Standard FP8 truncation has an explicit source seam and no policy inference."""

import subprocess

import pytest

from merlin.llvmlower.fp8_narrowing import MARKER, PRELUDE, SUPPORT, bind_runner, edit_pipeline
from merlin.llvmlower.toolchain import available, compiler_python

FORMATS = ("f8E4M3FN", "f8E4M3FNUZ", "f8E5M2", "f8E5M2FNUZ")


def rewrite(tmp_path, source, expected):
    original, script = tmp_path / "source.mlir", tmp_path / "run.py"
    original.write_text(source)
    script.write_text(
        "from torch_mlir import ir, passmanager\nimport sys\n"
        + SUPPORT
        + """
with ir.Context():
    module=ir.Module.parse(open(sys.argv[1]).read())
    before=str(module)
    if sys.argv[2]=='pass':
        assert _fp8_narrow(module)==1
        assert _fp8_narrow(module)==0
        assert 'prov.op = "quantize"' in str(module)
        assert 'arith.shrui' in str(module)
        module.operation.verify()
        passmanager.PassManager.parse('builtin.module(arith-expand,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)').run(module.operation)
        module.operation.verify()
        assert 'unrealized_conversion_cast' not in str(module)
    else:
        try: _fp8_narrow(module)
        except ValueError: assert str(module)==before
        else: raise AssertionError('unqualified narrowing was accepted')
    print('VERIFIED')
"""
    )
    result = subprocess.run(
        [str(compiler_python()), str(script), str(original), expected], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "VERIFIED"


@pytest.mark.skipif(not available(), reason="existing native compiler required")
@pytest.mark.parametrize("source", ["f16", "bf16", "f32", "f64"])
@pytest.mark.parametrize("target", FORMATS)
def test_all_scalar_formats_reach_complete_llvm(tmp_path, source, target):
    program = f"""module {{func.func @cast(%x:{source})->{target} {{
      %r="arith.truncf"(%x) {{prov.op="quantize"}}:({source})->{target}
      return %r:{target}
    }}}}"""
    rewrite(tmp_path, program, "pass")


@pytest.mark.skipif(not available(), reason="existing native compiler required")
@pytest.mark.parametrize("change", ["tensor", "vector", "fastmath", "property", "mixed"])
def test_unqualified_candidates_refuse_before_edits(tmp_path, change):
    if change in ("tensor", "vector"):
        source, target = f"{change}<2xf32>", f"{change}<2xf8E4M3FN>"
        payload = f"%r=arith.truncf %x:{source} to {target}"
    else:
        source, target = "f32", "f8E4M3FN"
        attributes = "fastmath=#arith.fastmath<nnan>" if change in ("fastmath", "mixed") else "unknown=true"
        payload = f'%r="arith.truncf"(%x) {{{attributes}}}:(f32)->f8E4M3FN'
        if change == "mixed":
            payload = "%good=arith.truncf %x:f32 to f8E5M2 " + payload
    rewrite(tmp_path, f"module {{func.func @f(%x:{source}) {{{payload} return}}}}", "refuse")


def test_source_seam_runs_once_and_keeps_every_callback():
    calls, rewrites = [], []
    scope = {"_run_stages": lambda *args: calls.append(args)}
    exec(PRELUDE, scope)
    scope["_fp8_narrow"] = lambda module: rewrites.append(module)
    ctx, module = object(), object()
    callbacks = (17, ("mid",), ("late",), ("post",), ("pre",))
    scope["_run_stages"](ctx, module, MARKER + ",canonicalize", *callbacks)
    assert rewrites == [module]
    assert calls == [(ctx, module, "canonicalize", *callbacks)]
    with pytest.raises(ValueError):
        scope["_run_stages"](ctx, module, MARKER + ",canonicalize", *callbacks)
    with pytest.raises(ValueError):
        scope["_run_stages"](ctx, module, "canonicalize," + MARKER, *callbacks)
    assert edit_pipeline(["canonicalize"]) == [MARKER, "canonicalize"]
    with pytest.raises(ValueError):
        edit_pipeline([MARKER, "canonicalize"])
    source = "def _run_stages(*args): pass\n_run_stages(ctx,module,pipeline,0)\n"
    assert bind_runner(source, selected=False) == source
    assert "if not _F8N_CONSUMED" in bind_runner(source, selected=True)
    with pytest.raises(ValueError):
        bind_runner("", selected=True)


def test_narrow_then_widen_then_bfloat_keeps_callback_order():
    from merlin.llvmlower.fp8_widening import MARKER as WIDE, PRELUDE as WIDE_PRELUDE
    from merlin.llvmlower.bfloat_elementwise import MARKER as BF, PRELUDE as BF_PRELUDE

    calls, rewrites = [], []
    scope = {"_run_stages": lambda *args: calls.append(args)}
    for code in (BF_PRELUDE, WIDE_PRELUDE, PRELUDE):
        exec(code, scope)
    scope["_fp8_narrow"] = lambda module: rewrites.append("narrow")
    scope["_fp8_widen"] = lambda module: rewrites.append("widen")
    scope["_bf_elementwise"] = lambda module: rewrites.append("bfloat")
    ctx, module = object(), object()
    callbacks = (17, ("mid",), ("late",), ("post",), ("pre",))
    scope["_run_stages"](ctx, module, ",".join((MARKER, WIDE, BF, "canonicalize")), *callbacks)
    assert rewrites == ["narrow", "widen", "bfloat"]
    assert calls == [(ctx, module, "canonicalize", *callbacks)]
