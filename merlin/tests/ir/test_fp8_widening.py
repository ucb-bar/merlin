"""FP8 cast lowering is explicit, transactional and independent of storage ABI."""

import subprocess

import pytest

from merlin.llvmlower.fp8_widening import MARKER, PRELUDE, SUPPORT, bind_runner, edit_pipeline
from merlin.llvmlower.toolchain import available, compiler_python

FORMATS = ("f8E4M3FN", "f8E4M3FNUZ", "f8E5M2", "f8E5M2FNUZ")


def rewrite(tmp_path, source, expected, *, casts=1):
    original, script = tmp_path / "source.mlir", tmp_path / "run.py"
    original.write_text(source)
    script.write_text(
        "from torch_mlir import ir\nimport sys\n"
        + SUPPORT
        + """
with ir.Context():
    module=ir.Module.parse(open(sys.argv[1]).read())
    before=str(module)
    if sys.argv[2]=='pass':
        assert _fp8_widen(module)==int(sys.argv[3])
        assert _fp8_widen(module)==0
        assert all(op.name!='arith.extf' or str(op.operands[0].type) not in _F8W_FORMATS for op in _f8w_walk(module.operation))
        assert 'prov.op = "widen"' in str(module)
        assert 'arith.bitcast' in str(module)
        module.operation.verify()
    else:
        try: _fp8_widen(module)
        except ValueError: assert str(module)==before
        else: raise AssertionError('unqualified FP8 operation was accepted')
    print('VERIFIED')
"""
    )
    process = subprocess.run(
        [str(compiler_python()), str(script), str(original), expected, str(casts)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    assert process.stdout.strip() == "VERIFIED"


@pytest.mark.skipif(not available(), reason="existing native compiler required")
@pytest.mark.parametrize("source", FORMATS)
@pytest.mark.parametrize("target", ["f16", "bf16", "f32", "f64"])
def test_all_formats_and_wider_float_targets_keep_source_annotations(tmp_path, source, target):
    program = f"""module {{func.func @widen(%x:{source})->{target} {{
      %r="arith.extf"(%x) {{prov.op="widen"}} : ({source})->{target}
      return %r:{target}
    }}}}"""
    rewrite(tmp_path, program, "pass")


@pytest.mark.skipif(not available(), reason="existing native compiler required")
@pytest.mark.parametrize(
    "change", ["arithmetic", "narrow", "fastmath", "property", "call", "vector", "tensor", "mixed"]
)
def test_unqualified_operations_refuse_before_any_edits(tmp_path, change):
    dtype = "vector<2xf8E4M3FN>" if change == "vector" else "tensor<2xf8E4M3FN>" if change == "tensor" else "f8E4M3FN"
    target = "vector<2xf32>" if change == "vector" else "tensor<2xf32>" if change == "tensor" else "f32"
    payload = {
        "arithmetic": "%r=arith.addf %x,%x:f8E4M3FN",
        "narrow": "%r=arith.truncf %y:f32 to f8E4M3FN",
        "fastmath": "%r=arith.extf %x fastmath<nnan>:f8E4M3FN to f32",
        "property": '%r="arith.extf"(%x) {unknown.policy="opaque"}:(f8E4M3FN)->f32',
        "call": "%r=func.call @unknown(%x):(f8E4M3FN)->f32",
        "mixed": "%good=arith.extf %x:f8E4M3FN to f32 %r=arith.addf %x,%x:f8E4M3FN",
    }.get(change, f"%r=arith.extf %x:{dtype} to {target}")
    declaration = "func.func private @unknown(f8E4M3FN)->f32" if change == "call" else ""
    program = f"module {{{declaration} func.func @f(%x:{dtype},%y:f32) {{{payload} return}}}}"
    rewrite(tmp_path, program, "refuse")


def named_program(source, target, batch=False, *, bad=False):
    prefix = "2x" if batch else ""
    a, b, c = f"tensor<{prefix}3x4x{source}>", f"tensor<{prefix}4x5x{target}>", f"tensor<{prefix}3x5xf32>"
    name = "linalg.batch_matmul" if batch else "linalg.matmul"
    payload = f"%bad=arith.addf %z,%z:{source}" if bad else ""
    return f"""module {{func.func @f(%a:{a},%b:{b},%c:{c},%z:{source})->{c} {{
      %r={name} {{prov.op="widen"}} ins(%a,%b:{a},{b}) outs(%c:{c})->{c}
      {payload}
      return %r:{c}
    }}}}"""


@pytest.mark.skipif(not available(), reason="existing native compiler required")
@pytest.mark.parametrize("source", FORMATS)
@pytest.mark.parametrize("batch", [False, True])
def test_named_contractions_generalize_only_in_selected_native_lowering(tmp_path, source, batch):
    rewrite(tmp_path, named_program(source, "f8E5M2", batch), "pass", casts=2)


@pytest.mark.skipif(not available(), reason="existing native compiler required")
def test_invalid_whole_module_refuses_before_generalizing_named_contraction(tmp_path):
    rewrite(tmp_path, named_program("f8E4M3FN", "f8E5M2", bad=True), "refuse")


@pytest.mark.skipif(not available(), reason="existing native compiler required")
@pytest.mark.parametrize("source", FORMATS)
def test_generalized_fill_keeps_scalar_fp8_transport_with_contractions(tmp_path, source):
    program = named_program(source, "f8E5M2").replace(
        "%r=linalg.matmul",
        f"%filled=linalg.fill ins(%z:{source}) outs(%a:tensor<3x4x{source}>)->tensor<3x4x{source}> %r=linalg.matmul",
    )
    rewrite(tmp_path, program, "pass", casts=2)


@pytest.mark.skipif(not available(), reason="existing native compiler required")
@pytest.mark.parametrize("call", [False, True])
def test_scalar_generic_transport_does_not_hide_unqualified_body_operations(tmp_path, call):
    body = "%v=func.call @unknown(%x):(f8E4M3FN)->f8E4M3FN" if call else "%v=arith.addf %x,%x:f8E4M3FN"
    generic = f"""%bad=linalg.generic {{indexing_maps=[affine_map<(i,j)->()>,affine_map<(i,j)->(i,j)>],iterator_types=["parallel","parallel"]}}
      ins(%z:f8E4M3FN) outs(%a:tensor<3x4xf8E4M3FN>) {{
        ^bb0(%x:f8E4M3FN,%unused:f8E4M3FN): {body} linalg.yield %v:f8E4M3FN
      }}->tensor<3x4xf8E4M3FN>"""
    program = named_program("f8E4M3FN", "f8E5M2").replace("%r=linalg.matmul", generic + " %r=linalg.matmul")
    if call:
        program = program.replace("module {", "module {func.func private @unknown(f8E4M3FN)->f8E4M3FN ")
    rewrite(tmp_path, program, "refuse")


def test_source_seam_preserves_all_callbacks_and_runs_once():
    calls, rewrites = [], []
    scope = {"_run_stages": lambda *args: calls.append(args)}
    exec(PRELUDE, scope)
    scope["_fp8_widen"] = lambda module: rewrites.append(module)
    ctx, module = object(), object()
    callbacks = (17, ("mid",), ("late",), ("post",), ("pre",))
    scope["_run_stages"](ctx, module, MARKER + ",canonicalize", *callbacks)
    assert rewrites == [module] and calls == [(ctx, module, "canonicalize", *callbacks)]
    with pytest.raises(ValueError):
        scope["_run_stages"](ctx, module, MARKER + ",canonicalize", *callbacks)
    with pytest.raises(ValueError):
        scope["_run_stages"](ctx, module, "canonicalize," + MARKER, *callbacks)
    assert edit_pipeline(["canonicalize"]) == [MARKER, "canonicalize"]
    with pytest.raises(ValueError):
        edit_pipeline([MARKER, "canonicalize"])
    source = "def _run_stages(*args): pass\n_run_stages(ctx,module,pipeline,0)\n"
    assert bind_runner(source, selected=False) == source
    assert "if not _F8W_CONSUMED" in bind_runner(source, selected=True)
    with pytest.raises(ValueError):
        bind_runner("", selected=True)


def test_fp8_and_bfloat_source_seams_compose_without_dropping_callbacks():
    from merlin.llvmlower.bfloat_elementwise import MARKER as BF_MARKER, PRELUDE as BF_PRELUDE

    calls, rewrites = [], []
    scope = {"_run_stages": lambda *args: calls.append(args)}
    exec(BF_PRELUDE, scope)
    exec(PRELUDE, scope)
    scope["_fp8_widen"] = lambda module: rewrites.append("fp8")
    scope["_bf_elementwise"] = lambda module: rewrites.append("bf16")
    callbacks = (17, ("mid",), ("late",), ("post",), ("pre",))
    ctx, module = object(), object()
    scope["_run_stages"](ctx, module, MARKER + "," + BF_MARKER + ",canonicalize", *callbacks)
    assert rewrites == ["fp8", "bf16"]
    assert calls == [(ctx, module, "canonicalize", *callbacks)]
