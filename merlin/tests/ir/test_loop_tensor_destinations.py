"""Complete destination proofs at the real post-fusion native seam."""

import os
import subprocess

import pytest

from merlin.llvmlower.loop_tensor_destinations import FEATURE, MARKER, PRELUDE, bind_runner, edit_pipeline
from merlin.llvmlower.toolchain import available, compiler_python


def source(*, shape="4", init_read=False, opaque=False):
    empty = "%size" if shape == "?" else ""
    value = "%init" if init_read else "%one"
    payload = f"%v=arith.addf %in,{value}:f32"
    declaration = ""
    if opaque:
        declaration = "func.func private @unknown(f32)->f32"
        payload = "%v=func.call @unknown(%in):(f32)->f32"
    type_ = f"tensor<{shape}xf32>"
    return f"""module {{ {declaration}
      func.func @forward(%x:{type_},%bound:index) -> {type_} {{
        %c0=arith.constant 0:index
        %c1=arith.constant 1:index
        %one=arith.constant 1.0:f32
        %seven=arith.constant 7.0:f32
        %size=tensor.dim %x,%c0:{type_}
        %e=tensor.empty({empty}):{type_}
        %seed=linalg.fill ins(%seven:f32) outs(%e:{type_})->{type_}
        %result=scf.for %iv=%c0 to %bound step %c1 iter_args(%old=%x)->{type_} {{
          %next=linalg.generic {{indexing_maps=[affine_map<(i)->(i)>,affine_map<(i)->(i)>],iterator_types=["parallel"]}}
            ins(%old:{type_}) outs(%seed:{type_}) {{
            ^bb0(%in:f32,%init:f32):
              {payload}
              linalg.yield %v:f32
          }} -> {type_}
          scf.yield %next:{type_}
        }}
        return %result:{type_}
      }}
    }}"""


def rewrite(tmp_path, text):
    original, output, script = (tmp_path / name for name in ("original.mlir", "rewritten.mlir", "rewrite.py"))
    original.write_text(text)
    script.write_text("from torch_mlir import ir\nimport sys\n_run_stages = None\n" + PRELUDE + """
with ir.Context() as ctx:
    module = ir.Module.parse(open(sys.argv[1]).read())
    module.operation.verify()
    _ltd_reuse(ctx, module)
    open(sys.argv[2], 'w').write(str(module))
""")
    result = subprocess.run([str(compiler_python()), str(script), str(original), str(output)],
        check=True, capture_output=True, text=True, timeout=20,
        env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(tmp_path)))
    return result.stdout, output.read_text()


@pytest.mark.skipif(not available(), reason="existing native MLIR required")
@pytest.mark.parametrize("shape,init_read,opaque,changed", [
    ("4", False, False, 1), ("17", False, False, 1),
    ("4", True, False, 0), ("0", False, False, 0),
    ("?", False, False, 0), ("4", False, True, 0),
])
def test_only_complete_unobserved_init_destinations_change(tmp_path, shape, init_read, opaque, changed):
    stdout, text = rewrite(tmp_path, source(shape=shape, init_read=init_read, opaque=opaque))
    assert f"OK loop_tensor_destinations {changed}" in stdout
    assert "scf.for" in text


def test_explicit_feature_places_rewrite_after_fusion_before_bufferization():
    passes = ["canonicalize", "func.func(linalg-fuse-elementwise-ops)", "one-shot-bufferize{bufferize-function-boundaries}", "canonicalize"]
    edited = edit_pipeline(passes)
    assert edited[2:4] == [MARKER, passes[2]]
    original = "def _run_stages(*args): pass\n_run_stages(ctx, module, pipeline, 0)\n"
    assert bind_runner(original, selected=False) == original
    assert "_ltd_reuse" in bind_runner(original, selected=True)
    with pytest.raises(ValueError, match="one native stage runner call"):
        bind_runner("pass\n", selected=True)
    with pytest.raises(ValueError, match="one unmodified bufferization seam"):
        edit_pipeline([*passes, passes[2]])
    from merlin.llvmlower.impr_features import known
    assert FEATURE in known()
    from merlin.llvmlower.pipeline import _upstream_pipeline
    assert MARKER in _upstream_pipeline(frozenset({FEATURE}))


@pytest.mark.parametrize("pipeline", [
    "one-shot-bufferize{}", MARKER + ",canonicalize,one-shot-bufferize{}",
    MARKER + "," + MARKER + ",one-shot-bufferize{}",
    "one-shot-bufferize{}," + MARKER + ",one-shot-bufferize{}",
])
def test_missing_duplicate_or_moved_rewrite_seam_refuses(pipeline):
    scope = {"_run_stages": lambda *args: pytest.fail("unsafe pipeline reached native lowering")}
    exec(PRELUDE, scope)
    with pytest.raises(ValueError, match="marker|immediately precede"):
        scope["_run_stages"](None, None, pipeline, 0)
