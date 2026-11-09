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


def read_source(*, fresh=True, readonly=True, generalize=False, conditional=True, write=False, opaque=False, constant=False, while_state=None):
    marker = "read_only" if readonly else ""
    seed = "%seed" if fresh else "%x"
    declaration = "func.func private @observe()" if opaque else ""
    extra = "func.call @observe() : () -> ()" if opaque else ""
    store = "memref.store %v, %mutable[%i] : memref<4xf32>" if write else ""
    read = "%v = memref.load %view[%i] : memref<?xf32>"
    if conditional:
        read = f"""%yes=arith.cmpi slt,%i,%c2:index
          %v=scf.if %yes -> f32 {{
            %r=memref.load %view[%i]:memref<?xf32>
            scf.yield %r:f32
          }} else {{ {extra} scf.yield %zero:f32 }}"""
    fill = "%seed=linalg.fill ins(%zero:f32) outs(%empty:tensor<4xf32>)->tensor<4xf32>"
    if generalize:
        fill = """%seed=linalg.generic {indexing_maps=[affine_map<(i)->()>,affine_map<(i)->(i)>],iterator_types=["parallel"]}
          ins(%zero:f32) outs(%empty:tensor<4xf32>) {^bb0(%v:f32,%init:f32):linalg.yield %v:f32}->tensor<4xf32>"""
    if constant:
        fill = "%seed=arith.constant dense<0.0>:tensor<4xf32>"
    header = f"%result=scf.for %iv=%c0 to %bound step %c1 iter_args(%old={seed})->tensor<4xf32> {{"
    tail = "scf.yield %next:tensor<4xf32>"
    returned = "%result"
    if while_state is not None:
        forwarded = {"identity": "%before,%other_before", "computed": "%x,%other_before",
                     "permuted": "%other_before,%before"}[while_state]
        header = f"""%other_seed=arith.constant dense<9.0>:tensor<4xf32>
        %result:3=scf.while (%before={seed},%other_before=%other_seed,%iv_before=%c0)
          :(tensor<4xf32>,tensor<4xf32>,index)->(tensor<4xf32>,tensor<4xf32>,index) {{
          %continue=arith.cmpi slt,%iv_before,%bound:index
          scf.condition(%continue) {forwarded},%iv_before:tensor<4xf32>,tensor<4xf32>,index
        }} do {{ ^bb0(%old:tensor<4xf32>,%other:tensor<4xf32>,%iv:index):"""
        tail = "%next_iv=arith.addi %iv,%c1:index\nscf.yield %next,%other,%next_iv:tensor<4xf32>,tensor<4xf32>,index"
        returned = "%result#0"
    return f"""module {{ {declaration}
      func.func @forward(%x:tensor<4xf32>,%bound:index,%mutable:memref<4xf32>)->tensor<4xf32> {{
        %c0=arith.constant 0:index
        %c1=arith.constant 1:index
        %c2=arith.constant 2:index
        %zero=arith.constant 0.0:f32
        %base=bufferization.to_buffer %x {marker}:tensor<4xf32> to memref<4xf32>
        %view=memref.cast %base:memref<4xf32> to memref<?xf32>
        %empty=tensor.empty():tensor<4xf32>
        {fill}
        {header}
          %next=linalg.generic {{indexing_maps=[affine_map<(i)->(i)>,affine_map<(i)->(i)>],iterator_types=["parallel"]}}
            ins(%old:tensor<4xf32>) outs(%empty:tensor<4xf32>) {{^bb0(%in:f32,%init:f32):
              %i=linalg.index 0:index
              {read}
              {store}
              %sum=arith.addf %in,%v:f32
              linalg.yield %sum:f32
            }}->tensor<4xf32>
          {tail}
        }}
        return {returned}:tensor<4xf32>
      }}
    }}"""


@pytest.mark.skipif(not available(), reason="existing native MLIR required")
@pytest.mark.parametrize("options,changed", [({}, 1), ({"generalize": True}, 1), ({"constant": True}, 1), ({"conditional": False}, 1),
    ({"fresh": False}, 0), ({"readonly": False}, 0), ({"write": True}, 0), ({"opaque": True}, 0)])
def test_read_capture_requires_fresh_seed_and_closed_readonly_entry_origin(tmp_path, options, changed):
    stdout, output = rewrite(tmp_path, read_source(**options))
    assert f"OK loop_tensor_destinations {changed}" in stdout
    assert "memref.load" in output and "scf.for" in output
    if options.get("conditional", True):
        assert "scf.if" in output


@pytest.mark.skipif(not available(), reason="existing native MLIR required")
@pytest.mark.parametrize("state,options,changed", [
    ("identity", {}, 1), ("identity", {"generalize": True}, 1),
    ("identity", {"constant": True}, 1), ("identity", {"fresh": False}, 0),
    ("identity", {"readonly": False}, 0), ("identity", {"write": True}, 0),
    ("identity", {"opaque": True}, 0), ("computed", {}, 0), ("permuted", {}, 0),
])
def test_while_destinations_require_unmodified_condition_state(tmp_path, state, options, changed):
    stdout, output = rewrite(tmp_path, read_source(while_state=state, **options))
    assert f"OK loop_tensor_destinations {changed}" in stdout
    assert "scf.while" in output and "scf.condition" in output and "memref.load" in output


@pytest.mark.skipif(not available(), reason="existing native MLIR required")
def test_function_cfg_phi_is_not_an_independent_entry_tensor(tmp_path):
    source = '''module {
      func.func @forward(%bound:index)->tensor<4xf32> {
        %zero=arith.constant 0.0:f32
        %c0=arith.constant 0:index
        %c1=arith.constant 1:index
        %empty=tensor.empty():tensor<4xf32>
        %seed=linalg.fill ins(%zero:f32) outs(%empty:tensor<4xf32>)->tensor<4xf32>
        cf.br ^other(%seed:tensor<4xf32>)
      ^other(%alias:tensor<4xf32>):
        %base=bufferization.to_buffer %alias read_only:tensor<4xf32> to memref<4xf32>
        %result=scf.for %iv=%c0 to %bound step %c1 iter_args(%old=%seed)->tensor<4xf32> {
          %next=linalg.generic {indexing_maps=[affine_map<(i)->(i)>,affine_map<(i)->(i)>],iterator_types=["parallel"]}
            ins(%old:tensor<4xf32>) outs(%empty:tensor<4xf32>) {^bb0(%in:f32,%init:f32):
              %i=linalg.index 0:index
              %read=memref.load %base[%i]:memref<4xf32>
              %sum=arith.addf %in,%read:f32
              linalg.yield %sum:f32
            }->tensor<4xf32>
          scf.yield %next:tensor<4xf32>
        }
        return %result:tensor<4xf32>
      }
    }'''
    stdout, _ = rewrite(tmp_path, source)
    assert "OK loop_tensor_destinations 0" in stdout


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
