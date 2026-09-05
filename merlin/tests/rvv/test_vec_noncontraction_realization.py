"""The REALIZATION of the non-contraction vectorize lever: the tile it vectorizes must not be
copied onto itself once per tile.

``vectorize_non_contraction_generics`` tiles each all-parallel ``linalg.generic`` on TENSORS and
vectorizes the tile, so every tile ends in a ``tensor.insert_slice`` back into the loop-carried
destination. Bufferization realizes that destructive update as: a subview of the destination, a
``vector.transfer_write`` into it, a SECOND structurally identical subview, and a ``memref.copy``
between the two -- the tile copied onto ITSELF, inside the innermost loop, per ``lanes`` elements.
It reaches the emitted code as ``call void @llvm.memcpy(ptr %x, ptr %x, i64 <lanes*4>)`` immediately
after the vector store.

That is the whole realization cost the lever's 1.28x regression was paying, and it is why counting
``@memrefCopy`` call sites missed it: these copies are contiguous in the innermost dim, so they
lower to ``llvm.memcpy``, not to the rank-generic runtime helper.

The fix is that the lever now IMPLIES ``erase_self_copy`` (which is also what makes
``lower_to_llvm_ir`` splice the post-bufferization ``canonicalize,cse`` that collapses the two
subviews into the one SSA value the erase keys on). These tests pin BOTH halves: the closure, and
the emitted code -- with a CONTROL that runs the identical schedule WITHOUT the feature set, so a
passing treatment is attributable to the implication rather than to the arms.

Deliberately NOT asserted here: that the lever is now a speedup. That is a board measurement.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from merlin.common.paths import repo_root
from merlin.llvmlower import lower as _lower_mod  # noqa: F401  (registers runner-gated features)
from merlin.llvmlower import impr_features as impr
from merlin.llvmlower.selfcopy import FEATURE as SELF_COPY
from merlin.llvmlower.toolchain import available as _toolchain_available

#: An all-parallel elementwise generic tagged exactly as the prepare pass tags one: rank 2, no
#: gather in the body, no transcendental, innermost extent a whole multiple of the lane count.
_TAGGED_RELU = """
#par = affine_map<(d0, d1) -> (d0, d1)>
func.func @f(%a: tensor<4x64xf32>) -> tensor<4x64xf32> {
  %z = arith.constant 0.000000e+00 : f32
  %e = tensor.empty() : tensor<4x64xf32>
  %r = linalg.generic {indexing_maps = [#par, #par],
                       iterator_types = ["parallel", "parallel"]}
      ins(%a : tensor<4x64xf32>) outs(%e : tensor<4x64xf32>)
      attrs = {merlin.vec_r2} {
  ^bb0(%in: f32, %out: f32):
    %m = arith.maximumf %in, %z : f32
    linalg.yield %m : f32
  } -> tensor<4x64xf32>
  return %r : tensor<4x64xf32>
}
"""


# ---------------------------------------------------------------------------------------------
# 1. the closure -- cheap, always runs
# ---------------------------------------------------------------------------------------------

def test_the_lever_implies_the_self_copy_erase():
    """Without this, everyone who names the lever in ``compiler_features`` gets the cancelled
    version: the vector win paid back as a memcpy per tile."""
    assert SELF_COPY in impr.normalize([impr.VEC_NONCONTRACTION_NAME])


@pytest.mark.parametrize("lanes", [8, 16, 32, 64])
def test_every_lane_width_point_implies_it_too(lanes):
    """The width is a knob space; a width the search picks must be measured in the same
    realization as the default one, or the search compares two different things."""
    name = impr.ensure_vec_noncontraction(lanes)
    assert SELF_COPY in impr.normalize([name])


def test_the_baseline_is_untouched():
    """Default-off stays default-off: an empty feature set implies nothing and edits nothing."""
    assert impr.normalize(None) == frozenset()
    assert impr.normalize([]) == frozenset()
    from merlin.llvmlower.pipeline import RVV_TRANSFORM_SCHEDULE
    assert impr.apply_schedule(RVV_TRANSFORM_SCHEDULE, frozenset()) == RVV_TRANSFORM_SCHEDULE
    assert SELF_COPY not in impr.normalize([])


def test_the_hygiene_is_named_once():
    """One definition for the two registration sites (the bare name and every lane-width point),
    so they cannot drift apart -- the same reason ``_tile_epilogue_hygiene`` exists."""
    assert impr._vec_noncontraction_hygiene() == frozenset({SELF_COPY})


# ---------------------------------------------------------------------------------------------
# 2. the emitted code -- does the per-tile self-copy actually go away?
# ---------------------------------------------------------------------------------------------

def _self_memcpy_sizes(ll_text: str) -> list[str]:
    """Byte sizes of every ``llvm.memcpy`` whose SOURCE and DESTINATION are the same SSA value.

    Parsed structurally (``split``/``strip``, no pattern matching): a copy of a buffer onto itself
    is identified by operand identity, which is what makes erasing it value-preserving.
    """
    out: list[str] = []
    for line in ll_text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("call void @llvm.memcpy"):
            continue
        args = stripped.split("(", 1)[1].rsplit(")", 1)[0].split(",")
        if len(args) < 3:
            continue
        dst, src, size = (a.strip() for a in args[:3])
        if dst == src:
            out.append(size)
    return out


def _lower(text: str, sched: str, features: frozenset[str], tmp_path) -> str:
    from merlin.llvmlower.pipeline import lower_to_llvm_ir
    work = Path(tempfile.mkdtemp(prefix="vecrealize_", dir=str(tmp_path)))
    return lower_to_llvm_ir(text, workdir=work, vectorize=True,
                            transform_schedule=sched, features=features)


@pytest.mark.skipif(not _toolchain_available(), reason="m2m venv / clang not configured")
def test_the_vectorized_tile_is_not_copied_onto_itself(tmp_path):
    """CONTROL vs TREATMENT on the IDENTICAL transform schedule.

    ``apply_schedule`` is a no-op on a schedule that already carries the arms, so the two arms below
    differ ONLY in the feature set -- which is what makes the difference attributable to the
    implication and not to the tiling.
    """
    from merlin.llvmlower.pipeline import RVV_TRANSFORM_SCHEDULE
    armed = impr.apply_schedule(RVV_TRANSFORM_SCHEDULE,
                                impr.normalize([impr.VEC_NONCONTRACTION_NAME]))
    assert "merlin.vec_r2" in armed
    assert impr.apply_schedule(armed, impr.normalize([impr.VEC_NONCONTRACTION_NAME])) == armed

    control = _lower(_TAGGED_RELU, armed, frozenset(), tmp_path)
    treated = _lower(_TAGGED_RELU, armed, frozenset([impr.VEC_NONCONTRACTION_NAME]), tmp_path)

    # The control shows the realization cost: one self-copy, sized exactly one vector tile.
    tile_bytes = f"i64 {impr.VEC_NONCONTRACTION_LANES * 4}"
    assert _self_memcpy_sizes(control) == [tile_bytes], _self_memcpy_sizes(control)
    # ...and the lever, as registered, emits none.
    assert _self_memcpy_sizes(treated) == [], _self_memcpy_sizes(treated)

    # The vector win is KEPT -- the copy is what went away, not the vectorization.
    for arm in (control, treated):
        assert arm.count(f"load <{impr.VEC_NONCONTRACTION_LANES} x float>") >= 1
        assert arm.count(f"store <{impr.VEC_NONCONTRACTION_LANES} x float>") >= 1
    # ...and it is not paid for with new memory traffic: total copies and allocations both fall.
    assert (treated.count("call void @llvm.memcpy")
            < control.count("call void @llvm.memcpy"))
    assert treated.count("call ptr @malloc") < control.count("call ptr @malloc")


@pytest.mark.skipif(not _toolchain_available(), reason="m2m venv / clang not configured")
def test_an_untagged_module_is_unaffected_by_the_lever(tmp_path):
    """The arms match by attribute; a module carrying none must lower identically with the feature
    on or off, so nothing here can be attributed to the implied erase firing on unrelated code."""
    from merlin.llvmlower.pipeline import RVV_TRANSFORM_SCHEDULE
    untagged = _TAGGED_RELU.replace("\n      attrs = {merlin.vec_r2}", "")
    assert "merlin.vec_r2" not in untagged
    off = _lower(untagged, RVV_TRANSFORM_SCHEDULE, frozenset(), tmp_path)
    on = _lower(untagged, RVV_TRANSFORM_SCHEDULE,
                frozenset([impr.VEC_NONCONTRACTION_NAME]), tmp_path)
    assert _self_memcpy_sizes(off) == [] and _self_memcpy_sizes(on) == []
    assert off == on


# ---------------------------------------------------------------------------------------------
# 3. where the cost comes from -- the tiling on tensors, before any of the above
# ---------------------------------------------------------------------------------------------

def _mlir_opt() -> Path:
    """``mlir-opt`` of the same standalone LLVM install the pipeline's translate comes from --
    derived from the resolved tool, never a hardcoded path."""
    from merlin.llvmlower.toolchain import mlir_translate
    return mlir_translate().parent / "mlir-opt"


@pytest.mark.skipif(not _mlir_opt().is_file(), reason="standalone mlir-opt not present")
def test_the_tiling_on_tensors_is_what_emits_the_copy(tmp_path):
    """LOCALISATION. Run the RVV pipeline as far as bufferization and read the loop body: the
    destination slice is written by the vector store and then copied onto itself.

    This is the evidence the fix is aimed at the right thing. It fails if the realization stops
    producing the copy for some other reason -- at which point the implication above is redundant
    and this test says so.
    """
    import subprocess

    from merlin.llvmlower.pipeline import RVV_TRANSFORM_SCHEDULE, build_rvv_pipeline

    armed = impr.apply_schedule(RVV_TRANSFORM_SCHEDULE,
                                impr.normalize([impr.VEC_NONCONTRACTION_NAME]))
    sched = tmp_path / "sched.mlir"
    sched.write_text(armed, encoding="utf-8")
    pipeline = build_rvv_pipeline(sched)

    # Top-level split of the pass list (brace-aware), truncated after bufferization + the
    # canonicalize/cse the erase depends on -- the window in which the copy is literal.
    passes, depth, cur = [], 0, []
    for ch in pipeline:
        if ch in "{(":
            depth += 1
        elif ch in "})":
            depth -= 1
        if ch == "," and depth == 0:
            passes.append("".join(cur))
            cur = []
            continue
        cur.append(ch)
    passes.append("".join(cur))
    stop = max(i for i, p in enumerate(passes) if "buffer-loop-hoisting" in p)
    prefix = ",".join(passes[:stop + 1] + ["canonicalize", "cse"])

    src = tmp_path / "in.mlir"
    src.write_text(_TAGGED_RELU, encoding="utf-8")
    out = tmp_path / "out.mlir"
    proc = subprocess.run([str(_mlir_opt()), str(src),
                           f"--pass-pipeline=builtin.module({prefix})", "-o", str(out)],
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr
    text = out.read_text(encoding="utf-8")

    copies = [ln.strip() for ln in text.splitlines() if "memref.copy" in ln]
    assert copies, "the tile-and-vectorize-on-tensors realization emitted no copy at all"
    selfies = []
    for ln in copies:
        operands = ln.split("memref.copy", 1)[1].split(":", 1)[0].split(",")
        if len(operands) == 2 and operands[0].strip() == operands[1].strip():
            selfies.append(ln)
    assert selfies, f"no self-copy among {copies}"
    # ...and it is INSIDE the loop nest, not a prologue: the copied slice is the tile.
    assert f"1x{impr.VEC_NONCONTRACTION_LANES}xf32" in selfies[0], selfies[0]


def test_paths_come_from_the_helper():
    """Location-independence: this file must survive being moved."""
    assert (repo_root() / "merlin" / "tests" / "rvv").is_dir()
