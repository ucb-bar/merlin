"""Where the two pass-order defects of the RVV lowering are pinned.

Both are COVERAGE defects -- a stage that is enabled, reports as applied, and cannot see most of
the ops it exists to act on. Neither shows up as a failure; both show up as a lever that does not
pay, which is the failure mode this pipeline keeps re-learning.

1. ``linalg-specialize-generic-ops`` EATS THE VECTORIZE TAG. It has to run before the transform
   interpreter so the contraction arms can match named ``linalg.matmul``/``batch_matmul``. On the
   way it rewrites most of the all-parallel generics the prepare pass tagged ``merlin.vec_r{rank}``
   into named ops (``linalg.broadcast`` above all), and a rewrite does not carry a DISCARDABLE
   attribute onto the op it produces. MEASURED on the int8 recaptures: of 93 / 107 / 189 ops tagged
   by the prepare pass, 15 / 33 / 36 reached an arm. ``vectorize_non_contraction_generics`` was
   therefore measured at 1.28x SLOWER while doing a fifth of the intended work and paying all of
   the loop overhead. Fixed by PLACEMENT -- the arms run from their own preloaded library, before
   specialization, while the tags are still on the ops.

2. ``linalg-fuse-elementwise-ops`` CANNOT SEE A NAMED OP. Upstream's fusion pattern is an
   ``OpRewritePattern<linalg::GenericOp>`` whose producer must also be a ``GenericOp``, so a
   ``linalg.broadcast`` is never matched -- and the fusion stage runs BEFORE the generalization that
   would turn it back into one. On the fixture below the fusion feature is literally inert: the
   emitted module is byte-identical to the arm that never enabled it.

Deliberately NOT asserted here: that either change is a speedup. Both are default-off placements
whose payoff is a board measurement.
"""
from __future__ import annotations

import ctypes
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from merlin.common.paths import artifacts_dir
from merlin.llvmlower import lower as _lower_mod  # noqa: F401  (registers runner-gated features)
from merlin.llvmlower import impr_features as impr
from merlin.llvmlower import pipeline as P
from merlin.llvmlower.toolchain import available as _toolchain_available

_needs_m2m = pytest.mark.skipif(not _toolchain_available(),
                                reason="m2m venv / clang not configured")

#: The whole defect in nine lines: a tagged all-parallel generic that IS a broadcast (so
#: ``linalg-specialize-generic-ops`` renames it and drops the tag) feeding an elementwise consumer
#: that specialization also renames (so the fusion stage cannot see the pair either).
TAGGED_BROADCAST = """
#b = affine_map<(d0, d1) -> (d1)>
#p = affine_map<(d0, d1) -> (d0, d1)>
func.func @forward(%s: tensor<64xf32>, %a: tensor<4x64xf32>) -> tensor<4x64xf32> {
  %e = tensor.empty() : tensor<4x64xf32>
  %bc = linalg.generic {indexing_maps = [#b, #p], iterator_types = ["parallel", "parallel"]}
      ins(%s : tensor<64xf32>) outs(%e : tensor<4x64xf32>) attrs = {merlin.vec_r2} {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<4x64xf32>
  %e2 = tensor.empty() : tensor<4x64xf32>
  %r = linalg.generic {indexing_maps = [#p, #p, #p], iterator_types = ["parallel", "parallel"]}
      ins(%a, %bc : tensor<4x64xf32>, tensor<4x64xf32>) outs(%e2 : tensor<4x64xf32>) {
  ^bb0(%x: f32, %y: f32, %o: f32):
    %m = arith.mulf %x, %y : f32
    linalg.yield %m : f32
  } -> tensor<4x64xf32>
  return %r : tensor<4x64xf32>
}
"""

VEC = impr.VEC_NONCONTRACTION_NAME
FUSE = impr.FUSE_ELEMENTWISE_NAME
SPECIALIZE = "func.func(linalg-specialize-generic-ops)"
VEC_INTERP = f"transform-interpreter{{entry-point={P.VEC_PRE_ENTRY}}}"
MAIN_INTERP = "transform-interpreter{entry-point=__transform_main}"


def _passes(features=frozenset(), **kw) -> list[str]:
    """The pass list as ``lower_to_llvm_ir`` builds it: the pre-library exists exactly when
    :func:`vec_pre_schedule` returns one, which is the same decision the driver makes."""
    feats = impr.normalize(features)
    kw.setdefault("vec_sched_path", "/VEC" if P.vec_pre_schedule(feats) else None)
    return P.build_rvv_pipeline("/SCHED", features=feats, **kw).split(",")


def _lower(text: str, features, tmp_path, **kw) -> str:
    work = Path(tempfile.mkdtemp(prefix="vectag_", dir=str(tmp_path)))
    return P.lower_to_llvm_ir(text, workdir=work, vectorize=True,
                              features=impr.normalize(features), **kw)


# ---------------------------------------------------------------------------------------------
# 1. the frozen baseline -- nothing here may reach a build that did not ask for it
# ---------------------------------------------------------------------------------------------

def test_the_lever_off_pipeline_carries_no_pre_specialization_stage():
    """The whole placement is conditional on the lever. With it off the pass string is the one
    HEAD builds: specialization first, then the single preload, then the one interpreter."""
    passes = _passes()
    assert P.VEC_PRE_ENTRY not in ",".join(passes)
    assert passes.count(MAIN_INTERP) == 1
    assert passes.index(SPECIALIZE) < next(i for i, p in enumerate(passes)
                                           if p.startswith("transform-preload-library"))
    assert P.vec_pre_schedule(frozenset()) is None


def test_the_reorder_is_default_off_and_changes_nothing_unasked(monkeypatch):
    monkeypatch.delenv("MERLIN_GENERALIZE_BEFORE_FUSE", raising=False)
    assert not P._generalize_before_fuse()
    monkeypatch.setenv("MERLIN_FUSE_POST", "1")
    passes = _passes()
    assert passes.index(P._FUSE_ELEMENTWISE) < passes.index(P._GENERALIZE_NAMED)


# ---------------------------------------------------------------------------------------------
# 2. blocker 2 -- the arms run before the pass that eats their tag
# ---------------------------------------------------------------------------------------------

def test_the_arms_run_before_specialization_when_the_lever_is_on():
    passes = _passes({VEC})
    assert passes.index(VEC_INTERP) < passes.index(SPECIALIZE) < passes.index(MAIN_INTERP)
    # the contraction arms still run exactly once, and still after specialization
    assert passes.count(MAIN_INTERP) == 1


def test_the_pre_library_is_preloaded_alongside_the_package_schedule():
    """One preload op, both libraries: the package's own schedule -- and every impr_features edit
    that anchors on ``__transform_main`` -- has to compose unchanged."""
    text = ",".join(_passes({VEC}, vec_sched_path="/VEC"))
    head, _, tail = text.partition("transform-preload-library{transform-library-paths=")
    assert "transform-preload-library" not in head and "transform-preload-library" not in tail, \
        "the pre-library must join the existing preload, not add a second one"
    libs = tail.partition("}")[0].split(",")
    assert libs == ["/VEC", "/SCHED"], libs


def test_the_escape_hatch_rebuilds_the_old_placement(monkeypatch):
    """``MERLIN_NO_DEALLOC``'s reason: this changes the emitted code of every build that names the
    lever, so both arms must be buildable or no claim about it is defensible."""
    monkeypatch.setenv("MERLIN_VEC_AFTER_SPECIALIZE", "1")
    assert P.vec_pre_schedule(impr.normalize([VEC])) is None
    assert P.VEC_PRE_ENTRY not in ",".join(_passes({VEC}))


def test_the_pre_library_arms_come_from_the_same_generator_as_the_package_ones():
    """The tagging predicate and the arms have to agree on the lane count; generating both from
    ``apply_schedule`` at the width the feature names is what keeps them from drifting."""
    for lanes in (8, 16, 32):
        name = impr.ensure_vec_noncontraction(lanes)
        pre = P.vec_pre_schedule(impr.normalize([name]))
        armed = impr.apply_schedule(P.RVV_TRANSFORM_SCHEDULE, impr.normalize([name]))
        for rank in (2, 3, 4):
            arm = f"transform.structured.vectorize %gt{rank} vector_sizes ["
            assert arm in pre and arm in armed
        assert f"tile_sizes [1, {lanes}]" in pre
        assert f"tile_sizes [1, {lanes}]" in armed


def test_a_skeleton_the_arms_cannot_splice_into_fails_closed(monkeypatch):
    """"Enabled and changed nothing" is the failure this file keeps re-learning. If the anchor the
    splice keys on ever moves, the build must stop, not preload an empty entry point."""
    monkeypatch.setattr(P, "_VEC_PRE_SKELETON", "module {}\n")
    with pytest.raises(P.PipelineError):
        P.vec_pre_schedule(impr.normalize([VEC]))


@_needs_m2m
def test_specialization_is_what_drops_the_tag():
    """LOCALISATION, on the IR itself. Not "the tag went missing somewhere" -- this pass, this op."""
    before = P.apply_passes(TAGGED_BROADCAST, "canonicalize,cse")
    after = P.apply_passes(TAGGED_BROADCAST, f"canonicalize,cse,{SPECIALIZE}")
    assert before.count("merlin.vec_r2") == 1
    assert after.count("merlin.vec_r2") == 0
    assert after.count("linalg.broadcast") == 1, after


@_needs_m2m
def test_a_tag_specialization_would_eat_now_reaches_an_arm(tmp_path, monkeypatch):
    """The deliverable, end to end: THREE arms on one fixture.

    off            -- no vector op at all (the baseline is untouched)
    on, old place  -- STILL no vector op: the tag was gone before the arm looked for it
    on, new place  -- the tagged op is tiled and vectorized at the lane width
    """
    monkeypatch.delenv("MERLIN_VEC_AFTER_SPECIALIZE", raising=False)
    off = _lower(TAGGED_BROADCAST, frozenset(), tmp_path)
    new = _lower(TAGGED_BROADCAST, {VEC}, tmp_path)
    monkeypatch.setenv("MERLIN_VEC_AFTER_SPECIALIZE", "1")
    old = _lower(TAGGED_BROADCAST, {VEC}, tmp_path)

    lanes = impr.VEC_NONCONTRACTION_LANES
    assert off.count("load <") == 0 and off.count("store <") == 0
    assert old.count("load <") == 0 and old.count("store <") == 0, \
        "the old placement is supposed to reach nothing on this fixture"
    assert new.count(f"load <{lanes} x float>") >= 1, new.count("load <")
    assert new.count(f"store <{lanes} x float>") >= 1
    # ...and not by copying the tile onto itself, which is the realization the lever already pays for
    assert new.count("call void @llvm.memcpy") == 0


# ---------------------------------------------------------------------------------------------
# 3. blocker 1 -- the fusion stage cannot see a named op
# ---------------------------------------------------------------------------------------------

def test_the_reorder_moves_generalization_in_front_of_every_fusion_stage(monkeypatch):
    monkeypatch.setenv("MERLIN_GENERALIZE_BEFORE_FUSE", "1")
    for passes in (P._upstream_pipeline().split(","), P._parallel_pipeline().split(",")):
        assert passes.index(P._GENERALIZE_NAMED) < passes.index(P._FUSE_ELEMENTWISE)


def test_the_reorder_reaches_the_feature_driven_stage_not_only_the_literal_one(monkeypatch):
    """``fuse_elementwise_post_contraction`` splices its own copy of the stage through
    ``impr_features.apply_pipeline``, anchored on the generalization. A reorder written into the
    literal pass list would miss it entirely and the feature would keep the blind spot."""
    monkeypatch.delenv("MERLIN_FUSE_POST", raising=False)
    monkeypatch.setenv("MERLIN_GENERALIZE_BEFORE_FUSE", "1")
    passes = _passes({FUSE})
    assert P._FUSE_ELEMENTWISE in passes
    assert passes.index(P._GENERALIZE_NAMED) < passes.index(P._FUSE_ELEMENTWISE)
    # the cleanup stays attached to the fusion it cleans up after
    assert passes[passes.index(P._FUSE_ELEMENTWISE) + 1:
                  passes.index(P._FUSE_ELEMENTWISE) + 3] == ["canonicalize", "cse"]


def test_the_reorder_never_crosses_the_transform_interpreter():
    """A generalization hoisted in front of the schedule leaves ``ops{["linalg.matmul"]}`` nothing
    to match -- a silent 0-vectorization, which is the failure the current order exists to avoid."""
    with pytest.raises(ValueError):
        P._reorder_generalize_before_fuse(
            [P._FUSE_ELEMENTWISE, MAIN_INTERP, P._GENERALIZE_NAMED])


def test_the_reorder_is_a_no_op_where_there_is_nothing_to_reorder():
    assert P._reorder_generalize_before_fuse(["canonicalize"]) == ["canonicalize"]
    already = [P._GENERALIZE_NAMED, P._FUSE_ELEMENTWISE]
    assert P._reorder_generalize_before_fuse(already) == already


@_needs_m2m
def test_the_fusion_feature_is_inert_until_the_reorder(tmp_path, monkeypatch):
    """The blind spot, measured on the emitted module rather than argued from upstream's source.

    With the stage in its current position the fusion feature produces a module BYTE-IDENTICAL to
    the arm that never enabled it -- both operands it would fuse are named ops it cannot match. With
    the generalization moved in front, the broadcast's materialization disappears: two stores of the
    tensor become one.
    """
    monkeypatch.delenv("MERLIN_GENERALIZE_BEFORE_FUSE", raising=False)
    plain = _lower(TAGGED_BROADCAST, frozenset(), tmp_path)
    fused = _lower(TAGGED_BROADCAST, {FUSE}, tmp_path)
    monkeypatch.setenv("MERLIN_GENERALIZE_BEFORE_FUSE", "1")
    reordered = _lower(TAGGED_BROADCAST, {FUSE}, tmp_path)

    assert fused == plain, "the fusion stage is supposed to match nothing here"
    assert plain.count("store float") == 2
    assert reordered.count("store float") == 1, reordered.count("store float")


@_needs_m2m
def test_the_reorder_does_not_change_the_numbers(tmp_path, monkeypatch):
    """A pass reordering has no business changing results. Run both arms and compare bitwise --
    each in the same process is safe here because the two libraries are loaded LOCAL, not global."""
    from merlin.llvmlower.abi import HostModel
    from merlin.llvmlower.codegen import build_host_shared

    from merlin.llvmlower.passes_xdsl import preprocess_text_textual
    ciface, _ = preprocess_text_textual(TAGGED_BROADCAST)   # adds llvm.emit_c_interface

    rng = np.random.default_rng(0)
    s = rng.standard_normal(64, dtype=np.float32)
    a = rng.standard_normal((4, 64), dtype=np.float32)
    expect = a * s

    got = {}
    for tag, on in (("off", False), ("on", True)):
        if on:
            monkeypatch.setenv("MERLIN_GENERALIZE_BEFORE_FUSE", "1")
        else:
            monkeypatch.delenv("MERLIN_GENERALIZE_BEFORE_FUSE", raising=False)
        work = Path(tempfile.mkdtemp(prefix=f"num_{tag}_", dir=str(tmp_path)))
        ll = work / "model.ll"
        ll.write_text(_lower(ciface, {FUSE}, work), encoding="utf-8")
        so = build_host_shared(ll, work / f"model_{tag}.so")
        out = np.zeros((4, 64), dtype=np.float32)
        bufs = [(s.ctypes.data, [64]), (a.ctypes.data, [4, 64]), (out.ctypes.data, [4, 64])]
        HostModel.load(str(so), n_args=len(bufs))(bufs)
        got[tag] = out.copy()

    assert np.array_equal(got["off"], expect)
    assert np.array_equal(got["on"], got["off"]), "the reorder changed the values"


# ---------------------------------------------------------------------------------------------
# 4. the whole models -- the tag-survival table, and the frozen baseline
# ---------------------------------------------------------------------------------------------

BUNDLE = artifacts_dir() / "recaptures" / "small_llama_int8_consistent"


def _prepared(bundle, work, features):
    """The module the lowering receives: the prepare pass (which applies the tags) then the
    textual preprocessing, exactly as every whole-model backend does it."""
    from merlin.llvmlower.passes_xdsl import preprocess_text_textual
    from merlin.runtime.backends.zephyr_model import prepare_for_lowering

    work.mkdir(parents=True, exist_ok=True)
    prepared, _ = prepare_for_lowering(bundle / "model.mlir", work, int8_compute=True,
                                       features=impr.normalize(features), blocking=False)
    text, _stats = preprocess_text_textual(prepared.read_text(encoding="utf-8"))
    return text


@pytest.mark.skipif(not os.environ.get("MERLIN_RUN_SLOW"),
                    reason="whole-model lowering; MERLIN_RUN_SLOW=1")
@_needs_m2m
@pytest.mark.skipif(not (BUNDLE / "golden_w8a8.npy").is_file(), reason="int8 capture bundle absent")
def test_whole_model_tag_survival(tmp_path, monkeypatch):
    """The measured table, as an invariant, on a real model.

    small_llama int8: 107 ops tagged by the prepare pass, 84 still tagged when the pipeline reaches
    the point the arms now run, and 33 left once specialization has been through them. The
    assertions are RATIOS rather than the exact counts, so an upstream canonicalizer that merges two
    more generics does not fail the build -- but a placement that puts the arms back behind
    specialization does, and so does one that leaves them there for the ESCAPE HATCH to rebuild.
    """
    monkeypatch.delenv("MERLIN_VEC_AFTER_SPECIALIZE", raising=False)
    monkeypatch.delenv("MERLIN_GENERALIZE_BEFORE_FUSE", raising=False)
    upstream = _prepared(BUNDLE, tmp_path / "tagged", {VEC})
    tagged = upstream.count("merlin.vec_r")
    assert tagged > 0, "the prepare pass tagged nothing; this fixture cannot measure survival"

    reach_new = P.apply_passes(upstream, "canonicalize,cse", timeout=3600).count("merlin.vec_r")
    reach_old = P.apply_passes(upstream, f"canonicalize,cse,{SPECIALIZE}",
                               timeout=3600).count("merlin.vec_r")
    assert reach_old < reach_new / 2, (tagged, reach_new, reach_old)
    assert reach_new > 0.7 * tagged, (tagged, reach_new)

    # ...and the coverage difference reaches the EMITTED CODE, not only the tag census.
    new = _lower(upstream, {VEC}, tmp_path)
    monkeypatch.setenv("MERLIN_VEC_AFTER_SPECIALIZE", "1")
    old = _lower(upstream, {VEC}, tmp_path)
    assert new != old, "the two placements produced the same module"
    assert new.count("load <") > 1.5 * old.count("load <"), \
        (new.count("load <"), old.count("load <"))


@pytest.mark.skipif(not os.environ.get("MERLIN_RUN_SLOW"),
                    reason="whole-model lowering; MERLIN_RUN_SLOW=1")
@_needs_m2m
@pytest.mark.skipif(not (BUNDLE / "golden_w8a8.npy").is_file(), reason="int8 capture bundle absent")
def test_whole_model_baseline_is_frozen_and_the_reorder_reaches_the_fusion(tmp_path, monkeypatch):
    """Two claims on one real model, because they are the two halves of "default-off".

    FROZEN: a lowering that names no feature must be byte-identical with the reorder gate on and
    off -- the RVV pass list carries no fusion stage for it to move.
    REACHED: a lowering that DOES name the fusion feature must not be, or the gate is decoration.
    """
    monkeypatch.delenv("MERLIN_VEC_AFTER_SPECIALIZE", raising=False)
    monkeypatch.delenv("MERLIN_FUSE_POST", raising=False)
    base_text = _prepared(BUNDLE, tmp_path / "plain", frozenset())

    monkeypatch.delenv("MERLIN_GENERALIZE_BEFORE_FUSE", raising=False)
    off = _lower(base_text, frozenset(), tmp_path)
    fuse_off = _lower(base_text, {FUSE}, tmp_path)
    monkeypatch.setenv("MERLIN_GENERALIZE_BEFORE_FUSE", "1")
    on = _lower(base_text, frozenset(), tmp_path)
    fuse_on = _lower(base_text, {FUSE}, tmp_path)

    assert off == on, "the reorder touched a pipeline that carries no fusion stage"
    assert fuse_on != fuse_off, "the reorder did not reach the feature-driven fusion stage"
    # what it buys, measured: the broadcasts specialize created stop being materialized
    assert fuse_on.count("call ptr @malloc") < fuse_off.count("call ptr @malloc"), \
        (fuse_on.count("call ptr @malloc"), fuse_off.count("call ptr @malloc"))
