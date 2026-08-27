"""The non-contraction vectorize lever: selectable by the loop, and MEASURED rather than assumed.

The dominant structural loss on every captured workload is that 86-89% of linalg ops are not
contractions, so the contraction-only schedule leaves them to convert-linalg-to-loops (scalar, one
core). The mechanism to vectorize them existed only as an env-var experiment, which is precisely why no
search ever selected it. These tests pin that it is now a registered feature with a lane-width knob
space, that its tagging predicate refuses the cases that break the build, and that the composition is
default-off.

Deliberately NOT asserted here: that the lever is a speedup. Measured on deepjscc int8 (spike) it emits
4.9x more vector instructions with bit-identical output and runs 1.28x SLOWER at every lane width. That
negative result is the reason the axis is routed — the loop has to be able to try it and reject it.
"""
from __future__ import annotations

import pytest

from merlin.llvmlower import impr_features as impr


def test_the_lever_is_a_registered_feature_not_an_env_var():
    """A lever only reachable through an env var cannot be selected by the tuning loop."""
    assert impr.VEC_NONCONTRACTION_NAME in impr.known()


def test_it_is_additive_so_it_layers_on_a_micro_kernel_recipe():
    """It must not be schedule_replace: the contraction arms have to survive underneath it."""
    from merlin.mining.registry import load_rvv_package

    pkg = load_rvv_package("out/artifacts/targets/rvv/impr_tuned_wholemodel_vf_int8")
    base = impr.apply_schedule(pkg.schedule_text, frozenset(pkg.compiler_features))
    with_lever = impr.apply_schedule(
        pkg.schedule_text, frozenset([*pkg.compiler_features, impr.VEC_NONCONTRACTION_NAME]))
    assert "merlin.vec_r" not in base, "the baseline must be untouched (default-off)"
    assert "merlin.vec_r" in with_lever
    # every contraction arm of the base schedule survives
    for line in base.splitlines():
        if "linalg.matmul" in line or "linalg.batch_matmul" in line:
            assert line in with_lever, f"the lever clobbered a contraction arm: {line.strip()}"


def test_each_arm_tiles_before_vectorizing():
    """structured.vectorize does not tile: sizes must cover the iteration space.

    Vectorizing an untiled 1x64 relu with [1, 8] fails the WHOLE pipeline ("Attempted to vectorize, but
    failed"), so every arm has to tile to the vector width first.
    """
    arms = impr._vec_rank_arms(8)
    for rank_handle in ("%g2", "%g3", "%g4"):
        assert f"tile_using_for {rank_handle} tile_sizes" in arms
    assert arms.count("transform.structured.vectorize") == 3


def test_the_leading_unit_dim_is_cast_away():
    """A [1, N] tile vectorizes to vector<1xNxT>, whose leading unit dim makes the lowering emit rank-2
    vector.extracts this LLVM build cannot translate."""
    arms = impr._vec_rank_arms(8)
    assert "cast_away_vector_leading_one_dim" in arms


@pytest.mark.parametrize("lanes", [8, 16, 32])
def test_the_lane_width_is_a_knob_space(lanes):
    """The width must be searchable: measured cycles are flat at 0.78x across 8/16/32, which is itself
    a finding the loop can only reach if it can vary the width."""
    name = impr.ensure_vec_noncontraction(lanes)
    assert name in impr.known()
    assert impr.vec_noncontraction_lanes(frozenset([name])) == lanes
    arms = impr._vec_rank_arms(lanes)
    assert f"tile_sizes [1, {lanes}]" in arms


def test_lanes_helper_is_none_without_the_feature():
    """The tagging predicate keys off this, so a false positive would tag ops for a schedule that has no
    arms to match them."""
    assert impr.vec_noncontraction_lanes(frozenset(["hand_v0"])) is None
    assert impr.vec_noncontraction_lanes(frozenset()) is None


def test_tagging_refuses_the_cases_that_break_the_build():
    """The predicate must skip a data-dependent gather, a transcendental body, and a non-multiple extent.

    Each was measured to fail: a gather has no affine access ("Attempted to vectorize, but failed"), a
    math.exp body gets scalarized back into per-lane extracts AFTER vector->LLVM has run, and a partial
    tail means a masked parallel dim, which does not lower on the integer path.
    """
    import inspect

    from merlin.runtime.backends import zephyr_model as zm

    src = inspect.getsource(zm._prepare_model_mlir)
    assert "tensor.extract" in src and "memref.load" in src, "gather bodies must be skipped"
    assert 'startswith("math.")' in src, "transcendental bodies must be skipped"
    assert "% vec_lanes" in src, "the innermost extent must be a whole multiple of the lane count"
