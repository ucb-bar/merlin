"""The TARGET-AGNOSTIC micro-kernel codegen space: one knob space, per-target resolvers, and
honest UnsupportedAxis for anything a target cannot yet EMIT (never a silent no-op)."""
from __future__ import annotations

import pytest

from merlin.kernels.microkernel import (PRUNED_AXES, VL_DYNAMIC, VL_FIXED, MicrokernelSpec,
                                        UnsupportedAxis, is_axis_proposable, proposable_axes,
                                        register_resolver, registered_targets, resolve)


def test_spec_validates_and_round_trips():
    s = MicrokernelSpec(MR=7, NR=32, KC=16, unroll_m=True, vl_strategy=VL_DYNAMIC, pack=True)
    assert MicrokernelSpec.from_knobs(s.to_knobs()) == s
    assert s.with_(MR=4).MR == 4 and s.with_(MR=4).NR == 32      # neighbouring point (a beam mutation)
    with pytest.raises(ValueError):
        MicrokernelSpec(vl_strategy="sometimes")                  # unknown strategy is loud
    with pytest.raises(ValueError):
        MicrokernelSpec(MR=0)                                     # non-positive block is loud
    with pytest.raises(ValueError):
        MicrokernelSpec.from_knobs({"MR": 4, "bogus": 1})         # unknown knob is loud


def test_space_is_target_agnostic_and_unregistered_target_raises():
    # RVV registers its resolver on import of the rvv generator...
    from merlin.rvvgen import from_strategy  # noqa: F401
    assert "rvv" in registered_targets()
    assert resolve("rvv", MicrokernelSpec(MR=7, NR=16, KC=16)) == ["accum_resident_v3_7_16_16",
                                                                   "erase_self_copy"]
    # ...and ANY other target plugs in the same way (this is the point: not an RVV-only capability).
    register_resolver("fake_accel", lambda spec: [f"tile_{spec.MR}x{spec.NR}x{spec.KC}"])
    assert resolve("fake_accel", MicrokernelSpec(MR=8, NR=8, KC=4)) == ["tile_8x8x4"]
    # a target with no resolver NEVER silently no-ops
    with pytest.raises(UnsupportedAxis):
        resolve("target_with_no_resolver", MicrokernelSpec())


def test_unexpressible_axes_raise_instead_of_being_ignored():
    """An axis the target cannot EMIT must stay an OPEN divergence — crediting it would be a fake win."""
    from merlin.rvvgen import from_strategy  # noqa: F401
    # composing unroll_m with pack is not emitted yet (each replaces the schedule)
    with pytest.raises(UnsupportedAxis, match="pack"):
        resolve("rvv", MicrokernelSpec(MR=4, unroll_m=True, pack=True))
    # the supported axes resolve
    assert resolve("rvv", MicrokernelSpec(MR=4, NR=16, KC=16, vl_strategy=VL_FIXED))


def test_vl_dynamic_is_emitted_through_the_scalable_route():
    """vl_strategy='dynamic' USED to raise (MLIR scalable->RVV was thought incomplete). It now
    resolves: the N register block is a scalable vector<[k]xT>, so the emitted loop sizes to the
    runtime VL — no _zvl march pin, correct on any RVV part. It rides the ORDINARY MLIR scalable
    lowering (the custom_isa inline-asm hatch was NOT needed)."""
    from merlin.rvvgen import from_strategy  # noqa: F401
    feats = resolve("rvv", MicrokernelSpec(MR=4, NR=16, KC=16, vl_strategy=VL_DYNAMIC))
    assert feats == ["accum_resident_v3vl_4_16_16", "erase_self_copy"]
    # a DISTINCT point from the fixed-width block the beam can trade against
    assert feats != resolve("rvv", MicrokernelSpec(MR=4, NR=16, KC=16, vl_strategy=VL_FIXED))
    # NR counts lanes at the RVV minimum VLEN (a scalable vector<[k]> holds 2k of them), so it must
    # be even; an odd NR is rejected loudly rather than silently rounded.
    with pytest.raises(ValueError, match="even NR"):
        resolve("rvv", MicrokernelSpec(MR=4, NR=15, KC=16, vl_strategy=VL_DYNAMIC))
    # dynamic VL does not compose with the recipe-replacing axes yet; it must RAISE, not silently
    # emit fixed-width code for a spec that asked for a runtime VL.
    for kw in ({"unroll_m": True}, {"pack": True}, {"k_block": True}):
        with pytest.raises(UnsupportedAxis, match="dynamic"):
            resolve("rvv", MicrokernelSpec(MR=4, NR=16, KC=16, vl_strategy=VL_DYNAMIC, **kw))


def test_vl_dynamic_schedule_is_scalable_peeled_and_unmasked():
    """The emitted transform schedule must carry what a WORKING scalable micro-kernel needs: a
    scalable N tile (vector<[k]>), the N loop PEELED, and the vectorize marked
    assume_dynamic_dims_match_vec_sizes (which the peel is what makes true). Without the peel the
    scalable transfers are masked, which blocks the accumulator hoist and leaves an unlowerable
    scalable vector.transpose. (M is tiled by MR with the MR|M precondition the fixed 2-D recipe
    also carries; it is NOT peeled — transform.loop.peel fails on a statically-divisible loop.)"""
    from merlin.llvmlower.impr_features import (_accumulator_resident_v3_scalable_pre_schedule as S,
                                                 scalable_lanes)
    sch = S(4, 16, 16)
    assert f"[4, [{scalable_lanes(16)}], 0]" in sch          # scalable N tile alongside static MR
    assert sch.count("transform.loop.peel") == 1             # N peeled (M carries the MR|M precond)
    assert "assume_dynamic_dims_match_vec_sizes" in sch
    # KC does not tile the reduction in this recipe -> two KC values emit byte-identical schedules
    # (a free identical-config noise control on the board).
    assert S(4, 16, 16) == S(4, 16, 64)


def test_unroll_m_is_emitted_and_shape_agnostic():
    """unroll_m holds M as MR INDEPENDENT accumulators, so ANY MR is expressible — including the
    non-power-of-2 shapes the 2-D vector<MRxNR> formulation collapses on (measured 255-279x off)."""
    from merlin.rvvgen import from_strategy  # noqa: F401
    for MR in (4, 6, 7, 8):
        feats = resolve("rvv", MicrokernelSpec(MR=MR, NR=16, KC=16, unroll_m=True))
        assert feats == [f"accum_resident_v3u_{MR}_16_16", "erase_self_copy"]
    # the two formulations are DISTINCT points the beam can trade between
    assert (resolve("rvv", MicrokernelSpec(MR=4, NR=16, KC=16, unroll_m=True))
            != resolve("rvv", MicrokernelSpec(MR=4, NR=16, KC=16, unroll_m=False)))


def test_every_realization_carries_the_recipe_lowering_hygiene():
    """`erase_self_copy` rides on EVERY micro-kernel realization, so any dtype and any point in the
    space inherits it at once.

    It is a property of the recipe SHAPE (tile the output, bufferize per tile, get a
    `memref.copy %x, %x` in the tile epilogue that survives as an opaque @memrefCopy runtime call),
    not of the dtype -- which is why it must live in the resolver rather than being re-listed per
    package. MEASURED on the live K1, kernel region, correctness-gated: int8 128^3 4,230,288 ->
    3,002,346 retired instructions, 85,760 -> 69,428 ticks; f32 128^3 1.88x. The int8 arm was
    1.6-1.8x off our own best path across 64/128/256^3 purely for want of this + v3.
    """
    from merlin.rvvgen import from_strategy  # noqa: F401
    for spec in (MicrokernelSpec(MR=4, NR=16, KC=16),
                 MicrokernelSpec(MR=4, NR=16, KC=64, k_block=True),
                 MicrokernelSpec(MR=7, NR=16, KC=16, unroll_m=True),
                 MicrokernelSpec(MR=4, NR=16, KC=16, pack=True)):
        feats = resolve("rvv", spec)
        assert feats[-1] == "erase_self_copy", feats
        assert len(feats) == len(set(feats))          # never duplicated
        assert feats[0] != "erase_self_copy"          # the recipe still names the point


def test_KC_and_unroll_m_are_pruned_from_proposal_but_stay_resolvable():
    """The two INERT/structurally-wrong levers (KC inert, MR-under-unroll_m ~2.4x slower) must NOT be
    beam-explorable (a proposer would burn certify budget for no possible win), but they must stay
    RESOLVABLE so a package or test can still pin them (no code-path deleted)."""
    from merlin.rvvgen import from_strategy  # noqa: F401  (registers the rvv resolver)
    # pruned from the proposal space ...
    assert set(PRUNED_AXES) == {"KC", "unroll_m"}
    assert not is_axis_proposable("KC") and not is_axis_proposable("unroll_m")
    assert "KC" not in proposable_axes() and "unroll_m" not in proposable_axes()
    # ... while the genuinely-useful axes remain proposable.
    for a in ("MR", "NR", "vl_strategy", "pack", "k_block"):
        assert is_axis_proposable(a) and a in proposable_axes()
    # ... and every pruned axis is STILL resolvable (the code path is not deleted).
    assert resolve("rvv", MicrokernelSpec(MR=4, NR=16, KC=64, k_block=True))   # KC via k_block
    assert resolve("rvv", MicrokernelSpec(MR=7, NR=16, KC=16, unroll_m=True))  # unroll_m


def test_hand_v0_never_reaches_the_hygiene_and_stays_byte_identical():
    """The frozen control carries no `microkernel` knob block, so it never enters the resolver and
    its lowering is unchanged -- the whole reason the erase is a resolver concern and not a global
    default-on pass."""
    from pathlib import Path
    from merlin.common.paths import repo_root
    from merlin.rvvgen.registry import load_rvv_package
    pkg_dir = Path(repo_root()) / "out/artifacts/targets/rvv/hand_v0"
    if not pkg_dir.is_dir():
        pytest.skip("hand_v0 package not present")
    pkg = load_rvv_package(pkg_dir)
    assert "microkernel" not in pkg.knobs
    assert pkg.compiler_features == []
