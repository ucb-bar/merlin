"""The TARGET-AGNOSTIC micro-kernel codegen space: one knob space, per-target resolvers, and
honest UnsupportedAxis for anything a target cannot yet EMIT (never a silent no-op)."""
from __future__ import annotations

import pytest

from merlin.kernels.microkernel import (VL_DYNAMIC, VL_FIXED, MicrokernelSpec, UnsupportedAxis,
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
    assert resolve("rvv", MicrokernelSpec(MR=7, NR=16, KC=16)) == ["accum_resident_v3_7_16_16"]
    # ...and ANY other target plugs in the same way (this is the point: not an RVV-only capability).
    register_resolver("fake_accel", lambda spec: [f"tile_{spec.MR}x{spec.NR}x{spec.KC}"])
    assert resolve("fake_accel", MicrokernelSpec(MR=8, NR=8, KC=4)) == ["tile_8x8x4"]
    # a target with no resolver NEVER silently no-ops
    with pytest.raises(UnsupportedAxis):
        resolve("target_with_no_resolver", MicrokernelSpec())


def test_unexpressible_axes_raise_instead_of_being_ignored():
    """An axis the target cannot EMIT must stay an OPEN divergence — crediting it would be a fake win."""
    from merlin.rvvgen import from_strategy  # noqa: F401
    with pytest.raises(UnsupportedAxis, match="unroll_m"):
        resolve("rvv", MicrokernelSpec(MR=7, unroll_m=True))
    with pytest.raises(UnsupportedAxis, match="vsetvli|dynamic"):
        resolve("rvv", MicrokernelSpec(vl_strategy=VL_DYNAMIC))
    # the supported axes still resolve
    assert resolve("rvv", MicrokernelSpec(MR=4, NR=16, KC=16, vl_strategy=VL_FIXED))
