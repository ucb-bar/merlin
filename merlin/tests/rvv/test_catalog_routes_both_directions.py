"""The catalog must be able to route the expert DOWNWARD, not only upward.

Every magnitude route was written assuming the expert is an upper bound we climb toward, so an
expert sitting BELOW us fell through every guard and routed to nothing. The lesson was mined,
compared, recorded on the node -- and discarded for having the wrong sign.

It is not a corner case. XNNPACK's int8 ukernel is ``xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4v__rvv``
-- MR=1 -- while the catalog's register-block note cites "XNNPACK 7x4v" and concludes MR should be
raised; 7x4v is the F32 kernel. On int8 the reuse comes from the VLEN-scaled N tile, not from M. The
lifted CCA said so on every generation of two searches::

    compute.register_block  expert=(1, ('vsetvlmax', 4.0))  ours=(4, ('vsetvlmax', 8.0))
    vector.lmul             expert=4.0                      ours=8.0

and our MR=4 variant measured 1.61x SLOWER than the default, which is what that looks like from
outside.
"""
from __future__ import annotations

import pytest

from merlin.kernels.action_catalog import _is_higher, _is_lower, _mr_of, route
from merlin.kernels.cca_compare import Divergence


def _d(axis, expert, ours):
    return Divergence(axis=axis, expert=expert, ours=ours, backend="rvv", evidence=["expert"])


def test_lmul_routes_in_both_directions_and_not_when_equal():
    up = route(_d("vector.lmul", 8.0, 2.0))
    down = route(_d("vector.lmul", 4.0, 8.0))
    assert up is not None and "widen" in up.change
    assert down is not None and "narrow" in down.change
    # narrowing to the expert is an EXACT promise: overshooting downward is not "keeping" it either
    assert down.promise_comparison == "exact"
    assert down.forkable_now is True
    # no divergence, no action
    assert route(_d("vector.lmul", 4.0, 4.0)) is None


def test_register_block_lowers_toward_an_expert_below_us():
    """`at_least` cannot express this: reaching MORE blocking than an expert that chose MR=1 is not
    a kept promise, it is the regression."""
    down = route(_d("compute.register_block", (1, ("vsetvlmax", 4.0)), (4, ("vsetvlmax", 8.0))))
    assert down is not None
    assert "lower" in down.change
    assert down.promise_comparison == "exact"
    assert down.forkable_now is True


def test_register_block_still_raises_toward_an_expert_above_us():
    """The f32 case must keep working -- this is a widened ladder, not a flipped one."""
    up = route(_d("compute.register_block", (7, ("vsetvlmax", 4.0)), (1, ("vsetvlmax", 4.0))))
    assert up is not None
    assert "raise" in up.change
    assert up.promise_comparison == "at_least"


def test_the_direction_predicates_agree_and_fail_closed():
    assert _is_lower(_d("vector.lmul", 4.0, 8.0)) and not _is_higher(_d("vector.lmul", 4.0, 8.0))
    assert _is_higher(_d("vector.lmul", 8.0, 4.0)) and not _is_lower(_d("vector.lmul", 8.0, 4.0))
    for same in (4.0, "m4"):
        assert not _is_lower(_d("vector.lmul", same, same))
    # unorderable values are NOT silently treated as a direction
    assert not _is_lower(_d("vector.lmul", "na", 8.0))
    assert not _is_higher(_d("vector.lmul", "na", 8.0))


def test_mr_is_read_out_of_the_register_block_tuple():
    assert _mr_of((1, ("vsetvlmax", 4.0))) == 1.0
    assert _mr_of(4) == 4.0
    for junk in ("x", None, (), {"mr": 1}):
        assert _mr_of(junk) is None


def test_the_experts_own_mr_is_in_the_search_space():
    """A ladder that cannot express the expert's number cannot converge on it. The int8 tile ladder
    began at MR=2 while the expert uses MR=1."""
    from merlin.llvmlower.impr_features import MRPAD_INT8_TILES
    assert any("_mr1_" in n for n in MRPAD_INT8_TILES), "the expert's MR=1 is not searchable"
