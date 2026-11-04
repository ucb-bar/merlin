"""A rank-0 result compares by value instead of crashing the tier.

`SY_movement_fill_scalar_i8_aligned` (a `fill` to `tensor<i8>`) reads back as a bare scalar. Before
cc44e65ce the L1 comparison iterated it (`TypeError: 'int' object is not iterable`) and the capsule
was reported RUNNER_CRASH for every compiler, so no candidate could prove fill.Scalar or
scalar_tensor. These checks pin that fix: a scalar still has to equal the reference exactly.
"""

from __future__ import annotations

from merlin.targetgen.capsule_common import _flat


def test_rank0_readback_flattens_to_one_scalar():
    assert _flat(5) == [5]
    assert _flat(-128) == [-128]
    assert _flat([[1, 2], [3]]) == [1, 2, 3]
    assert _flat([]) == []


def test_rank0_exact_match_compares_the_value_and_still_fails_closed():
    from merlin.targetgen.capsule_runner import _exact_match

    assert _exact_match({"Y0": 5}, {"Y0": 5}) is True
    assert _exact_match({"Y0": 5}, {"Y0": 6}) is False
    assert _exact_match({"Y0": 5}, {"Y1": 5}) is False
    assert _exact_match({}, {}) is False  # no outputs is never a match
