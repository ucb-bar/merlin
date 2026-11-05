"""Logical tensor ranks and dense console rows share ordered scalar comparison."""

from math import prod

import pytest

from merlin.targetgen import capsule_runner as R


def _outputs(shape):
    values = iter(range(prod(shape)))

    def nested(dimensions):
        if not dimensions:
            return next(values)
        return [nested(dimensions[1:]) for _ in range(dimensions[0])]

    logical = {"Y": nested(shape)}
    width = shape[-1]
    carrier = {"Y": [list(range(start, start + width)) for start in range(0, prod(shape), width)]}
    return logical, carrier


@pytest.mark.parametrize("shape", [(7,), (2, 7), (2, 16, 16), (2, 2, 3, 4)])
@pytest.mark.parametrize("mode", ["exact_int", "exact", "tolerance_float"])
def test_logical_rank_matches_complete_dense_rows(shape, mode):
    logical, carrier = _outputs(shape)
    assert R._match_by_policy(carrier, logical, {"compare": mode, "atol": 0.01, "rtol": 0.0})


@pytest.mark.parametrize("shape", [(2, 16, 16), (2, 2, 3, 4)])
@pytest.mark.parametrize("mutation", ["late_scalar", "dropped_scalar", "extra_scalar", "missing_output", "extra_output"])
@pytest.mark.parametrize("mode", ["exact_int", "tolerance_float"])
def test_ranked_comparison_preserves_value_count_and_output_roster(shape, mutation, mode):
    logical, carrier = _outputs(shape)
    if mutation == "late_scalar":
        carrier["Y"][-1][-1] += 1
    elif mutation == "dropped_scalar":
        carrier["Y"][-1].pop()
    elif mutation == "extra_scalar":
        carrier["Y"][-1].append(0)
    elif mutation == "missing_output":
        carrier.clear()
    else:
        carrier["extra"] = [[0]]
    assert not R._match_by_policy(carrier, logical, {"compare": mode, "atol": 0.01, "rtol": 0.0})


def test_ranked_float_comparison_retains_declared_tolerance():
    logical, carrier = _outputs((2, 2, 3))
    carrier["Y"][-1][-1] += 0.005
    policy = {"compare": "tolerance_float", "atol": 0.01, "rtol": 0.0}
    assert R._match_by_policy(carrier, logical, policy)
    assert not R._exact_match(carrier, logical)
    carrier["Y"][-1][-1] += 0.02
    assert not R._match_by_policy(carrier, logical, policy)


def test_ranked_comparison_preserves_row_major_order_and_empty_roster_refusal():
    logical, carrier = _outputs((2, 2, 3))
    carrier["Y"].reverse()
    assert not R._exact_match(carrier, logical)
    assert not R._exact_match({}, {})
