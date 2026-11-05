"""Priority arithmetic controls; these fixtures issue no timing qualification."""

from copy import deepcopy

import pytest
from merlin_experiments.phase2.contracts import StageGateError
from merlin_experiments.phase2.search_priorities import measured_priorities


def row(cycles, *, macs=None, rate=None, bound=None):
    return {
        "candidate_gsim_cycles": cycles,
        "declared_macs": macs,
        "achievable_macs_per_cycle": rate,
        "roofline": {"status": "derived" if bound else "unknown", "roofline_cycles": bound},
    }


def test_large_absolute_gap_outweighs_low_utilization_of_small_member():
    small, large, host = ("compute", "small"), ("compute", "large"), ("host", "copy")
    result = measured_priorities(
        {small: [row(100, macs=1, rate=1)], large: [row(10_000, macs=5_000, rate=1)], host: [row(20_000)]},
        {small: 0.25, large: 0.25, host: 0.5},
    )
    assert [member["capsule"] for member in result["ranked"]] == ["large", "small"]
    assert result["ranked"][0]["potential_weighted_savings"] == {"lo": 1_250, "hi": 1_250}
    assert result["unknown"][0]["capsule"] == "copy"
    assert result["unpriced_weighted_cycles"] == {"lo": 10_000, "hi": 10_000}
    assert result["weighted_cycles"] == {"lo": 12_525, "hi": 12_525}


def test_repeats_preserve_all_observed_bounds_and_do_not_invent_cost():
    identity = ("compute", "tail")
    result = measured_priorities({identity: [row(200, bound=100), row(300, bound=100)]}, {identity: 1})
    member = result["ranked"][0]
    assert member["potential_weighted_savings"] == {"lo": 100, "hi": 200}
    assert member["evaluation_cost_seconds"] is None
    assert member["savings_per_evaluation_second"] is None
    assert "upper bound" in member["basis"]


@pytest.mark.parametrize("second", [row(200, bound=101), row(99, bound=100), row(200)])
def test_disagreeing_absent_or_refuted_bound_retains_all_time_as_unknown(second):
    identity = ("compute", "tail")
    result = measured_priorities({identity: [row(200, bound=100), second]}, {identity: 1})
    assert result["ranked"] == []
    assert result["unpriced_weighted_cycles"] == result["weighted_cycles"]


def test_faster_than_empirical_target_is_zero_estimated_gap_not_refuted_timing():
    identity = ("compute", "tail")
    result = measured_priorities({identity: [row(50, macs=100, rate=1)]}, {identity: 1})
    assert result["ranked"][0]["potential_weighted_savings"] == {"lo": 0, "hi": 0}


@pytest.mark.parametrize("weights", [{}, {("compute", "tail"): 0.5}, {("compute", "tail"): float("nan")}])
def test_partial_or_changed_objective_weights_refuse(weights):
    with pytest.raises(StageGateError):
        measured_priorities({("compute", "tail"): [row(100)]}, weights)


def test_projection_does_not_modify_issued_observations():
    members = {("compute", "tail"): [row(100, bound=10)]}
    original = deepcopy(members)
    measured_priorities(members, {("compute", "tail"): 1})
    assert members == original


@pytest.mark.parametrize("macs,rate", [(100, float("nan")), (100, float("inf")), (1e300, 1e-300), (1 << 10_000, 1)])
def test_nonfinite_or_overflowing_throughput_estimates_cannot_price_member(macs, rate):
    identity = ("compute", "tail")
    result = measured_priorities({identity: [row(100, macs=macs, rate=rate)]}, {identity: 1})
    assert result["ranked"] == []
    assert result["unpriced_weighted_cycles"] == {"lo": 100, "hi": 100}
