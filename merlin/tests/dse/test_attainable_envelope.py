"""A slow independent reference cannot masquerade as a necessary cost."""

import pytest

from merlin.perf.attainable_envelope import project
from merlin.perf.capsule_roofline import compose_service_floor


def floor(*, unresolved=None, scope="workload"):
    return compose_service_floor(
        {"arithmetic": dict(unit="products", capacity_per_cycle_upper=8, provenance="synthetic source capacity")},
        [
            dict(
                id="dot",
                demand={"arithmetic": 800},
                compulsory=scope == "workload",
                provenance="synthetic mandatory work",
            )
        ],
        unresolved=unresolved,
        scope=scope,
    )


def args():
    return dict(
        workload_sha256="1" * 64,
        scope_sha256="2" * 64,
        observed_cycles=500,
        observation_kind="estimated",
        compute_floor_cycles=100,
    )


def reference():
    return dict(
        workload_sha256="1" * 64,
        scope_sha256="2" * 64,
        complete_invocation=True,
        independent=True,
        cycles_lower=180,
        cycles_upper=200,
        domain="synthetic exact geometry/inputs/cold state",
        evidence_sha256="3" * 64,
    )


def test_slow_reference_only_supplies_feasible_upper_bound():
    result = project(floor(), **args(), independent_reference=reference())
    assert result["theoretical_floor_cycles"] == 100
    assert result["conditional_optimal_cycles_interval"] == [100, 200]
    assert result["conditional_attainment_interval"] == [0.2, 0.4]
    assert result["complete_attainable_roofline_cycles"] is None
    assert result["complete_attainment"] is None
    assert result["convergence_eligible"] is False
    assert result["compute_peak_utilization_fraction"] == 0.2


def test_partial_inventory_and_unknown_reference_stay_explicit():
    result = project(floor(unresolved={"host": "unproved compulsory software service"}), **args())
    assert result["conditional_optimal_cycles_interval"] == [100, None]
    assert result["conditional_attainment_interval"] is None
    assert result["reason"] == "incomplete_service_inventory"
    assert result["unresolved"] == {"host": "unproved compulsory software service"}
    assert result["theoretical_floor_ratio"] == 0.2


def test_below_bound_retains_raw_estimate_without_attainment():
    result = project(floor(), **{**args(), "observed_cycles": 90})
    assert result["observed_cycles"] == 90
    assert result["comparison_consistent"] is False
    assert result["reason"] == "estimate_below_structural_bound"
    assert result["theoretical_floor_ratio"] is None
    assert result["compute_peak_utilization_fraction"] is None


@pytest.mark.parametrize(
    "change",
    [
        {"workload_sha256": "4" * 64},
        {"scope_sha256": "4" * 64},
        {"complete_invocation": False},
        {"independent": False},
        {"cycles_upper": 99, "cycles_lower": 90},
        {"cycles_upper": float("inf")},
    ],
)
def test_different_probe_partial_run_and_inconsistent_reference_refuse(change):
    with pytest.raises(ValueError):
        project(floor(), **args(), independent_reference={**reference(), **change})


def test_emitted_schedule_waste_cannot_be_fixed_workload_floor():
    with pytest.raises(ValueError, match="compulsory service floor"):
        project(floor(scope="emitted_schedule"), **args())


def test_measured_and_estimated_labels_stay_distinct_and_unknown_is_not_zero():
    result = project(floor(), **{**args(), "observation_kind": "measured", "observed_cycles": None})
    assert result["observation_kind"] == "measured"
    assert result["theoretical_floor_ratio"] is None
    with pytest.raises(ValueError):
        project(floor(), **{**args(), "observed_cycles": 0})
