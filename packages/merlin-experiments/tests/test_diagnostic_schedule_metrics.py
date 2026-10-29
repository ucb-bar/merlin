"""Public count projection controls; no target or timing owner is constructed."""

import copy

import pytest
from merlin_experiments.phase2 import diagnostic_schedule_metrics as M


def metrics():
    return dict(
        schema=M.SCHEMA,
        scope=M.SCOPE,
        quantities={
            "workload_products": dict(baseline=16384, candidate=16384),
            "output_elements": dict(baseline=1024, candidate=1024),
            "operand_load_commands": dict(baseline=8, candidate=6),
            "requested_operand_bytes": dict(baseline=2048, candidate=1536),
        },
    )


def test_checked_requests_have_explicit_units_and_no_latency_or_physical_claim():
    source = metrics()
    before = copy.deepcopy(source)
    report = M.project(source)
    assert source == before
    assert report["quantities"]["operand_load_commands"] == dict(
        baseline=8,
        candidate=6,
        unit="commands",
        baseline_minus_candidate=2,
    )
    assert report["quantities"]["requested_operand_bytes"]["baseline_minus_candidate"] == 512
    assert all(
        report[name] is None for name in ("measured_cycles", "physical_traffic_bytes", "utilization", "attainment")
    )


@pytest.mark.parametrize("fault", ["source_work", "outputs", "scope", "values", "physical", "cycles", "missing"])
def test_no_dropped_work_or_private_or_unqualified_fields(fault):
    source = metrics()
    if fault == "source_work":
        source["quantities"]["workload_products"]["candidate"] -= 1
    elif fault == "outputs":
        source["quantities"]["output_elements"]["candidate"] = 0
    elif fault == "scope":
        source["scope"] = "complete_model"
    elif fault == "values":
        source["quantities"]["operand_values"] = [1, 2, 3]
    elif fault == "physical":
        source["quantities"]["physical_traffic_bytes"] = dict(baseline=20, candidate=10)
    elif fault == "cycles":
        source["measured_cycles"] = 100
    else:
        del source["quantities"]["workload_products"]
    with pytest.raises(M.StageGateError):
        M.project(source)


@pytest.mark.parametrize("value", [True, -1, float("nan"), float("inf"), 1 << 64, "private/path", [1, 2]])
def test_only_bounded_integer_counts_are_public(value):
    source = metrics()
    source["quantities"]["requested_operand_bytes"]["candidate"] = value
    with pytest.raises(M.StageGateError, match="bounded nonnegative"):
        M.project(source)
