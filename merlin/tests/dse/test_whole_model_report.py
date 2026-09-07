"""Success is gated on held-out complete models and attainable-gap closure."""
import pytest

from merlin.perf.global_planner import OccupancySummary
from merlin.perf.whole_model_report import ModelPerformance, evaluate_whole_models


def _occupancy(total, compute, movement, overlap, encodings=0):
    return OccupancySummary(
        total_cycles=total,
        busy_cycles=(("array", compute), ("dma", movement)),
        compute_resources=("array",), movement_resources=("dma",),
        overlap_cycles=overlap,
        overlap_available_cycles=min(compute, movement),
        idle_cycles=0, critical_path_cycles=total,
        movement_bytes=1000, movement_commands=10,
        encoding_transitions=encodings,
        provenance=("warm whole-model counter",))


def test_pass_requires_model_speedup_gap_closure_and_roofline_attainment() -> None:
    report = evaluate_whole_models((
        ModelPerformance("model_a", 1000, 200, 190, 150, True,
                         _occupancy(200, 180, 60, 50, 1),
                         provenance=("test measured A/B receipt",), cycle_basis="measured"),
        ModelPerformance("model_b", 2000, 400, 380, 300, True,
                         _occupancy(400, 360, 100, 80),
                         provenance=("test measured A/B receipt",), cycle_basis="measured"),
    ))

    assert report.passed
    assert report.heldout_geomean_speedup == pytest.approx(5)
    assert report.worst_legal_attainment == 0.95
    assert report.aggregate_gap_closed > 0.8
    assert report.opportunities[0].kind == "accelerator_bubbles"
    assert report.to_dict()["status"] == "pass"


@pytest.mark.parametrize("basis", ["unspecified", "model_estimate"])
def test_favorable_estimates_cannot_pass_measured_performance_gate(basis):
    row = ModelPerformance("model", 1000, 200, 190, 150, True,
        _occupancy(200, 180, 60, 50), provenance=("cost-model source",), cycle_basis=basis)
    report = evaluate_whole_models((row,))
    assert report.heldout_geomean_speedup == pytest.approx(5)
    assert not report.passed
    assert not dict(report.gates)["measured_whole_model_cycles"]
    assert report.to_dict()["aggregate"]["cycle_basis"] == basis


def test_plan_conversion_preserves_estimate_basis():
    from types import SimpleNamespace
    from merlin.perf.whole_model_report import performance_from_plan
    result = SimpleNamespace(
        plan=SimpleNamespace(cycles=SimpleNamespace(resolved=True, hi=200), digest="plan"),
        legal_floor=SimpleNamespace(cycles=190), physical_floor=SimpleNamespace(cycles=150),
        occupancy=_occupancy(200, 180, 60, 50))
    row = performance_from_plan("model", 1000, result, heldout=True,
                                provenance=("calibrated short probes",))
    assert row.cycle_basis == "model_estimate"
    assert not evaluate_whole_models((row,)).passed


def test_measured_label_without_provenance_does_not_earn_a_verdict():
    row = ModelPerformance("model", 1000, 200, 190, 150, True,
                          _occupancy(200, 180, 60, 50), cycle_basis="measured")
    assert not evaluate_whole_models((row,)).passed


def test_micro_result_cannot_substitute_for_missing_heldout_whole_model() -> None:
    report = evaluate_whole_models((
        ModelPerformance("training_model", 1000, 100, 90, 80, False,
                         _occupancy(100, 90, 20, 15)),
    ))

    assert not report.passed
    assert "no held-out whole-model result" in report.refusals


def test_unknown_roofline_blocks_gap_closure_instead_of_reading_as_zero() -> None:
    report = evaluate_whole_models((
        ModelPerformance("model", 1000, 300, None, None, True,
                         _occupancy(300, 250, 100, 60)),
    ))

    assert not report.passed
    assert report.aggregate_gap_closed is None
    assert any("UNKNOWN" in item for item in report.refusals)


def test_fast_geomean_does_not_hide_one_model_regression() -> None:
    report = evaluate_whole_models((
        ModelPerformance("fast", 1000, 100, 95, 80, True,
                         _occupancy(100, 95, 10, 5)),
        ModelPerformance("regressed", 1000, 1021, 900, 800, True,
                         _occupancy(1021, 900, 200, 79)),
    ))

    assert not report.passed
    assert not dict(report.gates)["individual_regression"]


def test_speedup_cannot_pass_without_whole_model_occupancy_evidence() -> None:
    report = evaluate_whole_models((
        ModelPerformance("model", 1000, 100, 95, 80, True, None),
    ))

    assert not report.passed
    assert not dict(report.gates)["whole_model_occupancy_evidence"]
    assert any("occupancy/movement/latency-hiding" in item for item in report.refusals)


def test_overlapping_transfer_engines_do_not_invent_exposed_movement() -> None:
    from merlin.perf.activity_schedule import ActivityEvent, schedule_activity
    occupancy = schedule_activity((
        ActivityEvent("compute", "compute", "compute", 10),
        ActivityEvent("load", "load", "movement", 4),
        ActivityEvent("store", "store", "movement", 6),
    )).occupancy()
    report = evaluate_whole_models((ModelPerformance("model", 100, 10, 10, 10, True, occupancy),))
    exposure = next(row for row in report.opportunities if row.kind == "exposed_movement")
    assert occupancy.movement_elapsed_cycles == 6
    assert exposure.cycles == 0


def test_aggregate_resource_counters_do_not_imply_transfer_elapsed_union() -> None:
    occupancy = OccupancySummary(10, (("compute", 10), ("load", 4), ("store", 6)),
        compute_resources=("compute",), movement_resources=("load", "store"), overlap_cycles=6)
    report = evaluate_whole_models((ModelPerformance("model", 100, 10, 10, 10, True, occupancy),))
    exposure = next(row for row in report.opportunities if row.kind == "unknown_exposed_movement")
    assert exposure.cycles is None


def test_missing_compute_counter_is_not_zero_busy_or_ranked_as_idle_capacity():
    occupancy = OccupancySummary(10, (("first", 10),),
        compute_resources=("first", "second"))
    assert occupancy.compute_busy_cycles is None
    assert occupancy.compute_utilization is None
    assert "busy cycles for declared resource second" in occupancy.to_dict()["missing"]
    report = evaluate_whole_models((ModelPerformance("model", 100, 10, 10, 10, True, occupancy),))
    assert not any(item.kind == "accelerator_bubbles" for item in report.opportunities)
    unknown = next(item for item in report.opportunities if item.kind == "unknown_accelerator_bubbles")
    assert unknown.cycles is None
    assert not report.passed


def test_explicit_zero_busy_is_known_idle_capacity_not_missing_evidence():
    occupancy = OccupancySummary(10, (("first", 10), ("second", 0)),
        compute_resources=("first", "second"))
    assert occupancy.compute_busy_cycles == 10
    assert occupancy.compute_utilization == 0.5
    assert occupancy.missing == ()
    report = evaluate_whole_models((ModelPerformance("model", 100, 10, 10, 10, True, occupancy),))
    bubbles = next(item for item in report.opportunities if item.kind == "accelerator_bubbles")
    assert bubbles.cycles == 10


def test_missing_declared_movement_engine_cannot_pass_complete_occupancy_gate():
    from dataclasses import replace
    occupancy = replace(_occupancy(200, 180, 60, 50), movement_resources=("dma", "store"))
    report = evaluate_whole_models((ModelPerformance("model", 1000, 200, 190, 150, True,
        occupancy, provenance=("counter fixture",), cycle_basis="measured"),))
    assert not dict(report.gates)["whole_model_occupancy_evidence"]
    assert not report.passed


@pytest.mark.parametrize("kwargs", [
    {"busy_cycles": (("first", 2), ("first", 3))},
    {"busy_cycles": (("first", 2),), "compute_resources": ("first", "first")},
    {"busy_cycles": (("first", 2),), "movement_resources": ("first", "first")},
])
def test_duplicate_resource_evidence_cannot_change_utilization_denominator(kwargs):
    with pytest.raises(ValueError):
        OccupancySummary(total_cycles=10, **kwargs)
