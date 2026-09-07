"""Reduced witnesses project exact full-layer pipeline fill, throughput, and occupancy."""
import pytest

from merlin.perf.pipeline_projection import (
    PipelineBuffer,
    PipelineProjectionPolicy,
    PipelineStage,
    project_pipeline,
)
from merlin.xdsl_dialects.lowering.global_plan import CycleInterval


def _stage(name, resource, kind, cycles, **kwargs):
    return PipelineStage(
        name, resource, kind, CycleInterval.point(cycles, f"warm reduced {name} witness"),
        **kwargs)


def test_full_repetition_uses_fill_plus_initiation_interval_not_full_simulation() -> None:
    projection = project_pipeline((
        _stage("load", "reader", "movement", 40,
               moved_bytes_per_item=4096, commands_per_item=1),
        _stage("execute", "array", "compute", 80),
        _stage("store", "writer", "movement", 20,
               moved_bytes_per_item=1024, commands_per_item=1),
    ), repetitions=513)

    assert projection.fill_cycles == 140
    assert projection.initiation_interval == 80
    assert projection.cycles.hi == 140 + 512 * 80
    assert projection.occupancy.busy == {
        "array": 513 * 80, "reader": 513 * 40, "writer": 513 * 20}
    assert projection.occupancy.compute_utilization > 0.99
    assert projection.occupancy.movement_bytes == 513 * (4096 + 1024)
    assert projection.occupancy.movement_commands == 513 * 2
    assert projection.occupancy.latency_hiding_efficiency is not None


def test_compulsory_fill_is_visible_on_one_item() -> None:
    projection = project_pipeline((
        _stage("load", "reader", "movement", 40),
        _stage("execute", "array", "compute", 80),
    ), repetitions=1)

    assert projection.cycles.hi == 120
    assert projection.occupancy.overlap_cycles == 0
    assert projection.occupancy.compute_utilization == 80 / 120


def test_shared_resource_requires_an_explicit_arbitration_schedule() -> None:
    with pytest.raises(ValueError, match="shared-resource arbitration"):
        project_pipeline((
            _stage("load", "dma", "movement", 40),
            _stage("execute", "array", "compute", 80),
            _stage("store", "dma", "movement", 20),
        ), repetitions=8)


def test_large_projection_keeps_exact_latency_but_marks_overlap_unknown() -> None:
    projection = project_pipeline((
        _stage("load", "reader", "movement", 2),
        _stage("execute", "array", "compute", 3),
    ), repetitions=100, policy=PipelineProjectionPolicy(max_overlap_intervals=10))

    assert projection.cycles.hi == 5 + 99 * 3
    assert projection.occupancy.overlap_cycles is None
    assert projection.occupancy.compute_utilization is not None
    assert "interval cap" in projection.occupancy.missing[0]


def test_buffer_ownership_distinguishes_serialization_from_latency_hiding() -> None:
    stages = (_stage("load", "reader", "movement", 40),
              _stage("execute", "array", "compute", 80))
    single = project_pipeline(stages, 10**12, buffers=(
        PipelineBuffer("load", "execute", 1, "one source allocation", 4096),))
    double = project_pipeline(stages, 10**12, buffers=(
        PipelineBuffer("load", "execute", 2, "two disjoint source slots", 4096),))
    assert single.cycles.hi == 120 * 10**12
    assert double.cycles.hi == 120 + (10**12 - 1) * 80
    assert single.occupancy.compute_utilization == 2 / 3
    assert double.occupancy.compute_utilization > .99
    assert single.occupancy.overlap_cycles is None
    assert single.initiation_interval is None
    assert single.to_dict()["buffers"][0]["allocation_bytes"] == 4096
    assert double.to_dict()["cycle_basis"] == "model_estimate"


@pytest.mark.parametrize("durations,slots", [
    ((2, 7, 3), (1, 1)), ((7, 2, 9), (2, 3)),
    ((3, 0, 2, 5), (1, 2, 1)), ((2, 1, 3), (30, 40)),
])
def test_powered_completion_matches_small_explicit_slot_schedule(durations, slots) -> None:
    # Tiny independent reference only: no full-model event expansion in the implementation.
    stages = tuple(_stage(str(i), f"resource{i}", "compute", p)
                   for i, p in enumerate(durations))
    buffers = tuple(PipelineBuffer(str(i), str(i + 1), count, "fixture allocation")
                    for i, count in enumerate(slots))
    completions = []
    for item in range(9):
        row = []
        for j, duration in enumerate(durations):
            ready = [row[j - 1] if j else 0,
                     completions[-1][j] if completions else 0]
            if j < len(slots) and item >= slots[j]:
                ready.append(completions[item - slots[j]][j + 1])
            row.append(max(ready) + duration)
        completions.append(row)
        result = project_pipeline(stages, item + 1, buffers=buffers)
        assert result.cycles.hi == row[-1]


def test_finite_buffer_contract_refuses_missing_edges_and_large_state_space() -> None:
    stages = (_stage("a", "r1", "movement", 2), _stage("b", "r2", "compute", 3))
    with pytest.raises(ValueError, match="every adjacent"):
        project_pipeline(stages, 4, buffers=())
    with pytest.raises(ValueError, match="exceeds cap"):
        project_pipeline(stages, 1000, buffers=(PipelineBuffer("a", "b", 100, "proof"),))
    with pytest.raises(ValueError, match="endpoints and evidence"):
        PipelineBuffer("a", "b", 2, "")
    with pytest.raises(ValueError, match="positive integer"):
        PipelineBuffer("a", "b", True, "proof")


def test_interval_floor_uses_lower_cost_and_idealization_is_explicit() -> None:
    stages = (PipelineStage("a", "r1", "movement", CycleInterval(2, 10)),
              PipelineStage("b", "r2", "compute", CycleInterval(3, 20)))
    result = project_pipeline(stages, 5)
    assert result.resource_floor.cycles == 15
    assert result.to_dict()["buffering"] == "unlimited_buffer_idealization"
    assert result.occupancy.movement_elapsed_cycles == 50
    finite = project_pipeline(stages, 5, buffers=(PipelineBuffer("a", "b", 1, "proof"),))
    assert (finite.cycles.lo, finite.cycles.hi) == (25, 150)
    assert finite.resource_floor.cycles == 15
    empty = project_pipeline(stages, 0, buffers=(PipelineBuffer("a", "b", 1, "proof"),))
    assert empty.cycles.hi == empty.fill_cycles == 0
    assert empty.occupancy.movement_elapsed_cycles == 0
