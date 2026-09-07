"""The data-driven planner edge accounts for representations and emitted activity exactly."""
from __future__ import annotations

import pytest

from merlin.perf.activity_schedule import ActivityEvent, schedule_activity
from merlin.perf.decompose import ResourceKind
from merlin.perf.explicit_plan_adapter import ExplicitPlanningAdapter, TransitionRule
from merlin.perf.global_planner import Bound, GlobalPlanPolicy, optimize_program
from merlin.perf.headroom import Composition
from merlin.xdsl_dialects.lowering.dispatch_program import Buffer, DispatchProgram, Node
from merlin.xdsl_dialects.lowering.global_plan import (
    BufferRepresentation,
    CycleInterval,
    RegionAlternative,
    ResourceOccupancy,
    ValueRepresentation,
)


HOST = ValueRepresentation("host", "row_major", "i8", encoding="plain")
PACKED = ValueRepresentation("local", "blocked", "i8", encoding="packed")


def _program() -> DispatchProgram:
    return DispatchProgram(
        entry="forward",
        args=[0],
        buffers={
            "input": Buffer("input", [8, 8], "i8", "arg", 0),
            "output": Buffer("output", [8, 8], "i8", "intermediate"),
        },
        nodes=[Node("dispatch", "kernel", ["input"], ["output"], captures=[])],
        results=["output"],
    )


def _alternative() -> RegionAlternative:
    return RegionAlternative(
        id="packed_kernel",
        nodes=(0,),
        implementation="packed_kernel",
        placement="array",
        cycles=CycleInterval.point(80, "warm kernel probe"),
        occupancy=(ResourceOccupancy(
            "array", CycleInterval.point(80, "warm kernel probe"), "warm kernel probe"),),
        inputs=(BufferRepresentation("input", HOST),),
        outputs=(BufferRepresentation("output", PACKED),),
    )


def _event_builder(program, selected, transitions, endpoint):
    compute = selected[0]
    conversion = transitions[0]
    return schedule_activity((
        ActivityEvent(
            "compute", compute.placement, "compute", getattr(compute.cycles, endpoint),
            provenance="warm kernel probe"),
        ActivityEvent(
            "unpack", "dma", "encoding", getattr(conversion.cycles, endpoint),
            depends_on=("compute",), movement_bytes=64, movement_commands=1,
            encoding_transition=True, provenance="warm conversion probe"),
    ))


def _adapter(*, event_builder=_event_builder,
             composition: Composition = Composition.MAX) -> ExplicitPlanningAdapter:
    return ExplicitPlanningAdapter(
        alternatives=(_alternative(),),
        boundaries={("input", "input"): HOST, ("output", "output"): HOST},
        transition_rules=(TransitionRule(
            "unpack", PACKED, HOST, CycleInterval.point(40, "warm conversion probe"),
            "dma", 64, 1, provenance="exact output shape"),),
        resource_kinds={"array": ResourceKind.COMPUTE, "dma": ResourceKind.MOVEMENT},
        composition=composition,
        composition_eta=1.0 if composition is Composition.MAX else 0.0,
        physical=Bound(80, provenance=("resource floor",)),
        event_builder=event_builder,
        composition_provenance="paired overlap probe",
    )


def test_dependency_timeline_exposes_unhidden_output_conversion() -> None:
    result = optimize_program(
        _program(), _adapter(), policy=GlobalPlanPolicy(timeout_s=1))

    assert result.plan is not None
    assert result.plan.cycles.hi == 120
    assert result.plan.transitions[0].kind == "encoding"
    # The max-composed resource floor is 80, but an output conversion cannot begin until the
    # producer completes.  The event timeline therefore exposes all 40 cycles instead of claiming
    # that an aggregate movement bucket was hidden.
    assert result.plan.cycles.hi > result.physical_floor.cycles


def test_event_timeline_must_account_for_every_priced_resource() -> None:
    def missing_conversion(program, selected, transitions, endpoint):
        compute = selected[0]
        return schedule_activity((ActivityEvent(
            "compute", "array", "compute", getattr(compute.cycles, endpoint)),))

    result = optimize_program(
        _program(), _adapter(event_builder=missing_conversion),
        policy=GlobalPlanPolicy(timeout_s=1))

    assert result.plan is None
    assert any("does not account for the priced occupancy" in item for item in result.refusals)


def test_serial_composition_rejects_an_illegally_overlapped_timeline() -> None:
    def illegally_overlapped(program, selected, transitions, endpoint):
        compute = selected[0]
        conversion = transitions[0]
        return schedule_activity((
            ActivityEvent("compute", "array", "compute", getattr(compute.cycles, endpoint)),
            ActivityEvent("unpack", "dma", "encoding", getattr(conversion.cycles, endpoint),
                          movement_bytes=64, movement_commands=1,
                          encoding_transition=True),
        ))

    result = optimize_program(
        _program(), _adapter(event_builder=illegally_overlapped, composition=Composition.SUM),
        policy=GlobalPlanPolicy(timeout_s=1))

    assert result.plan is None
    assert any("below the explicit sum resource bound" in item for item in result.refusals)


def test_transition_matching_includes_encoding_direction() -> None:
    adapter = _adapter()
    adapter.transition_rules = (TransitionRule(
        "wrong_direction", HOST, PACKED, CycleInterval.point(40), "dma", 64, 1),)

    result = optimize_program(
        _program(), adapter, policy=GlobalPlanPolicy(timeout_s=1))

    assert result.plan is None
    assert any("no transition establishes" in item for item in result.refusals)


def test_composition_requires_provenance() -> None:
    with pytest.raises(ValueError, match="composition provenance"):
        ExplicitPlanningAdapter(
            alternatives=(_alternative(),), boundaries={}, transition_rules=(),
            resource_kinds={"array": ResourceKind.COMPUTE},
            composition=Composition.MAX, composition_eta=1.0,
            physical=Bound(0), composition_provenance="")
