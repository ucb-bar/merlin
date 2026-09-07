"""Whole-model choices include representations, movement, and producer/consumer edges."""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace

from merlin.perf.global_planner import (
    Bound,
    GlobalPlanPolicy,
    OccupancySummary,
    PlanEvaluation,
    optimize_program,
)
from merlin.xdsl_dialects.lowering.dispatch_program import Buffer, DispatchProgram, Node
from merlin.xdsl_dialects.lowering.global_plan import (
    BufferRepresentation,
    CycleInterval,
    PlanDemand,
    RegionAlternative,
    TransitionAlternative,
    ValueRepresentation,
)


PLAIN = ValueRepresentation("memory", "row_major", "i8", encoding="plain")
PACKED = ValueRepresentation("array", "blocked", "i8", encoding="packed")


def _program() -> DispatchProgram:
    return DispatchProgram(
        entry="forward",
        args=[0],
        buffers={
            "b0": Buffer("b0", [32, 32], "i8", "arg", 0),
            "b1": Buffer("b1", [32, 32], "i8", "intermediate"),
            "b2": Buffer("b2", [32, 32], "i8", "intermediate"),
        },
        nodes=[
            Node("dispatch", "producer", ["b0"], ["b1"], captures=[]),
            Node("dispatch", "consumer", ["b1"], ["b2"], captures=[]),
        ],
        results=["b2"],
    )


def _alt(ident: str, nodes: tuple[int, ...], cycles: float, *,
         inputs: tuple[tuple[str, ValueRepresentation], ...],
         outputs: tuple[tuple[str, ValueRepresentation], ...],
         placement: str = "array") -> RegionAlternative:
    return RegionAlternative(
        id=ident,
        nodes=nodes,
        implementation=ident,
        placement=placement,
        cycles=CycleInterval.point(cycles, "fixture measurement"),
        inputs=tuple(BufferRepresentation(name, rep) for name, rep in inputs),
        outputs=tuple(BufferRepresentation(name, rep) for name, rep in outputs),
        demands=(PlanDemand("compute", cycles, "cycles", provenance="fixture"),),
    )


@dataclass
class SerialAdapter:
    alternatives: tuple[RegionAlternative, ...]
    transition_cycles: float = 100.0

    def region_alternatives(self, program):
        return self.alternatives

    def boundary_representation(self, program, buffer, direction):
        return PLAIN

    def transition(self, program, *, buffer, producer, consumer, source, destination):
        kind = "encoding" if source.encoding != destination.encoding else "movement"
        return TransitionAlternative(
            id=f"{buffer}:{producer.id if producer else 'input'}:"
               f"{consumer.id if consumer else 'output'}",
            kind=kind,
            buffer=buffer,
            producer=producer.id if producer else None,
            consumer=consumer.id if consumer else None,
            source=source,
            destination=destination,
            cycles=CycleInterval.point(self.transition_cycles, "paired conversion probe"),
            demands=(PlanDemand("movement", 1024, "bytes", provenance="exact shape"),),
        )

    def evaluate(self, program, selected, transitions):
        compute = sum(float(item.cycles.hi) for item in selected)
        movement = sum(float(item.cycles.hi) for item in transitions)
        total = compute + movement
        occupancy = OccupancySummary(
            total_cycles=total,
            busy_cycles=(("compute", compute), ("movement", movement)),
            compute_resources=("compute",),
            movement_resources=("movement",),
            overlap_cycles=0,
            overlap_available_cycles=min(compute, movement),
            idle_cycles=0,
            critical_path_cycles=total,
            movement_bytes=sum(d.amount for t in transitions for d in t.demands),
            movement_commands=len(transitions),
            encoding_transitions=sum(t.kind == "encoding" for t in transitions),
            provenance=("explicit serial fixture",),
        )
        return PlanEvaluation(CycleInterval.point(total, "explicit serial composition"), occupancy)

    def lower_bound(self, program, selected, uncovered_nodes, alternatives):
        # Fractional exact-cover relaxation: every selected region pays its whole lower endpoint;
        # every uncovered node pays the cheapest per-covered-node share. Transitions have a
        # non-negative cost, so omitting them keeps this an admissible lower bound.
        total = sum(float(item.cycles.lo) for item in selected)
        for node in uncovered_nodes:
            total += min(float(item.cycles.lo) / len(item.nodes)
                         for item in alternatives if node in item.nodes)
        return Bound(total, provenance=("serial fractional exact-cover relaxation",))

    def physical_floor(self, program, alternatives):
        total = sum(min(float(item.cycles.lo) / len(item.nodes)
                        for item in alternatives if node in item.nodes)
                    for node in range(len(program.nodes)))
        return Bound(total, provenance=("fixture physical resource floor",))


def test_global_choice_can_reject_the_locally_fast_kernel_when_encoding_dominates() -> None:
    fast_packed = _alt(
        "fast_packed", (0,), 10,
        inputs=(("b0", PLAIN),), outputs=(("b1", PACKED),))
    coherent = _alt(
        "coherent", (0,), 25,
        inputs=(("b0", PLAIN),), outputs=(("b1", PLAIN),))
    consumer = _alt(
        "consumer", (1,), 10,
        inputs=(("b1", PLAIN),), outputs=(("b2", PLAIN),))

    result = optimize_program(
        _program(), SerialAdapter((fast_packed, coherent, consumer)),
        policy=GlobalPlanPolicy(timeout_s=1))

    assert result.resolved
    assert result.plan is not None
    assert [item.id for item in result.plan.selected] == ["coherent", "consumer"]
    assert result.plan.transitions == ()
    assert result.plan.cycles.hi == 35
    assert result.physical_attainment == 20 / 35


def test_fused_region_removes_intermediate_movement_and_wins_globally() -> None:
    producer = _alt(
        "producer", (0,), 15,
        inputs=(("b0", PLAIN),), outputs=(("b1", PACKED),))
    consumer = _alt(
        "consumer", (1,), 15,
        inputs=(("b1", PLAIN),), outputs=(("b2", PLAIN),))
    fused = _alt(
        "fused", (0, 1), 18,
        inputs=(("b0", PLAIN),), outputs=(("b2", PLAIN),))

    result = optimize_program(
        _program(), SerialAdapter((producer, consumer, fused)),
        policy=GlobalPlanPolicy(timeout_s=1))

    assert result.plan is not None
    assert [item.id for item in result.plan.selected] == ["fused"]
    assert result.plan.transitions == ()
    assert result.plan.cycles.hi == 18
    assert result.optimality_gap == 0


def test_missing_transition_rejects_only_that_candidate_not_the_complete_fallback() -> None:
    class RefusingAdapter(SerialAdapter):
        def transition(self, *args, **kwargs):
            return None

    packed = _alt(
        "packed", (0,), 1,
        inputs=(("b0", PLAIN),), outputs=(("b1", PACKED),))
    fallback = _alt(
        "fallback", (0,), 20,
        inputs=(("b0", PLAIN),), outputs=(("b1", PLAIN),))
    consumer = _alt(
        "consumer", (1,), 10,
        inputs=(("b1", PLAIN),), outputs=(("b2", PLAIN),))

    result = optimize_program(
        _program(), RefusingAdapter((packed, fallback, consumer)),
        policy=GlobalPlanPolicy(timeout_s=1))

    assert result.resolved
    assert result.plan is not None
    assert [item.id for item in result.plan.selected] == ["fallback", "consumer"]
    assert any("no transition establishes" in reason for reason in result.refusals)


def test_unknown_roofline_never_reads_as_attainment() -> None:
    class UnknownFloorAdapter(SerialAdapter):
        def physical_floor(self, program, alternatives):
            return Bound.unknown("a measured movement peak")

    producer = _alt(
        "producer", (0,), 10,
        inputs=(("b0", PLAIN),), outputs=(("b1", PLAIN),))
    consumer = _alt(
        "consumer", (1,), 10,
        inputs=(("b1", PLAIN),), outputs=(("b2", PLAIN),))
    result = optimize_program(
        _program(), UnknownFloorAdapter((producer, consumer)),
        policy=GlobalPlanPolicy(timeout_s=1))

    assert not result.resolved
    assert result.plan is not None
    assert result.physical_attainment is None
    assert result.physical_floor.missing == ("a measured movement peak",)


def test_adapter_cost_exception_becomes_a_refusal_not_a_planner_crash() -> None:
    producer = _alt(
        "producer", (0,), 10,
        inputs=(("b0", PLAIN),), outputs=(("b1", PLAIN),))
    consumer = _alt(
        "consumer", (1,), 10,
        inputs=(("b1", PLAIN),), outputs=(("b2", PLAIN),))
    adapter = SerialAdapter((producer, consumer))

    def broken(*_args, **_kwargs):
        raise RuntimeError("missing calibration")

    adapter.evaluate = broken
    result = optimize_program(_program(), adapter, policy=GlobalPlanPolicy(timeout_s=1))

    assert not result.resolved
    assert any("missing calibration" in item for item in result.refusals)


def test_uncalibrated_structural_alternative_cannot_enter_cycle_ranking() -> None:
    producer = _alt("producer", (0,), 10, inputs=(("b0", PLAIN),), outputs=(("b1", PLAIN),))
    producer = replace(producer, cycles=CycleInterval.unknown("mechanism probe absent"))
    consumer = _alt("consumer", (1,), 10, inputs=(("b1", PLAIN),), outputs=(("b2", PLAIN),))
    adapter = SerialAdapter((producer, consumer))
    adapter.physical_floor = lambda *_: Bound(0, provenance=("nonnegative resource time",))
    result = optimize_program(_program(), adapter)
    assert result.plan is None
    assert any("needs calibration before cycle ranking" in reason for reason in result.refusals)
