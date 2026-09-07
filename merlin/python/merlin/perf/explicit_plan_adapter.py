"""Data-driven adapter for the whole-program planner.

This adapter is the stable join between a compiler/target plugin and the target-neutral search.  The
plugin supplies alternatives, boundary encodings, explicit transition rules, resource kinds, and the
composition evidence.  The shared code contains no instruction, target, model, or geometry literals.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from merlin.perf.decompose import ResourceKind
from merlin.perf.envelope import Basis, ResourceTime, compose
from merlin.perf.global_planner import (
    Bound,
    OccupancySummary,
    PlanEvaluation,
    PlanRefusal,
)
from merlin.perf.headroom import Composition
from merlin.xdsl_dialects.lowering.dispatch_program import DispatchProgram
from merlin.xdsl_dialects.lowering.global_plan import (
    CycleInterval,
    PlanDemand,
    RegionAlternative,
    ResourceOccupancy,
    TransitionAlternative,
    ValueRepresentation,
)


@dataclass(frozen=True)
class TransitionRule:
    """An exact representation conversion measured or derived by a target plugin."""

    id: str
    source: ValueRepresentation
    destination: ValueRepresentation
    cycles: CycleInterval
    resource: str
    moved_bytes: float
    commands: int
    kind: str = "encoding"
    materializes: bool = True
    provenance: str = ""

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.resource.strip() or not self.kind.strip():
            raise ValueError("a transition rule must name its id, resource, and kind")
        if self.moved_bytes < 0 or self.commands < 0:
            raise ValueError("transition bytes and commands must be non-negative")


EventBuilder = Callable[[DispatchProgram, Sequence[RegionAlternative],
                         Sequence[TransitionAlternative], str], Any]


@dataclass
class ExplicitPlanningAdapter:
    """A target adapter whose every fact is supplied as data.

    ``event_builder`` may return an :class:`~merlin.perf.activity_schedule.ActivityTimeline` for an
    exact dependency/resource schedule.  Without one, evaluation uses the explicitly supplied
    generalized composition over aggregate resource busy times.  That fallback does not invent idle
    or critical-path evidence: those fields remain unresolved in the occupancy summary.
    """

    alternatives: tuple[RegionAlternative, ...]
    boundaries: Mapping[tuple[str, str], ValueRepresentation]
    transition_rules: tuple[TransitionRule, ...]
    resource_kinds: Mapping[str, ResourceKind]
    composition: Composition
    composition_eta: float
    physical: Bound
    event_builder: EventBuilder | None = None
    composition_provenance: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.composition, Composition):
            raise TypeError("ExplicitPlanningAdapter requires an explicitly derived Composition")
        if not 0 <= self.composition_eta <= 1:
            raise ValueError("composition_eta must be in [0, 1]")
        if not self.composition_provenance.strip():
            raise ValueError("composition provenance is required")

    def region_alternatives(self, program: DispatchProgram) -> Sequence[RegionAlternative]:
        return self.alternatives

    def boundary_representation(self, program: DispatchProgram, buffer: str,
                                direction: str) -> ValueRepresentation:
        key = (buffer, direction)
        if key not in self.boundaries:
            raise PlanRefusal(f"no {direction} boundary representation for buffer {buffer!r}")
        return self.boundaries[key]

    def transition(self, program: DispatchProgram, *, buffer: str,
                   producer: RegionAlternative | None, consumer: RegionAlternative | None,
                   source: ValueRepresentation,
                   destination: ValueRepresentation) -> TransitionAlternative | None:
        matches = [rule for rule in self.transition_rules
                   if rule.source == source and rule.destination == destination]
        if not matches:
            return None
        if len(matches) > 1:
            raise PlanRefusal(
                f"multiple transition rules connect the same representations: "
                f"{[rule.id for rule in matches]}")
        rule = matches[0]
        metadata = (("commands", str(rule.commands)), ("resource", rule.resource))
        return TransitionAlternative(
            id=f"{rule.id}:{buffer}:{producer.id if producer else 'input'}:"
               f"{consumer.id if consumer else 'output'}",
            kind=rule.kind,
            buffer=buffer,
            producer=producer.id if producer else None,
            consumer=consumer.id if consumer else None,
            source=source,
            destination=destination,
            cycles=rule.cycles,
            demands=(PlanDemand(rule.resource, rule.moved_bytes, "bytes", basis="moved",
                                provenance=rule.provenance),),
            occupancy=(ResourceOccupancy(rule.resource, rule.cycles, rule.provenance),),
            materializes=rule.materializes,
            metadata=metadata,
        )

    def _busy(self, selected: Sequence[RegionAlternative],
              transitions: Sequence[TransitionAlternative], endpoint: str) -> dict[str, float]:
        busy: dict[str, float] = {}
        for alternative in selected:
            if not alternative.occupancy:
                raise PlanRefusal(
                    f"alternative {alternative.id!r} has no per-resource occupancy evidence")
            for activity in alternative.occupancy:
                value = getattr(activity.cycles, endpoint)
                busy[activity.resource] = busy.get(activity.resource, 0.0) + float(value)
        for transition in transitions:
            if not transition.occupancy:
                raise PlanRefusal(
                    f"transition {transition.id!r} has no per-resource occupancy evidence")
            for activity in transition.occupancy:
                value = getattr(activity.cycles, endpoint)
                busy[activity.resource] = busy.get(activity.resource, 0.0) + float(value)
        absent = sorted(set(busy) - set(self.resource_kinds))
        if absent:
            raise PlanRefusal(f"resource kind is unestablished for {absent}")
        return busy

    def _compose(self, busy: Mapping[str, float]) -> float:
        times = tuple(ResourceTime(
            resource=name,
            kind=self.resource_kinds[name],
            cycles=float(cycles),
            unit="cycles",
            basis=Basis.MOVED,
            provenance=self.composition_provenance,
        ) for name, cycles in sorted(busy.items()))
        result = compose(times, operator=self.composition, eta=self.composition_eta)
        if not result.known:
            raise PlanRefusal(f"resource composition unresolved: {result.unresolved}")
        return float(result.cycles)

    def evaluate(self, program: DispatchProgram, selected: Sequence[RegionAlternative],
                 transitions: Sequence[TransitionAlternative]) -> PlanEvaluation:
        lo_busy = self._busy(selected, transitions, "lo")
        hi_busy = self._busy(selected, transitions, "hi")
        lo, hi = self._compose(lo_busy), self._compose(hi_busy)

        if self.event_builder is not None:
            lo_timeline = self.event_builder(program, selected, transitions, "lo")
            hi_timeline = self.event_builder(program, selected, transitions, "hi")
            occupancy = hi_timeline.occupancy(provenance=(self.composition_provenance,))

            # The aggregate composition is an independently derived lower bound. Dependencies,
            # finite queues, and pipeline fill may lengthen the emitted schedule, so equality is
            # neither required nor generally true.  The useful completeness check is instead that
            # every event's resource occupancy equals the alternative/transition work the planner
            # priced.  Missing activity cannot disappear behind a plausible total.
            for endpoint, timeline, expected, composed in (
                    ("lo", lo_timeline, lo_busy, lo),
                    ("hi", hi_timeline, hi_busy, hi)):
                actual = timeline.occupancy().busy
                names = sorted(set(expected) | set(actual))
                disagreement = [
                    name for name in names
                    if abs(expected.get(name, 0.0) - actual.get(name, 0.0)) > 1e-9
                ]
                if disagreement:
                    raise PlanRefusal(
                        f"{endpoint} event timeline does not account for the priced occupancy of "
                        f"resource(s) {disagreement}: expected {expected}, observed {actual}")
                if timeline.total_cycles + 1e-9 < composed:
                    raise PlanRefusal(
                        f"{endpoint} event timeline is {timeline.total_cycles:g} cycles, below the "
                        f"explicit {self.composition.value} resource bound {composed:g}")
            lo, hi = lo_timeline.total_cycles, hi_timeline.total_cycles
            if hi + 1e-9 < lo:
                raise PlanRefusal(
                    f"event timelines invert the cost interval: lo={lo:g}, hi={hi:g}")
        else:
            compute = tuple(sorted(name for name, kind in self.resource_kinds.items()
                                   if kind is ResourceKind.COMPUTE and name in hi_busy))
            movement = tuple(sorted(name for name, kind in self.resource_kinds.items()
                                    if kind is ResourceKind.MOVEMENT and name in hi_busy))
            engine_sum = sum(hi_busy[name] for name in hi_busy
                             if self.resource_kinds[name].is_engine)
            overlap = max(0.0, engine_sum - hi)
            vals = sorted((hi_busy[name] for name in hi_busy
                           if self.resource_kinds[name].is_engine), reverse=True)
            available = sum(vals[1:]) if len(vals) > 1 else 0.0
            occupancy = OccupancySummary(
                total_cycles=hi,
                busy_cycles=tuple(sorted(hi_busy.items())),
                compute_resources=compute,
                movement_resources=movement,
                overlap_cycles=overlap,
                overlap_available_cycles=available,
                idle_cycles=None,
                critical_path_cycles=None,
                movement_bytes=sum(d.amount for transition in transitions
                                   for d in transition.demands if d.unit == "bytes"),
                movement_commands=sum(int(dict(t.metadata).get("commands", "0"))
                                      for t in transitions),
                encoding_transitions=sum(t.kind == "encoding" for t in transitions),
                provenance=(self.composition_provenance,),
                missing=("an emitted dependency/resource timeline for idle and critical path",),
            )
        interval = CycleInterval(lo, hi, provenance=(self.composition_provenance,))
        return PlanEvaluation(interval, occupancy)

    def lower_bound(self, program: DispatchProgram, selected: Sequence[RegionAlternative],
                    uncovered_nodes: frozenset[int],
                    alternatives: Sequence[RegionAlternative]) -> Bound:
        if not self.physical.resolved:
            return self.physical
        # The physical envelope applies to every complete implementation. Selected alternatives can
        # only raise it. A stronger target-specific relaxation may replace this adapter; retaining the
        # physical floor is always admissible and keeps the reported optimality gap honest.
        return Bound(float(self.physical.cycles),
                     provenance=self.physical.provenance + ("whole-program physical relaxation",))

    def physical_floor(self, program: DispatchProgram,
                       alternatives: Sequence[RegionAlternative]) -> Bound:
        return self.physical
