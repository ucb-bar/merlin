"""Roofline-guided whole-program implementation selection.

The optimization unit here is a complete :class:`DispatchProgram`, never a capsule.  Target adapters
enumerate legal region implementations, explicit representation transitions, and an evidence-backed
composition/evaluation.  The shared search chooses a compatible exact cover of the program and keeps
an admissible lower bound for every unexplored state, so a timeout reports an optimality gap instead
of silently turning a bounded search into a claim of optimality.

No composition operator is defaulted in this module.  In particular, region and movement costs are
not summed by the core: whether engines serialize or overlap is a property the adapter must derive
and express through :meth:`TargetPlanningAdapter.evaluate` and ``lower_bound``.
"""
from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence, runtime_checkable

from merlin.xdsl_dialects.lowering.dispatch_program import DispatchProgram, verify_program
from merlin.xdsl_dialects.lowering.global_plan import (
    CycleInterval,
    GlobalPlan,
    RegionAlternative,
    TransitionAlternative,
    ValueRepresentation,
    verify_global_plan,
)

GLOBAL_PLAN_RESULT_SCHEMA = "global_plan_result_v1"


class PlanRefusal(ValueError):
    """A plan cannot be composed without guessing a missing legality or cost fact."""


@dataclass(frozen=True)
class Bound:
    """One admissible lower bound, with the evidence that licenses it."""

    cycles: float | None
    provenance: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.cycles is None:
            if not self.missing:
                raise ValueError("an unresolved bound must say what is missing")
            return
        if isinstance(self.cycles, bool) or not isinstance(self.cycles, (int, float)):
            raise TypeError("bound cycles must be numeric")
        if float(self.cycles) < 0:
            raise ValueError("a cycle lower bound cannot be negative")
        object.__setattr__(self, "cycles", float(self.cycles))

    @property
    def resolved(self) -> bool:
        return self.cycles is not None

    @classmethod
    def unknown(cls, *missing: str) -> "Bound":
        return cls(None, missing=tuple(str(item) for item in missing if str(item).strip()))

    def to_dict(self) -> dict[str, Any]:
        return {"cycles": self.cycles, "resolved": self.resolved,
                "provenance": list(self.provenance), "missing": list(self.missing)}


@dataclass(frozen=True)
class OccupancySummary:
    """Minimal whole-plan evidence for keeping engines occupied and hiding latency."""

    total_cycles: float
    busy_cycles: tuple[tuple[str, float], ...]
    compute_resources: tuple[str, ...] = ()
    movement_resources: tuple[str, ...] = ()
    overlap_cycles: float | None = None
    overlap_available_cycles: float | None = None
    idle_cycles: float | None = None
    critical_path_cycles: float | None = None
    movement_bytes: float | None = None
    movement_commands: int | None = None
    encoding_transitions: int | None = None
    provenance: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    movement_elapsed_cycles: float | None = None

    def __post_init__(self) -> None:
        if self.total_cycles < 0:
            raise ValueError("occupancy total_cycles cannot be negative")
        if tuple(sorted(self.busy_cycles)) != self.busy_cycles:
            raise ValueError("busy_cycles must be sorted for stable reporting")
        if len(dict(self.busy_cycles)) != len(self.busy_cycles):
            raise ValueError("resource busy counters must be unique")
        for resources in (self.compute_resources, self.movement_resources):
            if len(set(resources)) != len(resources):
                raise ValueError("occupancy resource roles must not contain duplicates")
        if any(value < 0 for _, value in self.busy_cycles):
            raise ValueError("resource busy cycles cannot be negative")
        for name in ("overlap_cycles", "overlap_available_cycles", "idle_cycles",
                     "critical_path_cycles", "movement_bytes", "movement_elapsed_cycles"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.movement_elapsed_cycles is not None:
            if self.movement_elapsed_cycles > self.total_cycles:
                raise ValueError("movement elapsed cycles cannot exceed the timeline")
            if self.overlap_cycles is not None and self.overlap_cycles > self.movement_elapsed_cycles:
                raise ValueError("overlap cannot exceed movement elapsed cycles")
        if self.movement_commands is not None and self.movement_commands < 0:
            raise ValueError("movement_commands cannot be negative")
        if self.encoding_transitions is not None and self.encoding_transitions < 0:
            raise ValueError("encoding_transitions cannot be negative")
        # A declared engine absent from the instrument is not an idle engine.
        # Preserve the declared denominator and expose the missing observation.
        absent = sorted(set(self.compute_resources + self.movement_resources) - self.busy.keys())
        object.__setattr__(self, "missing", tuple(dict.fromkeys(
            (*self.missing, *(f"busy cycles for declared resource {name}" for name in absent)))))

    @property
    def busy(self) -> dict[str, float]:
        return dict(self.busy_cycles)

    @property
    def compute_busy_cycles(self) -> float | None:
        if not self.compute_resources or any(name not in self.busy for name in self.compute_resources):
            return None
        return sum(self.busy[name] for name in self.compute_resources)

    @property
    def compute_utilization(self) -> float | None:
        busy = self.compute_busy_cycles
        if busy is None or self.total_cycles <= 0:
            return None
        # Multiple compute engines may overlap, so utilization can exceed one if treated as one
        # engine.  Normalize by the declared number of independently occupiable resources.
        capacity = self.total_cycles * len(self.compute_resources)
        return min(1.0, busy / capacity) if capacity else None

    @property
    def latency_hiding_efficiency(self) -> float | None:
        if self.overlap_cycles is None or self.overlap_available_cycles is None:
            return None
        if self.overlap_available_cycles == 0:
            return 1.0
        return min(1.0, self.overlap_cycles / self.overlap_available_cycles)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_cycles": self.total_cycles,
            "busy_cycles": dict(self.busy_cycles),
            "compute_resources": list(self.compute_resources),
            "movement_resources": list(self.movement_resources),
            "movement_elapsed_cycles": self.movement_elapsed_cycles,
            "compute_utilization": self.compute_utilization,
            "overlap_cycles": self.overlap_cycles,
            "overlap_available_cycles": self.overlap_available_cycles,
            "latency_hiding_efficiency": self.latency_hiding_efficiency,
            "idle_cycles": self.idle_cycles,
            "critical_path_cycles": self.critical_path_cycles,
            "movement_bytes": self.movement_bytes,
            "movement_commands": self.movement_commands,
            "encoding_transitions": self.encoding_transitions,
            "provenance": list(self.provenance),
            "missing": list(self.missing),
        }


@dataclass(frozen=True)
class PlanEvaluation:
    cycles: CycleInterval
    occupancy: OccupancySummary
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.cycles.resolved:
            raise ValueError("a complete plan evaluation must resolve its cycle interval")
        if abs(float(self.cycles.hi) - self.occupancy.total_cycles) > 1e-9:
            raise ValueError("occupancy total_cycles must equal the conservative cycle estimate")


@runtime_checkable
class TargetPlanningAdapter(Protocol):
    """Target edge for the shared whole-program planner.

    Every method is required.  A target that cannot establish one of them returns an unresolved
    :class:`Bound` or raises :class:`PlanRefusal`; the core never fills the absence with a guessed
    transition, resource rate, or overlap policy.
    """

    def region_alternatives(self, program: DispatchProgram) -> Sequence[RegionAlternative]: ...

    def boundary_representation(self, program: DispatchProgram, buffer: str,
                                direction: str) -> ValueRepresentation: ...

    def transition(self, program: DispatchProgram, *, buffer: str,
                   producer: RegionAlternative | None, consumer: RegionAlternative | None,
                   source: ValueRepresentation,
                   destination: ValueRepresentation) -> TransitionAlternative | None: ...

    def evaluate(self, program: DispatchProgram, selected: Sequence[RegionAlternative],
                 transitions: Sequence[TransitionAlternative]) -> PlanEvaluation: ...

    def lower_bound(self, program: DispatchProgram, selected: Sequence[RegionAlternative],
                    uncovered_nodes: frozenset[int],
                    alternatives: Sequence[RegionAlternative]) -> Bound: ...

    def physical_floor(self, program: DispatchProgram,
                       alternatives: Sequence[RegionAlternative]) -> Bound: ...


@dataclass(frozen=True)
class GlobalPlanPolicy:
    timeout_s: float = 60.0
    max_expanded_states: int = 100_000
    attainment_fraction: float = 0.90

    def __post_init__(self) -> None:
        if self.timeout_s <= 0 or self.timeout_s > 300:
            raise ValueError("the static global planner timeout must be in (0, 300] seconds")
        if self.max_expanded_states <= 0:
            raise ValueError("max_expanded_states must be positive")
        if not 0 < self.attainment_fraction <= 1:
            raise ValueError("attainment_fraction must be in (0, 1]")


@dataclass(frozen=True)
class GlobalPlanResult:
    plan: GlobalPlan | None
    physical_floor: Bound
    legal_floor: Bound
    solver_lower_bound: Bound
    optimality_gap: float | None
    expanded_states: int
    complete_plans: int
    elapsed_s: float
    timed_out: bool
    exhausted: bool
    refusals: tuple[str, ...] = ()
    occupancy: OccupancySummary | None = None
    schema: str = GLOBAL_PLAN_RESULT_SCHEMA

    @property
    def resolved(self) -> bool:
        # Candidate-specific refusals (for example an implementation that lacks a required encoding
        # transition) are useful coverage evidence but do not invalidate another complete plan.  The
        # two roofline bounds are the global completeness gate.
        return (self.plan is not None and self.legal_floor.resolved
                and self.physical_floor.resolved)

    @property
    def attainment(self) -> float | None:
        if self.plan is None or not self.legal_floor.resolved or not self.plan.cycles.resolved:
            return None
        cycles = float(self.plan.cycles.hi)
        if cycles == 0:
            return 1.0 if self.legal_floor.cycles == 0 else None
        return min(1.0, float(self.legal_floor.cycles) / cycles)

    @property
    def physical_attainment(self) -> float | None:
        if self.plan is None or not self.physical_floor.resolved or not self.plan.cycles.resolved:
            return None
        cycles = float(self.plan.cycles.hi)
        if cycles == 0:
            return 1.0 if self.physical_floor.cycles == 0 else None
        return min(1.0, float(self.physical_floor.cycles) / cycles)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "status": "resolved" if self.resolved else "refused",
            "plan": self.plan.to_dict() if self.plan is not None else None,
            "physical_floor": self.physical_floor.to_dict(),
            "legal_floor": self.legal_floor.to_dict(),
            "solver_lower_bound": self.solver_lower_bound.to_dict(),
            "optimality_gap": self.optimality_gap,
            "attainment": self.attainment,
            "physical_attainment": self.physical_attainment,
            "expanded_states": self.expanded_states,
            "complete_plans": self.complete_plans,
            "elapsed_s": self.elapsed_s,
            "timed_out": self.timed_out,
            "exhausted": self.exhausted,
            "refusals": list(self.refusals),
            "occupancy": self.occupancy.to_dict() if self.occupancy is not None else None,
        }


def _node_owners(selected: Sequence[RegionAlternative]) -> dict[int, RegionAlternative]:
    return {index: alternative for alternative in selected for index in alternative.nodes}


def _representations_and_transitions(
        program: DispatchProgram, selected: Sequence[RegionAlternative],
        adapter: TargetPlanningAdapter) -> tuple[TransitionAlternative, ...]:
    owners = _node_owners(selected)
    producer_node: dict[str, int] = {}
    for index, node in enumerate(program.nodes):
        for buffer in node.outputs:
            producer_node[buffer] = index

    transitions: list[TransitionAlternative] = []
    seen: set[tuple[str, str | None, str | None]] = set()

    def connect(buffer: str, producer: RegionAlternative | None,
                consumer: RegionAlternative | None) -> None:
        key = (buffer, producer.id if producer else None, consumer.id if consumer else None)
        if key in seen:
            return
        seen.add(key)
        if producer is None:
            source = adapter.boundary_representation(program, buffer, "input")
        else:
            source = producer.output_representation(buffer)
            if source is None:
                raise PlanRefusal(
                    f"alternative {producer.id!r} does not declare the representation of boundary "
                    f"output {buffer!r}")
        if consumer is None:
            destination = adapter.boundary_representation(program, buffer, "output")
        else:
            destination = consumer.input_representation(buffer)
            if destination is None:
                raise PlanRefusal(
                    f"alternative {consumer.id!r} does not declare the representation of boundary "
                    f"input {buffer!r}")
        if source == destination:
            return
        transition = adapter.transition(
            program, buffer=buffer, producer=producer, consumer=consumer,
            source=source, destination=destination)
        if transition is None:
            raise PlanRefusal(
                f"no transition establishes {buffer!r}: {source.to_dict()} -> "
                f"{destination.to_dict()}")
        if transition.source != source or transition.destination != destination:
            raise PlanRefusal(f"transition {transition.id!r} does not connect the requested encodings")
        transitions.append(transition)

    # Every original dataflow edge crossing selected alternatives must retain an explicit compatible
    # representation.  Edges internal to one fused alternative need no materialization.
    for consumer_index, node in enumerate(program.nodes):
        consumer = owners[consumer_index]
        for buffer in node.inputs:
            p_index = producer_node.get(buffer)
            producer = owners[p_index] if p_index is not None else None
            if producer is consumer:
                continue
            connect(buffer, producer, consumer)
    for buffer in program.results:
        p_index = producer_node.get(buffer)
        if p_index is None:
            raise PlanRefusal(f"program result {buffer!r} has no producing node")
        connect(buffer, owners[p_index], None)
    return tuple(transitions)


def _complete_plan(program: DispatchProgram, selected: Sequence[RegionAlternative],
                   adapter: TargetPlanningAdapter) -> tuple[GlobalPlan, PlanEvaluation]:
    transitions = _representations_and_transitions(program, selected, adapter)
    evaluation = adapter.evaluate(program, selected, transitions)
    plan = GlobalPlan(
        selected=tuple(sorted(selected, key=lambda item: item.nodes)),
        transitions=tuple(sorted(transitions, key=lambda item: item.id)),
        cycles=evaluation.cycles,
        notes=evaluation.notes,
    )
    problems = verify_global_plan(program, plan)
    if problems:
        raise PlanRefusal("invalid complete plan: " + "; ".join(problems))
    return plan, evaluation


@dataclass(order=True)
class _State:
    lower: float
    serial: int
    covered: frozenset[int] = field(compare=False)
    selected: tuple[RegionAlternative, ...] = field(compare=False)


def optimize_program(program: DispatchProgram, adapter: TargetPlanningAdapter, *,
                     policy: GlobalPlanPolicy | None = None) -> GlobalPlanResult:
    """Choose and evaluate a compatible whole-program plan under a bounded static search."""

    pol = policy or GlobalPlanPolicy()
    started = time.monotonic()
    dag_problems = verify_program(program)
    if dag_problems:
        problem = "invalid DispatchProgram: " + "; ".join(dag_problems)
        missing = Bound.unknown(problem)
        return GlobalPlanResult(None, missing, missing, missing, None, 0, 0, 0.0,
                                False, True, (problem,))

    try:
        alternatives = tuple(adapter.region_alternatives(program))
    except Exception as exc:  # adapter boundary: retain the target's refusal as evidence
        problem = f"alternative enumeration failed: {type(exc).__name__}: {exc}"
        missing = Bound.unknown(problem)
        return GlobalPlanResult(None, missing, missing, missing, None, 0, 0,
                                time.monotonic() - started, False, True, (problem,))

    refusals: list[str] = []
    n_nodes = len(program.nodes)
    by_anchor: dict[int, list[RegionAlternative]] = {}
    ids: set[str] = set()
    for alternative in alternatives:
        if not alternative.cycles.resolved:
            refusals.append(
                f"alternative {alternative.id!r} needs calibration before cycle ranking: "
                f"{alternative.cycles.missing}")
        if alternative.id in ids:
            refusals.append(f"duplicate alternative id {alternative.id!r}")
        ids.add(alternative.id)
        if any(index >= n_nodes for index in alternative.nodes):
            refusals.append(f"alternative {alternative.id!r} references a node outside the program")
            continue
        by_anchor.setdefault(alternative.nodes[0], []).append(alternative)
    absent = [index for index in range(n_nodes) if not any(index in alt.nodes for alt in alternatives)]
    if absent:
        refusals.append(f"no legal alternative covers node(s) {absent}")
    try:
        physical = adapter.physical_floor(program, alternatives)
    except Exception as exc:
        problem = f"physical-floor derivation failed: {type(exc).__name__}: {exc}"
        missing = Bound.unknown(problem)
        return GlobalPlanResult(None, missing, missing, missing, None, 0, 0,
                                time.monotonic() - started, False, True, (problem,))
    if not isinstance(physical, Bound):
        problem = "physical-floor derivation did not return a Bound"
        missing = Bound.unknown(problem)
        return GlobalPlanResult(None, missing, missing, missing, None, 0, 0,
                                time.monotonic() - started, False, True, (problem,))
    if refusals:
        missing = Bound.unknown(*refusals)
        return GlobalPlanResult(None, physical, missing, missing, None, 0, 0,
                                time.monotonic() - started, False, True, tuple(refusals))

    all_nodes = frozenset(range(n_nodes))
    try:
        root_bound = adapter.lower_bound(program, (), all_nodes, alternatives)
    except Exception as exc:
        problem = f"legal lower-bound derivation failed: {type(exc).__name__}: {exc}"
        missing = Bound.unknown(problem)
        return GlobalPlanResult(None, physical, missing, missing, None, 0, 0,
                                time.monotonic() - started, False, True, (problem,))
    if not isinstance(root_bound, Bound):
        problem = "legal lower-bound derivation did not return a Bound"
        missing = Bound.unknown(problem)
        return GlobalPlanResult(None, physical, missing, missing, None, 0, 0,
                                time.monotonic() - started, False, True, (problem,))
    if not root_bound.resolved:
        refusals.extend(root_bound.missing)
        # Non-negativity is an admissible search bound, but it is deliberately not published as the
        # legal roofline.  The unresolved adapter bound remains visible and blocks attainment.
        root_lower = 0.0
    else:
        root_lower = float(root_bound.cycles)

    heap: list[_State] = [_State(root_lower, 0, frozenset(), ())]
    serial = 1
    expanded = 0
    complete = 0
    best: GlobalPlan | None = None
    best_evaluation: PlanEvaluation | None = None
    best_hi = math.inf

    while heap and expanded < pol.max_expanded_states:
        if time.monotonic() - started >= pol.timeout_s:
            break
        state = heapq.heappop(heap)
        if state.lower >= best_hi:
            continue
        if state.covered == all_nodes:
            complete += 1
            try:
                plan, evaluation = _complete_plan(program, state.selected, adapter)
            except Exception as exc:  # target adapter boundary: one bad candidate is evidence
                refusals.append(f"complete-plan evaluation failed: {type(exc).__name__}: {exc}")
                continue
            if float(plan.cycles.hi) < best_hi:
                best, best_hi = plan, float(plan.cycles.hi)
                best_evaluation = evaluation
            continue

        expanded += 1
        anchor = min(all_nodes - state.covered)
        for alternative in by_anchor.get(anchor, ()):
            nodes = frozenset(alternative.nodes)
            if nodes & state.covered:
                continue
            covered = state.covered | nodes
            selected = state.selected + (alternative,)
            uncovered = all_nodes - covered
            try:
                bound = adapter.lower_bound(program, selected, uncovered, alternatives)
            except Exception as exc:
                refusals.append(
                    f"branch lower-bound derivation failed for {alternative.id!r}: "
                    f"{type(exc).__name__}: {exc}")
                continue
            if not isinstance(bound, Bound):
                refusals.append(
                    f"branch lower-bound derivation for {alternative.id!r} did not return a Bound")
                continue
            if not bound.resolved:
                refusals.extend(bound.missing)
                lower = 0.0
            else:
                lower = float(bound.cycles)
            if lower < best_hi:
                heapq.heappush(heap, _State(lower, serial, covered, selected))
                serial += 1

    elapsed = time.monotonic() - started
    timed_out = bool(heap) and elapsed >= pol.timeout_s
    exhausted = not heap
    state_limit = bool(heap) and expanded >= pol.max_expanded_states
    if state_limit:
        refusals.append(f"global-plan state limit {pol.max_expanded_states} reached")

    pending_lower = min((state.lower for state in heap), default=math.inf)
    if best is None:
        reason = "no complete compatible plan"
        refusals.append(reason)
        legal = Bound.unknown(reason, *tuple(dict.fromkeys(refusals)))
        solver = Bound(pending_lower, provenance=("adapter lower bound over unexplored states",)) \
            if math.isfinite(pending_lower) else Bound.unknown(reason)
        return GlobalPlanResult(None, physical, legal, solver, None, expanded, complete, elapsed,
                                timed_out, exhausted, tuple(dict.fromkeys(refusals)))

    # The best compatible implementation is no faster than the smallest admissible bound among the
    # incumbent and every unexplored state.  When the adapter bound was unresolved, legal attainment
    # remains unresolved even though the planner can still return the best complete plan it saw.
    incumbent_lo = float(best.cycles.lo)
    solver_lower = min(incumbent_lo, pending_lower) if math.isfinite(pending_lower) else incumbent_lo
    solver = Bound(solver_lower, provenance=("compatible-plan branch-and-bound",))
    if root_bound.resolved:
        legal = solver
    else:
        legal = Bound.unknown(*root_bound.missing)
    if physical.resolved and float(physical.cycles) > best_hi + 1e-9:
        problem = (f"selected plan upper bound {best_hi:g} is below the physical floor "
                   f"{float(physical.cycles):g}; cost evidence is unsound")
        refusals.append(problem)
        legal = Bound.unknown(problem)
    gap = max(0.0, (best_hi - solver_lower) / best_hi) if best_hi > 0 else 0.0
    return GlobalPlanResult(best, physical, legal, solver, gap, expanded, complete, elapsed,
                            timed_out, exhausted, tuple(dict.fromkeys(refusals)),
                            occupancy=(best_evaluation.occupancy
                                       if best_evaluation is not None else None))
