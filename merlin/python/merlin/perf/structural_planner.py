"""Bounded exact-cover selection using explicit non-cycle structural objectives.

Adapters own composition: the core never adds bytes, predicts cycles or assumes
overlap. Pareto dominance minimizes the named integer axes independently. A
frontier is relative to the supplied alternatives and objective, not hardware
performance. Existing cycle-ranking ``global_planner.optimize_program`` is unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from typing import Protocol, Sequence

from merlin.perf.global_planner import PlanRefusal, _representations_and_transitions
from merlin.perf.structural_transitions import StructuralTransitionAdapterView
from merlin.xdsl_dialects.lowering.dispatch_program import DispatchProgram, verify_program
from merlin.xdsl_dialects.lowering.global_plan import (
    CycleInterval, GlobalPlan, RegionAlternative, TransitionAlternative,
    ValueRepresentation, verify_global_plan,
)


@dataclass(frozen=True)
class StructuralAxis:
    name: str
    unit: str

    def __post_init__(self):
        if not self.name.strip() or not self.unit.strip():
            raise ValueError("a structural axis requires a name and unit")
        if self.unit.lower() in {"cycle", "cycles", "second", "seconds"}:
            raise ValueError("timing objectives belong to the cycle-ranking planner")


@dataclass(frozen=True)
class StructuralEvaluation:
    values: tuple[int, ...]
    provenance: tuple[str, ...]

    def __post_init__(self):
        if any(type(value) is not int or value < 0 for value in self.values):
            raise ValueError("structural quantities must be nonnegative exact integers")
        if not self.provenance or any(not item.strip() for item in self.provenance):
            raise ValueError("structural composition requires explicit evidence")


class StructuralPlanningAdapter(Protocol):
    def region_alternatives(self, program: DispatchProgram) -> Sequence[RegionAlternative]: ...
    def boundary_representation(self, program: DispatchProgram, buffer: str,
                                direction: str) -> ValueRepresentation: ...
    def transition(self, program: DispatchProgram, *, buffer: str,
                   producer: RegionAlternative | None, consumer: RegionAlternative | None,
                   source: ValueRepresentation, destination: ValueRepresentation
                   ) -> TransitionAlternative | None: ...
    def evaluate_structure(self, program: DispatchProgram, selected: Sequence[RegionAlternative],
                           transitions: Sequence[TransitionAlternative],
                           axes: Sequence[StructuralAxis]) -> StructuralEvaluation: ...


@dataclass(frozen=True)
class StructuralPolicy:
    timeout_s: float = 30.0
    max_states: int = 10000
    max_frontier: int = 128

    def __post_init__(self):
        if not 0 < self.timeout_s <= 300 or self.max_states <= 0 or self.max_frontier <= 0:
            raise ValueError("structural search requires positive bounded time/state/frontier limits")


@dataclass(frozen=True)
class StructuralChoice:
    plan: GlobalPlan
    evaluation: StructuralEvaluation


@dataclass(frozen=True)
class StructuralResult:
    axes: tuple[StructuralAxis, ...]
    frontier: tuple[StructuralChoice, ...]
    expanded_states: int
    complete_plans: int
    exhausted: bool
    elapsed_s: float
    stop_reason: str
    refusals: tuple[str, ...]
    equivalent_vectors_collapsed: int = 0
    frontier_truncated: bool = False

    def to_dict(self):
        return {
            "schema": "structural_plan_result_v1",
            "status": "selected_structural_frontier" if self.frontier else "refused",
            "axes": [{"name": axis.name, "unit": axis.unit} for axis in self.axes],
            "frontier": [{"plan": item.plan.to_dict(), "values": list(item.evaluation.values),
                          "provenance": list(item.evaluation.provenance)} for item in self.frontier],
            "expanded_states": self.expanded_states, "complete_plans": self.complete_plans,
            "exhausted": self.exhausted, "elapsed_s": self.elapsed_s,
            "stop_reason": self.stop_reason, "refusals": list(self.refusals),
            "equivalent_vectors_collapsed": self.equivalent_vectors_collapsed,
            "frontier_truncated": self.frontier_truncated,
            "complete_frontier_for_supplied_alternatives": self.exhausted and not self.refusals and not self.frontier_truncated,
            "cycles": None, "performance_optimality": "UNPROVEN",
            "scope": "one representative per nondominated structural cost vector; no timing or physical movement inference",
        }


def optimize_structure(program: DispatchProgram, adapter: StructuralPlanningAdapter, *,
                       axes: Sequence[StructuralAxis], policy: StructuralPolicy | None = None
                       ) -> StructuralResult:
    """Enumerate compatible exact covers and retain nondominated structural costs."""
    objective = tuple(axes)
    if not objective or len({axis.name for axis in objective}) != len(objective):
        raise ValueError("structural objective axes must be nonempty and uniquely named")
    policy = policy or StructuralPolicy()
    started = monotonic()
    problems = verify_program(program)
    alternatives = ()
    if not problems:
        try:
            alternatives = tuple(adapter.region_alternatives(program))
        except Exception as error:
            problems.append(f"alternative enumeration failed: {type(error).__name__}: {error}")
    ids, by_anchor = set(), {}
    all_nodes = frozenset(range(len(program.nodes)))
    for item in alternatives:
        if item.id in ids or any(index not in all_nodes for index in item.nodes):
            problems.append(f"duplicate or out-of-range alternative {item.id!r}")
        ids.add(item.id)
        by_anchor.setdefault(item.nodes[0], []).append(item)
    if set().union(*(set(item.nodes) for item in alternatives)) != set(all_nodes):
        problems.append("alternatives do not cover the complete source graph")
    if problems:
        return StructuralResult(objective, (), 0, 0, True, monotonic()-started,
                                "invalid_alternatives", tuple(problems))
    stack = [(frozenset(), ())]
    frontier = []
    expanded = complete = collapsed = 0
    truncated = False
    stop = "exhausted"
    while stack:
        if monotonic()-started >= policy.timeout_s:
            stop = "timeout"
            break
        if expanded >= policy.max_states:
            stop = "state_limit"
            break
        covered, selected = stack.pop()
        expanded += 1
        if covered != all_nodes:
            anchor = min(all_nodes-covered)
            for item in reversed(by_anchor.get(anchor, ())):
                nodes = frozenset(item.nodes)
                if not nodes.intersection(covered):
                    stack.append((covered | nodes, (*selected, item)))
            continue
        complete += 1
        try:
            transitions = _representations_and_transitions(program, selected, StructuralTransitionAdapterView(adapter))
            plan = GlobalPlan(tuple(sorted(selected, key=lambda item: item.nodes)),
                tuple(sorted(transitions, key=lambda item: item.id)),
                CycleInterval.unknown("non-cycle structural selection has no timing model"),
                notes=("selected using explicit structural axes, not cycle costs",))
            invalid = verify_global_plan(program, plan)
            if invalid:
                raise PlanRefusal("; ".join(invalid))
            cost = adapter.evaluate_structure(program, selected, transitions, objective)
            if not isinstance(cost, StructuralEvaluation) or len(cost.values) != len(objective):
                raise PlanRefusal("adapter returned the wrong structural objective dimensionality")
        except Exception as error:
            problems.append(f"complete-plan refusal: {type(error).__name__}: {error}")
            continue
        if any(item.evaluation.values == cost.values for item in frontier):
            collapsed += 1
            continue
        if any(all(a <= b for a, b in zip(item.evaluation.values, cost.values)) for item in frontier):
            continue
        frontier = [item for item in frontier
                    if not all(a <= b for a, b in zip(cost.values, item.evaluation.values))]
        frontier.append(StructuralChoice(plan, cost))
        frontier.sort(key=lambda item: item.evaluation.values)
        if len(frontier) > policy.max_frontier:
            truncated = True
            frontier = frontier[:policy.max_frontier]
    return StructuralResult(objective, tuple(frontier), expanded, complete, not stack,
        monotonic()-started, stop, tuple(dict.fromkeys(problems)), collapsed, truncated)
