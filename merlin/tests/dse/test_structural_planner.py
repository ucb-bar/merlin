"""Non-cycle exact-cover search never disguises structural counts as timing."""
from dataclasses import replace
import pytest
from merlin.perf.structural_planner import (
    StructuralAxis, StructuralEvaluation, StructuralPolicy, optimize_structure,
)
from merlin.perf.global_planner import Bound, optimize_program
from merlin.xdsl_dialects.lowering.dispatch_program import Buffer, DispatchProgram, Node
from merlin.xdsl_dialects.lowering.global_plan import (
    BufferRepresentation, CycleInterval, RegionAlternative, ValueRepresentation,
)

REP = ValueRepresentation("tensor_ssa", "logical", "f32")
AXES = (StructuralAxis("interface_payload", "bytes"), StructuralAxis("driver_sites", "sites"))


def program():
    return DispatchProgram("forward", [0], {
        "x": Buffer("x", [4], "f32", "arg", 0),
        "a": Buffer("a", [4], "f32", "intermediate"),
        "y": Buffer("y", [4], "f32", "intermediate")},
        [Node("dispatch", "producer", ["x"], ["a"], captures=[]),
         Node("dispatch", "consumer", ["a"], ["y"], captures=[])], ["y"])


def alternative(name, nodes, inputs, outputs):
    return RegionAlternative(name, nodes, name, "tensor_ssa", CycleInterval.unknown("no timing oracle"),
        inputs=tuple(BufferRepresentation(value, REP) for value in inputs),
        outputs=tuple(BufferRepresentation(value, REP) for value in outputs))


class Adapter:
    def region_alternatives(self, program):
        return (alternative("producer", (0,), ("x",), ("a",)),
            alternative("consumer", (1,), ("a",), ("y",)),
            alternative("fused", (0, 1), ("x",), ("y",)))

    def boundary_representation(self, program, buffer, direction):
        return REP

    def transition(self, *args, **kwargs):
        return None

    def physical_floor(self, *args):
        return Bound.unknown("structural evidence is not a physical cycle floor")

    def evaluate_structure(self, program, selected, transitions, axes):
        assert axes == AXES
        return StructuralEvaluation((16*(len(selected)-1), len(selected)),
            ("fixture interface byte count and emitted driver sites",))


def test_real_alternatives_unknown_cycles_and_explicit_objective():
    result = optimize_structure(program(), Adapter(), axes=AXES)
    assert result.complete_plans == 2
    assert result.exhausted
    assert result.frontier[0].evaluation.values == (0, 1)
    assert result.frontier[0].plan.selected[0].id == "fused"
    assert not result.frontier[0].plan.cycles.resolved
    assert result.to_dict()["performance_optimality"] == "UNPROVEN"
    assert not optimize_program(program(), Adapter()).resolved


def test_pareto_does_not_scalarize_conflicting_units():
    class Tradeoff(Adapter):
        def evaluate_structure(self, program, selected, transitions, axes):
            return StructuralEvaluation((0, 9) if len(selected) == 1 else (16, 2), ("explicit tradeoff",))
    result = optimize_structure(program(), Tradeoff(), axes=AXES)
    assert {item.evaluation.values for item in result.frontier} == {(0, 9), (16, 2)}


def test_state_limit_not_complete_frontier():
    result = optimize_structure(program(), Adapter(), axes=AXES, policy=StructuralPolicy(max_states=1))
    assert not result.exhausted
    assert result.stop_reason == "state_limit"
    assert not result.to_dict()["complete_frontier_for_supplied_alternatives"]


def test_unknown_transition_is_refused_not_zero_cost():
    class WrongRep(Adapter):
        def boundary_representation(self, *args):
            return replace(REP, encoding="different")
    result = optimize_structure(program(), WrongRep(), axes=AXES)
    assert not result.frontier
    assert any("no transition" in row for row in result.refusals)


def test_no_implicit_missing_cost_composition():
    class Missing(Adapter):
        def evaluate_structure(self, *args):
            return StructuralEvaluation((1,), ("wrong dimension",))
    assert not optimize_structure(program(), Missing(), axes=AXES).frontier


def test_invalid_objectives_are_rejected():
    with pytest.raises(ValueError):
        StructuralAxis("latency", "cycles")
    with pytest.raises(ValueError):
        StructuralEvaluation((True,), ("not an exact count",))
    with pytest.raises(ValueError):
        optimize_structure(program(), Adapter(), axes=(AXES[0], AXES[0]))


def test_frontier_capacity_loss_is_explicit():
    class Tradeoff(Adapter):
        def evaluate_structure(self, program, selected, transitions, axes):
            return StructuralEvaluation((0, 9) if len(selected) == 1 else (16, 2), ("tradeoff",))
    result = optimize_structure(program(), Tradeoff(), axes=AXES, policy=StructuralPolicy(max_frontier=1))
    assert result.exhausted and result.frontier_truncated
    assert not result.to_dict()["complete_frontier_for_supplied_alternatives"]
