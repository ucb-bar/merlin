"""A selected whole-model plan must change the executable program through a verified target seam."""
from __future__ import annotations

from dataclasses import replace

import pytest

from merlin.runtime.program import build_program
from merlin.xdsl_dialects.lowering.dispatch_program import Buffer, DispatchProgram, Node
from merlin.xdsl_dialects.lowering.global_plan import (
    BufferRepresentation,
    CycleInterval,
    GlobalPlan,
    RegionAlternative,
    ValueRepresentation,
    TransitionAlternative,
)
from merlin.xdsl_dialects.lowering.global_plan_emission import (
    BoundaryMapping,
    EmittedComponent,
    GlobalPlanEmission,
    dispatch_digest,
    verify_global_plan_emission,
)


def _logical() -> DispatchProgram:
    return DispatchProgram(
        "forward", [0],
        {
            "b0": Buffer("b0", [4], "i8", "arg", 0),
            "b1": Buffer("b1", [4], "i8", "intermediate"),
            "b2": Buffer("b2", [4], "i8", "intermediate"),
        },
        [Node("dispatch", "producer", ["b0"], ["b1"], captures=[]),
         Node("dispatch", "consumer", ["b1"], ["b2"], captures=[])],
        ["b2"])


def _plan(program: DispatchProgram) -> GlobalPlan:
    rep = ValueRepresentation("memory", "linear", "i8")
    fused = RegionAlternative(
        "fused", (0, 1), "fused_kernel", "engine", CycleInterval.point(10, "fixture"),
        inputs=(BufferRepresentation("b0", rep),),
        outputs=(BufferRepresentation("b2", rep),))
    return GlobalPlan((fused,), (), CycleInterval.point(10, "fixture"))


class _Emitter:
    def __init__(self, *, hidden_node: bool = False):
        self.hidden_node = hidden_node

    def emit_global_plan(self, program: DispatchProgram, plan: GlobalPlan) -> GlobalPlanEmission:
        nodes = [Node("dispatch", "fused_kernel", ["x0"], ["x1"], captures=[])]
        if self.hidden_node:
            nodes.append(Node("view", "hidden_copy", ["x1"], ["x2"], captures=[]))
        emitted = DispatchProgram(
            "forward", [0],
            {"x0": Buffer("x0", [4], "i8", "arg", 0),
             "x1": Buffer("x1", [4], "i8", "intermediate"),
             **({"x2": Buffer("x2", [4], "i8", "intermediate")}
                if self.hidden_node else {})},
            nodes, ["x1"])
        return GlobalPlanEmission(
            emitted, plan.digest, dispatch_digest(program),
            (EmittedComponent("fused", (0,), (("b0", "x0"),), (("b2", "x1"),)),), (),
            (BoundaryMapping("input", "b0", "x0"),
             BoundaryMapping("output", "b2", "x1")),
            ("fixture target emitter",))


def test_global_plan_emitter_replaces_two_logical_dispatches_with_one_fused_dispatch() -> None:
    logical = _logical()
    plan = _plan(logical)

    program = build_program(
        logical, capability="engine", global_plan=plan, global_plan_emitter=_Emitter())

    assert [node.op for node in program.dispatch.nodes] == ["fused_kernel"]
    assert set(program.kernels) == {"fused_kernel"}
    assert program.stats["global_plan_emitted"] is True
    assert program.to_dict()["global_plan_emission"]["plan_digest"] == plan.digest


def test_plan_without_emitter_is_explicit_shadow_analysis() -> None:
    logical = _logical()
    program = build_program(logical, global_plan=_plan(logical))

    assert [node.op for node in program.dispatch.nodes] == ["producer", "consumer"]
    assert program.stats["global_plan_emitted"] is False


def test_emitter_cannot_hide_unaccounted_executable_work() -> None:
    logical = _logical()
    with pytest.raises(ValueError, match="unaccounted"):
        build_program(logical, global_plan=_plan(logical),
                      global_plan_emitter=_Emitter(hidden_node=True))


def _split_emission():
    logical = _logical()
    plain = ValueRepresentation("memory", "linear", "i8")
    packed = replace(plain, encoding="packed")
    cost = CycleInterval.point(10, "test evidence")
    producer = RegionAlternative(
        "producer", (0,), "producer_kernel", "engine", cost,
        inputs=(BufferRepresentation("b0", plain),),
        outputs=(BufferRepresentation("b1", packed),))
    consumer = RegionAlternative(
        "consumer", (1,), "consumer_kernel", "engine", cost,
        inputs=(BufferRepresentation("b1", plain),),
        outputs=(BufferRepresentation("b2", plain),))
    transition = TransitionAlternative(
        "unpack", "encoding", "b1", "producer", "consumer", packed, plain, cost)
    plan = GlobalPlan((producer, consumer), (transition,), cost)
    emitted = DispatchProgram("forward", [0], {
        "x0": Buffer("x0", [4], "i8", "arg", 0),
        "x1": Buffer("x1", [4], "i8", "intermediate"),
        "x2": Buffer("x2", [4], "i8", "intermediate"),
        "x3": Buffer("x3", [4], "i8", "intermediate"),
    }, [Node("dispatch", "producer_kernel", ["x0"], ["x1"], captures=[]),
        Node("dispatch", "unpack", ["x1"], ["x2"], captures=[]),
        Node("dispatch", "consumer_kernel", ["x2"], ["x3"], captures=[])], ["x3"])
    emission = GlobalPlanEmission(
        emitted, plan.digest, dispatch_digest(logical),
        (EmittedComponent("producer", (0,), (("b0", "x0"),), (("b1", "x1"),)),
         EmittedComponent("consumer", (2,), (("b1", "x2"),), (("b2", "x3"),))),
        (EmittedComponent("unpack", (1,), (("b1", "x1"),), (("b1", "x2"),)),),
        (BoundaryMapping("input", "b0", "x0"), BoundaryMapping("output", "b2", "x3")),
        ("test emitter",))
    return logical, plan, emission


def test_encoding_transition_preserves_full_producer_consumer_chain() -> None:
    logical, plan, emission = _split_emission()
    assert verify_global_plan_emission(logical, plan, emission) == []
    assert emission.receipt()["schema"] == "global_plan_emission_v2"


def test_owned_nodes_cannot_hide_a_bypassed_encoding_transition() -> None:
    logical, plan, emission = _split_emission()
    emission.dispatch.nodes[2].inputs = ["x1"]
    # Even an updated self-report must fail against the original logical edge.
    emission = replace(emission, regions=(emission.regions[0], replace(
        emission.regions[1], inputs=(("b1", "x1"),))))
    assert any("bypasses" in issue for issue in verify_global_plan_emission(logical, plan, emission))


def test_owned_nodes_cannot_hide_a_missing_model_input() -> None:
    logical = _logical()
    plan = _plan(logical)
    emission = _Emitter().emit_global_plan(logical, plan)
    emission.dispatch.nodes[0].inputs = []
    assert any("boundary differs" in issue
               for issue in verify_global_plan_emission(logical, plan, emission))


def test_mapping_is_required_even_when_node_ownership_is_complete() -> None:
    logical = _logical()
    plan = _plan(logical)
    emission = _Emitter().emit_global_plan(logical, plan)
    emission = replace(emission, regions=(EmittedComponent("fused", (0,)),))
    assert any("complete logical boundary" in issue
               for issue in verify_global_plan_emission(logical, plan, emission))


def test_encoding_declaration_must_agree_with_the_actual_plan_edge() -> None:
    logical, plan, emission = _split_emission()
    transition = replace(plan.transitions[0], source=plan.selected[1].inputs[0].representation)
    plan = replace(plan, transitions=(transition,))
    emission = replace(emission, plan_digest=plan.digest)
    assert any("source encoding" in issue
               for issue in verify_global_plan_emission(logical, plan, emission))


@pytest.mark.parametrize("mutation", ["unknown_capture", "missing_capture", "redefine", "absent"])
def test_emission_rejects_unprovable_ssa_or_region_capture_accounting(mutation) -> None:
    logical, plan, emission = _split_emission()
    node = emission.dispatch.nodes[1]
    if mutation == "unknown_capture":
        node.regions, node.captures = 1, None
    elif mutation == "missing_capture":
        node.regions, node.captures = 1, ["x0"]
    elif mutation == "redefine":
        node.outputs = ["x1"]
    else:
        node.outputs = ["undeclared"]
    assert verify_global_plan_emission(logical, plan, emission)
