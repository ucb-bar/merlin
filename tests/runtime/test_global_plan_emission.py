from __future__ import annotations

import pytest

from merlin.compiler.dialects.lowering.dispatch_program import Buffer, DispatchProgram, Node
from merlin.compiler.dialects.lowering.global_plan import (
    BufferRepresentation,
    CycleInterval,
    GlobalPlan,
    RegionAlternative,
    ValueRepresentation,
)
from merlin.compiler.dialects.lowering.global_plan_emission import (
    BoundaryMapping,
    EmittedComponent,
    GlobalPlanEmission,
    dispatch_digest,
)
from merlin.runtime.program import build_program


def _logical() -> DispatchProgram:
    return DispatchProgram(
        "forward",
        [0],
        {
            "b0": Buffer("b0", [4], "i8", "arg", 0),
            "b1": Buffer("b1", [4], "i8", "intermediate"),
            "b2": Buffer("b2", [4], "i8", "intermediate"),
        },
        [
            Node("dispatch", "producer", ["b0"], ["b1"], captures=[]),
            Node("dispatch", "consumer", ["b1"], ["b2"], captures=[]),
        ],
        ["b2"],
    )


def _plan(program: DispatchProgram) -> GlobalPlan:
    rep = ValueRepresentation("memory", "linear", "i8")
    fused = RegionAlternative(
        "fused",
        (0, 1),
        "fused_kernel",
        "engine",
        CycleInterval.point(10, "fixture"),
        inputs=(BufferRepresentation("b0", rep),),
        outputs=(BufferRepresentation("b2", rep),),
    )
    return GlobalPlan((fused,), (), CycleInterval.point(10, "fixture"))


class _Emitter:
    def __init__(self, *, hidden_node: bool = False):
        self.hidden_node = hidden_node

    def emit_global_plan(self, program: DispatchProgram, plan: GlobalPlan) -> GlobalPlanEmission:
        nodes = [Node("dispatch", "fused_kernel", ["x0"], ["x1"], captures=[])]
        if self.hidden_node:
            nodes.append(Node("view", "hidden_copy", ["x1"], ["x2"], captures=[]))
        emitted = DispatchProgram(
            "forward",
            [0],
            {
                "x0": Buffer("x0", [4], "i8", "arg", 0),
                "x1": Buffer("x1", [4], "i8", "intermediate"),
                **({"x2": Buffer("x2", [4], "i8", "intermediate")} if self.hidden_node else {}),
            },
            nodes,
            ["x1"],
        )
        return GlobalPlanEmission(
            emitted,
            plan.digest,
            dispatch_digest(program),
            (EmittedComponent("fused", (0,), (("b0", "x0"),), (("b2", "x1"),)),),
            (),
            (BoundaryMapping("input", "b0", "x0"), BoundaryMapping("output", "b2", "x1")),
            ("fixture target emitter",),
        )


def test_plan_without_emitter_is_explicit_shadow_analysis() -> None:
    logical = _logical()
    program = build_program(logical, global_plan=_plan(logical))
    assert [node.op for node in program.dispatch.nodes] == ["producer", "consumer"]
    assert program.stats["global_plan_emitted"] is False


def test_emitter_cannot_hide_unaccounted_executable_work() -> None:
    logical = _logical()
    with pytest.raises(ValueError, match="unaccounted"):
        build_program(logical, global_plan=_plan(logical), global_plan_emitter=_Emitter(hidden_node=True))
