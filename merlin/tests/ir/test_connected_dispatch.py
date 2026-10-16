"""Connected dispatch planning requires graph, target, and numeric evidence."""

from __future__ import annotations

import pytest

from merlin.xdsl_dialects.lowering.connected_dispatch import (
    HandoffFacts,
    plan_connected_dispatches,
    require_realized,
)
from merlin.xdsl_dialects.lowering.dispatch_program import Buffer, DispatchProgram, Node


def _program(*, fanout: bool = False, live_out: bool = False, view: bool = False) -> DispatchProgram:
    nodes = [Node("dispatch", "first", ["input", "weight"], ["first_out"])]
    if view:
        nodes.append(Node("view", "tensor.cast", ["first_out"], ["view_out"]))
        second_input = "view_out"
    else:
        second_input = "first_out"
    nodes.append(Node("dispatch", "second", [second_input], ["second_out"]))
    if fanout:
        nodes.append(Node("dispatch", "third", ["first_out"], ["third_out"]))
    buffers = {
        name: Buffer(name, [16, 16], "bf16", kind)
        for name, kind in (
            ("input", "arg"),
            ("weight", "arg"),
            ("first_out", "intermediate"),
            ("view_out", "intermediate"),
            ("second_out", "intermediate"),
            ("third_out", "intermediate"),
        )
    }
    return DispatchProgram("forward", [0, 1], buffers, nodes, ["first_out" if live_out else "second_out"])


@pytest.mark.parametrize("interface", ["gemmini_accumulator", "mx_vpu_resident_tile"])
def test_same_planner_accepts_distinct_target_interfaces(interface: str) -> None:
    # A real target adapter must derive these facts from its selected profile.
    evidence = lambda _p, _c, _b: HandoffFacts(interface, 512, 1024, True, 40)
    plan = plan_connected_dispatches(_program(), evidence)
    assert [(h.producer, h.consumer, h.buffer, h.interface) for h in plan.handoffs] == [
        (0, 1, "first_out", interface)
    ]
    assert plan.groups == ((0, 1),)
    with pytest.raises(ValueError, match="not realized"):
        require_realized(plan, set())
    require_realized(plan, {(0, 1, "first_out")})


@pytest.mark.parametrize(
    "change", [{"resident_bytes": 2048}, {"equivalent_to_materialized": False}, {"saved_cycles": 0}]
)
def test_capacity_precision_and_profitability_each_gate_the_handoff(change: dict) -> None:
    base = dict(
        interface="direct", resident_bytes=512, capacity_bytes=1024,
        equivalent_to_materialized=True, saved_cycles=40,
    )
    plan = plan_connected_dispatches(_program(), lambda _p, _c, _b: HandoffFacts(**(base | change)))
    assert plan.handoffs == ()


@pytest.mark.parametrize("change", [{"fanout": True}, {"live_out": True}, {"view": True}])
def test_observable_or_intervening_use_blocks_a_direct_handoff(change: dict) -> None:
    plan = plan_connected_dispatches(
        _program(**change), lambda _p, _c, _b: HandoffFacts("direct", 512, 1024, True, 40)
    )
    assert plan.handoffs == ()


def test_multi_result_kernel_needs_bundle_wide_proof() -> None:
    program = _program()
    program.nodes[0].outputs.append("view_out")
    plan = plan_connected_dispatches(
        program, lambda _p, _c, _b: HandoffFacts("direct", 512, 1024, True, 40)
    )
    assert plan.handoffs == ()


def test_adjacent_handoffs_form_one_connected_region() -> None:
    program = _program()
    program.nodes.append(Node("dispatch", "third", ["second_out"], ["third_out"]))
    program.results = ["third_out"]
    plan = plan_connected_dispatches(
        program, lambda _p, _c, _b: HandoffFacts("direct", 512, 1024, True, 40)
    )
    assert [(h.producer, h.consumer) for h in plan.handoffs] == [(0, 1), (1, 2)]
    assert plan.groups == ((0, 1, 2),)


def test_invalid_or_unproved_program_cannot_select_handoffs() -> None:
    program = _program()
    program.nodes[0].outputs.append("missing")
    with pytest.raises(ValueError, match="unknown buffer"):
        plan_connected_dispatches(program, lambda _p, _c, _b: HandoffFacts("direct", 512, 1024, True, 40))


def test_calibration_can_measure_unknown_cost_without_claiming_a_speedup() -> None:
    facts = lambda _p, _c, _b: HandoffFacts("direct", 512, 1024, True, None)
    assert plan_connected_dispatches(_program(), facts).handoffs == ()
    plan = plan_connected_dispatches(_program(), facts, require_profitable=False)
    assert len(plan.handoffs) == 1 and plan.handoffs[0].saved_cycles is None


def test_compilation_refuses_backend_that_ignores_selected_residency() -> None:
    from types import SimpleNamespace

    from merlin.xdsl_dialects.lowering.connected_dispatch import lower_connected_dispatches

    class IgnoringBackend:
        def handoff_facts(self, producer, consumer, buffer):
            return dict(interface="direct", resident_bytes=512, capacity_bytes=1024,
                        equivalent_to_materialized=True, saved_cycles=40)

        def lower(self, selected):
            return SimpleNamespace(realized=set())

    with pytest.raises(ValueError, match="not realized"):
        lower_connected_dispatches(_program(), IgnoringBackend())
