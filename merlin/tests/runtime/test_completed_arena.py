"""Graph arena reuse must follow completion, not just dispatch issue order."""

from dataclasses import replace

import pytest

from merlin.runtime.program import build_program
from merlin.xdsl_dialects.lowering.arena_plan import ArenaPlanError, ArenaRequirements, plan_arena, plan_bounded_arena
from merlin.xdsl_dialects.lowering.dispatch_program import Buffer, DispatchProgram, Node
from merlin.xdsl_dialects.lowering.global_plan_emission import dispatch_digest


def graph():
    buffers = {
        name: Buffer(name, [4], "i8", "arg" if name == "x" else "intermediate", arg_index=0 if name == "x" else None)
        for name in ("x", "a", "c", "y", "z")
    }
    nodes = [
        Node("dispatch", "make_a", ["x"], ["a"]),
        Node("dispatch", "consume_a", ["a"], ["y"]),
        Node("dispatch", "make_c", ["x"], ["c"]),
        Node("dispatch", "consume_c", ["c"], ["z"]),
    ]
    return DispatchProgram("forward", [0], buffers, nodes, ["y", "z"])


def test_async_read_keeps_the_input_storage_live():
    program = graph()
    assert plan_arena(program).arena_bytes == 64
    request = ArenaRequirements(8, (0, 1, 2, 6), (1, 5, 3, 7), alignment=1, dispatch_digest=dispatch_digest(graph()))
    memory = plan_bounded_arena(program, request)
    assert memory.arena_bytes == 8
    assert memory.offsets == {"a": 0, "c": 4}
    with pytest.raises(ArenaPlanError, match="unsat"):
        plan_bounded_arena(program, replace(request, capacity_bytes=4))


def test_completed_values_reuse_storage_and_existing_program_format():
    request = ArenaRequirements(4, (0, 1, 2, 3), (0, 1, 2, 3), alignment=1, dispatch_digest=dispatch_digest(graph()))
    program = build_program(graph(), memory_requirements=request)
    assert program.memory_plan.offsets == {"a": 0, "c": 0}
    assert program.to_dict()["memory_plan"]["stats"]["lifetime_basis"] == "explicit launch/completion events"
    assert program.dispatch.results == ["y", "z"]
    with pytest.raises(ArenaPlanError, match="unsat"):
        build_program(graph(), memory_requirements=replace(request, persistent=("a",)))


def test_saved_placement_is_checked_without_resolving(monkeypatch):
    import merlin.runtime.program as program_module

    request = ArenaRequirements(8, (0, 1, 2, 6), (1, 5, 3, 7), alignment=1, dispatch_digest=dispatch_digest(graph()))
    memory = plan_bounded_arena(graph(), request)

    def unexpected_solver(*args):
        raise AssertionError("verification must check the saved placement without invoking allocation")

    monkeypatch.setattr(program_module, "plan_bounded_arena", unexpected_solver)
    replay = build_program(graph(), memory_requirements=request, memory_plan=memory)
    assert replay.memory_plan.to_dict() == memory.to_dict()
    with pytest.raises(ValueError, match="live buffer allocations overlap"):
        build_program(graph(), memory_requirements=request, memory_plan=replace(memory, offsets={"a": 0, "c": 0}))
    with pytest.raises(ValueError, match="statistics"):
        build_program(
            graph(), memory_requirements=request, memory_plan=replace(memory, stats={**memory.stats, "align": 4})
        )
    with pytest.raises(ValueError, match="requirements"):
        build_program(graph(), memory_plan=memory)


def test_aligned_serial_arena_uses_checked_first_fit_before_solver(monkeypatch):
    import merlin.semantic_compiler.buffer_allocation as allocation

    def unexpected_solver(*args, **kwargs):
        raise AssertionError("a checked first-fit witness needs no solver")

    monkeypatch.setattr(allocation, "allocate_buffers", unexpected_solver)
    request = ArenaRequirements(128, (0, 1, 2, 6), (1, 5, 3, 7), dispatch_digest=dispatch_digest(graph()))
    memory = plan_bounded_arena(graph(), request)
    assert memory.offsets == {"a": 0, "c": 64} and memory.arena_bytes == 68


def test_scratch_reservations_and_fp8_storage_size():
    program = graph()
    for buffer in program.buffers.values():
        buffer.dtype = "f8E4M3FN"
    request = ArenaRequirements(
        8, (0, 1, 2, 3), (0, 1, 2, 3), alignment=1, reserved=((0, 4),), dispatch_digest=dispatch_digest(program)
    )
    memory = plan_bounded_arena(program, request)
    assert memory.offsets == {"a": 4, "c": 4}
    assert memory.sizes["a"] == 4


def test_view_lifetime_retains_its_parent_allocation():
    program = graph()
    program.buffers["v"] = Buffer("v", [2], "i8", "intermediate")
    program.nodes[1] = Node("view", "tensor.extract_slice", ["a"], ["v"])
    program.nodes[3].inputs = ["v", "c"]
    program.results = ["z"]
    program.buffers.pop("y")
    request = ArenaRequirements(
        8, (0, 1, 2, 3), (0, 1, 2, 3), alignment=1, aliases=(("v", "a", 2),), dispatch_digest=dispatch_digest(program)
    )
    memory = plan_bounded_arena(program, request)
    assert memory.offsets == {"a": 0, "c": 4, "v": 2}
    with pytest.raises(ArenaPlanError, match="unsat"):
        plan_bounded_arena(program, replace(request, capacity_bytes=4))


def test_invalid_event_schedules_and_external_aliases_refuse():
    request = ArenaRequirements(8, (0, 1, 2, 3), (5, 1, 2, 3), alignment=1, dispatch_digest=dispatch_digest(graph()))
    with pytest.raises(ArenaPlanError, match="producer completes"):
        plan_bounded_arena(graph(), request)
    with pytest.raises(ArenaPlanError, match="launch and completion"):
        plan_bounded_arena(graph(), replace(request, launch_events=(0,)))
    with pytest.raises(ArenaPlanError, match="owned intermediates"):
        plan_bounded_arena(graph(), replace(request, completion_events=(0, 1, 2, 3), aliases=(("a", "x", 0),)))


def test_memory_requirements_cannot_be_reused_after_graph_changes():
    program = graph()
    request = ArenaRequirements(8, (0, 1, 2, 3), (0, 1, 2, 3), alignment=1, dispatch_digest=dispatch_digest(program))
    program.nodes[2].op = "different_implementation"
    with pytest.raises(ArenaPlanError, match="do not match"):
        plan_bounded_arena(program, request)


def test_target_padding_is_priced_and_cannot_shrink_logical_storage():
    request = ArenaRequirements(
        16,
        (0, 1, 2, 3),
        (0, 1, 2, 3),
        alignment=1,
        physical_sizes=(("a", 16), ("c", 16)),
        dispatch_digest=dispatch_digest(graph()),
    )
    memory = plan_bounded_arena(graph(), request)
    assert memory.arena_bytes == 16
    assert memory.sizes["a"] == 16
    with pytest.raises(ArenaPlanError, match="physical sizes"):
        plan_bounded_arena(graph(), replace(request, physical_sizes=(("a", 2),)))


def test_source_literal_constants_remain_outside_temporary_arena():
    program = graph()
    program.buffers["literal"] = Buffer("literal", [4], "i8", "const")
    program.nodes.insert(0, Node("view", "arith.constant", [], ["literal"]))
    program.nodes[1].inputs.append("literal")
    events = tuple(range(len(program.nodes)))
    request = ArenaRequirements(8, events, events, alignment=1, dispatch_digest=dispatch_digest(program))
    memory = plan_bounded_arena(program, request)
    assert "literal" not in memory.offsets
    assert "x" not in memory.offsets


def test_compute_output_cannot_be_declared_a_memory_view():
    program = graph()
    request = ArenaRequirements(
        8, (0, 1, 2, 3), (0, 1, 2, 3), alignment=1, aliases=(("c", "a", 0),), dispatch_digest=dispatch_digest(program)
    )
    with pytest.raises(ArenaPlanError, match="must be a view"):
        plan_bounded_arena(program, request)
