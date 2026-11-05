"""Necessary capacity checks must respect views and permitted in-place writes."""

from merlin.semantic_compiler.allocate import CandidateGraph, StorageBank, Value, allocate
from merlin.semantic_compiler.rules import AddressConstraint


def test_overfull_live_clique_is_infeasible_without_solver_timeout() -> None:
    count = 33
    graph = CandidateGraph(tuple(
        Value(i, f"produce_{i}", "register", 1, (), None, "instruction")
        for i in range(count)
    ), tuple(range(count)))
    result = allocate(
        graph, tuple(range(count)), (StorageBank("register", "register_file", 32, "word"),),
        timeout_ms=200,
    )
    assert result.status == "infeasible_candidate", result.reason
    assert not result.addresses


def test_overlapping_views_do_not_add_storage_requirements() -> None:
    graph = CandidateGraph((
        Value(0, "input", "register", 4, (), "x", "input"),
        Value(1, "low", "register", 2, (0,), None, "view",
              validity=(AddressConstraint("eq_offset", "out", "in0", 0),)),
        Value(2, "high", "register", 2, (0,), None, "view",
              validity=(AddressConstraint("eq_offset", "out", "in0", 2),)),
    ), (1, 2))
    result = allocate(
        graph, (), (StorageBank("register", "register_file", 4, "word"),),
        fixed_inputs={"x": 0}, fixed_outputs=(0, 2),
    )
    assert result.status == "feasible", result.reason


def test_in_place_read_before_write_does_not_require_two_registers() -> None:
    graph = CandidateGraph((
        Value(0, "produce", "register", 1, (), None, "instruction", completion_offset=1),
        Value(1, "update", "register", 1, (0,), None, "instruction",
              input_read_offsets=(0,), completion_offset=1, in_place_inputs=(0,)),
    ), (1,))
    result = allocate(graph, (0, 1), (StorageBank("register", "register_file", 1, "word"),))
    assert result.status == "feasible", result.reason
