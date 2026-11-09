"""Preserve current allocation events and bank origins while adding views."""

import pytest

from merlin.semantic_compiler.allocate import allocate, check_assignment, instruction_schedule, topological_orders
from merlin.semantic_compiler.allocation_types import CandidateGraph, Reservation, StorageBank, Value


def test_nonzero_bank_origin_is_enforced_by_solver_and_independent_checker():
    graph = CandidateGraph((Value(0, "input", "memory", 1, (), "x", "input"),), (0,))
    banks = (StorageBank("memory", "physical", 8, "byte", start=4),)
    result = allocate(graph, (), banks)
    assert result.status == "feasible"
    assert 4 <= result.addresses[0] < 8
    assert check_assignment(graph, (), {0: 3}, banks)[0] is False
    assert check_assignment(graph, (), {0: 4}, banks, fixed_inputs={"x": 3})[0] is False
    assert check_assignment(graph, (), {0: 4}, banks, reservations=(Reservation("memory", 3, 1),))[0] is False
    assert StorageBank.from_record(banks[0].record()) == banks[0]


def test_result_free_event_keeps_dependency_and_occupies_no_reserved_storage():
    graph = CandidateGraph((
        Value(0, "input", "memory", 1, (), "x", "input"),
        Value(1, "command", "memory", 0, (0,), None, "event", completion_offset=2),
        Value(2, "command", "memory", 0, (1,), None, "event", completion_offset=1),
    ), (0,))
    banks = (StorageBank("memory", "physical", 8, "byte", start=4),)
    assert list(topological_orders(graph, limit=4)) == [(1, 2)]
    assert instruction_schedule(graph, (1, 2))[2][0] > instruction_schedule(graph, (1, 2))[1][1]
    with pytest.raises(ValueError, match="consumer before"):
        instruction_schedule(graph, (2, 1))
    result = allocate(graph, (1, 2), banks, fixed_inputs={"x": 4},
                      reservations=(Reservation("memory", 5, 3),))
    assert result.status == "feasible"
    assert check_assignment(graph, (1, 2), result.addresses, banks, fixed_inputs={"x": 4},
                            reservations=(Reservation("memory", 5, 3),))[0]
    with pytest.raises(ValueError, match="consumes no storage"):
        Value(3, "command", "memory", 1, (), None, "event")
