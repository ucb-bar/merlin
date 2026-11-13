"""Host scheduling may abort native selection, but cannot choose its fallback."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
import z3

from merlin.semantic_compiler import search
from merlin.semantic_compiler.allocate import AllocationResult, CandidateGraph, StorageBank, Value, allocate
from merlin.semantic_compiler.egg_bridge import ENode, Exploration
from merlin.semantic_compiler.model import KernelRequest, SemanticNode, TensorType
from merlin.semantic_compiler.rules import InstructionDescriptor


def _problem(monkeypatch):
    tensor = TensorType((1,), "i8", "exact")
    request = KernelRequest(
        (SemanticNode("x", "input", (), tensor, effect="input"),
         SemanticNode("y", "step", ("x",), tensor)),
        ("y",), ("external",), (("x", "external"),), "synthetic-determinism",
    )
    descriptors = tuple(InstructionDescriptor(
        name, "step", ("external",), "external", "i8", "exact", (1,),
        input_dtypes=("i8",), input_numerical_policies=("exact",),
    ) for name in ("first", "fallback"))

    def explore(program, **kwargs):
        classes = {index: [ENode(row["symbol"], tuple(row["children"]))]
                   for index, row in enumerate(program.nodes)}
        for symbol, metadata in program.symbols.items():
            if metadata["kind"] == "instruction":
                classes[1].append(ENode(symbol, (0,)))
        return Exploration(program.roots, (0, 1),
                           {key: tuple(nodes) for key, nodes in classes.items()},
                           "saturated", 1, sum(map(len, classes.values())))

    monkeypatch.setattr(search, "explore", explore)
    return request, descriptors, (StorageBank("external", "dram", 3, "word"),)


def _select(problem):
    return search.select_and_allocate(
        *problem, bridge=Path("unused-synthetic-bridge"),
        fixed_inputs={"x": 0}, fixed_outputs=(1,),
    )


@pytest.mark.parametrize("reason", ("timeout", "canceled", "resource limits", "incomplete"))
def test_inconclusive_first_order_aborts_without_trying_fallback(monkeypatch, reason):
    problem = _problem(monkeypatch)
    calls = []

    def inconclusive(graph, order, banks, **kwargs):
        calls.append(order)
        return AllocationResult("search_timeout", order, {}, f"solver returned unknown: {reason}",
                                reason_unknown=reason)

    monkeypatch.setattr(search, "allocate", inconclusive)
    result = _select(problem)
    assert result.status == "search_timeout" and result.candidate is None
    assert len(calls) == result.ordering_attempts == result.rejected_allocation == 1
    assert len(result.allocation_attempts) == 1
    attempt = result.allocation_attempts[0]
    assert attempt.order == calls[0] and attempt.status == "search_timeout"
    assert attempt.reason_unknown == reason and attempt.candidate_digest
    assert json.loads(json.dumps(attempt.record()))["order"] == list(calls[0])


def test_time_limited_exploration_cannot_select_from_its_partial_graph(monkeypatch):
    problem = _problem(monkeypatch)
    explore = search.explore
    monkeypatch.setattr(search, "explore", lambda *args, **kwargs: replace(
        explore(*args, **kwargs), stop_reason="time_limit",
    ))

    def forbidden(*args, **kwargs):
        pytest.fail("a wall-limited graph must not reach candidate extraction")

    monkeypatch.setattr(search, "enumerate_candidates", forbidden)
    result = _select(problem)
    assert result.status == "search_timeout"
    assert result.candidate_attempts == result.ordering_attempts == 0
    assert result.allocation_attempts == ()


def test_successful_repeats_have_identical_placements_and_attempt_receipts(monkeypatch):
    problem = _problem(monkeypatch)
    results = [_select(problem) for _ in range(4)]
    assert all(result.status == "selected" for result in results)
    receipts = [json.dumps({
        "candidate": result.candidate.digest(), "order": result.allocation.order,
        "addresses": sorted(result.allocation.addresses.items()),
        "attempts": [attempt.record() for attempt in result.allocation_attempts],
    }, sort_keys=True) for result in results]
    assert len(set(receipts)) == 1
    assert results[0].allocation_attempts[-1].status == "feasible"


def test_proven_infeasibility_still_allows_candidate_fallback(monkeypatch):
    problem = _problem(monkeypatch)
    calls = []

    def first_infeasible(graph, order, banks, **kwargs):
        calls.append(order)
        if len(calls) == 1:
            return AllocationResult("infeasible_candidate", order, {}, "bounded formula is UNSAT")
        return allocate(graph, order, banks, **kwargs)

    monkeypatch.setattr(search, "allocate", first_infeasible)
    result = _select(problem)
    assert result.status == "selected" and result.candidate_attempts == 2
    assert result.rejected_allocation == 1 and len(calls) == 2
    assert [attempt.status for attempt in result.allocation_attempts] == ["infeasible_candidate", "feasible"]
    assert len({attempt.candidate_digest for attempt in result.allocation_attempts}) == 2


@pytest.mark.parametrize("bounds", ((0, 1), (0, "infinity")))
def test_sat_incumbent_without_proven_optimality_is_inconclusive(monkeypatch, bounds):
    original = z3.Optimize

    class IncumbentOptimizer:
        def __init__(self):
            self.solver = original()
            self.objectives = 0

        def __getattr__(self, name):
            return getattr(self.solver, name)

        def minimize(self, variable):
            objective = self.solver.minimize(variable)
            self.objectives += 1
            if self.objectives == 1:
                return objective

            class Bounds:
                def lower(self):
                    return z3.IntVal(bounds[0])

                def upper(self):
                    return z3.IntVal(bounds[1]) if type(bounds[1]) is int else z3.Int("oo")

            return Bounds()

        def reason_unknown(self):
            return "canceled"

        def model(self):
            pytest.fail("an unproved incumbent must never be published")

    monkeypatch.setattr(z3, "Optimize", IncumbentOptimizer)
    graph = CandidateGraph((Value(0, "first", "register", 1, (), None, "instruction"),
                            Value(1, "second", "register", 1, (), None, "instruction")), (0, 1))
    result = allocate(graph, (0, 1), (StorageBank("register", "register", 2, "word"),))
    assert result.status == "search_timeout" and result.addresses == {}
    assert result.reason_unknown == "canceled" and "optimal" in result.reason


def test_allocator_retains_raw_unknown_reason(monkeypatch):
    monkeypatch.setattr(z3.Optimize, "check", lambda self: z3.unknown)
    monkeypatch.setattr(z3.Optimize, "reason_unknown", lambda self: "timeout")
    graph = CandidateGraph((Value(0, "first", "register", 1, (), None, "instruction"),), (0,))
    result = allocate(graph, (0,), (StorageBank("register", "register", 2, "word"),))
    assert result.status == "search_timeout" and result.addresses == {}
    assert result.reason_unknown == "timeout" and "unknown" in result.reason


def test_search_limits_keep_legacy_receipts_and_wall_defaults():
    limits = search.SearchLimits()
    assert limits.solver_timeout_ms == 5000 and limits.wall_timeout_s == 60
    assert limits.record()["schema"] == "merlin.native_search_limits.v1"
    assert search.SearchLimits.from_record(json.loads(json.dumps(limits.record()))) == limits
