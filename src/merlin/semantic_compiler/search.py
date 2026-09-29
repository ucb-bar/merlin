"""Merlin-native exploration, extraction, order and placement feedback loop.

This is a selection/allocation stage. It cannot report an executable compiler
result until target-specific scheduling, emission and execution have passed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .allocate import AllocationResult, CandidateGraph, Reservation, StorageBank, allocate, lower_candidate, topological_orders
from .egg_bridge import EGraphUnavailable, Exploration, explore
from .extract import Candidate, enumerate_candidates
from .model import KernelRequest
from .rules import InstructionDescriptor, RuleProgram, generate_rules


@dataclass(frozen=True)
class SearchLimits:
    iterations: int = 8
    egraph_nodes: int = 5000
    candidate_nodes: int = 12
    candidates: int = 256
    orders_per_candidate: int = 32
    solver_timeout_ms: int = 5000
    wall_timeout_s: int = 60

    def __post_init__(self) -> None:
        if min(vars(self).values()) <= 0:
            raise ValueError("all search limits must be positive")


@dataclass(frozen=True)
class SearchResult:
    status: str
    request_digest: str
    candidate: Candidate | None
    graph: CandidateGraph | None
    allocation: AllocationResult | None
    candidate_attempts: int
    ordering_attempts: int
    rejected_allocation: int
    exploration: Exploration | None
    rules: RuleProgram | None
    reason: str = ""


def select_and_allocate(
    request: KernelRequest,
    descriptors: tuple[InstructionDescriptor, ...],
    banks: tuple[StorageBank, ...],
    *,
    bridge: Path,
    fixed_inputs: dict[str, int] | None = None,
    reservations: tuple[Reservation, ...] = (),
    limits: SearchLimits = SearchLimits(),
) -> SearchResult:
    program = generate_rules(request, descriptors)
    identity = request.digest()
    try:
        graph = explore(
            program,
            bridge=bridge,
            iterations=limits.iterations,
            node_limit=limits.egraph_nodes,
            wall_timeout_s=limits.wall_timeout_s,
        )
    except EGraphUnavailable as exc:
        return SearchResult("tool_unavailable", identity, None, None, None, 0, 0, 0, None, program, str(exc))
    candidate_attempts = 0
    ordering_attempts = 0
    rejected_allocation = 0
    seen: set[str] = set()
    inconclusive = False
    for budget in range(1, limits.candidate_nodes + 1):
        for candidate in enumerate_candidates(
            graph,
            request,
            program,
            node_budget=budget,
            max_candidates=limits.candidates,
        ):
            digest = candidate.digest()
            if digest in seen:
                continue
            seen.add(digest)
            candidate_attempts += 1
            candidate_graph = lower_candidate(candidate, program)
            for order in topological_orders(candidate_graph, limit=limits.orders_per_candidate):
                ordering_attempts += 1
                result = allocate(
                    candidate_graph,
                    order,
                    banks,
                    fixed_inputs=fixed_inputs,
                    reservations=reservations,
                    timeout_ms=limits.solver_timeout_ms,
                )
                if result.status == "feasible":
                    return SearchResult(
                        "selected",
                        identity,
                        candidate,
                        candidate_graph,
                        result,
                        candidate_attempts,
                        ordering_attempts,
                        rejected_allocation,
                        graph,
                        program,
                    )
                rejected_allocation += 1
                if result.status == "search_timeout":
                    inconclusive = True
                if result.status in {"tool_unavailable", "modeling_failure", "unqualified_target"}:
                    return SearchResult(
                        result.status,
                        identity,
                        None,
                        None,
                        None,
                        candidate_attempts,
                        ordering_attempts,
                        rejected_allocation,
                        graph,
                        program,
                        result.reason,
                    )
            if candidate_attempts >= limits.candidates:
                break
        if candidate_attempts >= limits.candidates:
            break
    if inconclusive:
        status = "search_timeout"
    elif candidate_attempts >= limits.candidates or graph.stop_reason != "saturated":
        status = "resource_limit"
    elif not any(info["kind"] == "instruction" for info in program.symbols.values()):
        status = "unsupported_semantics"
    else:
        status = "compile_error"
    return SearchResult(
        status,
        identity,
        None,
        None,
        None,
        candidate_attempts,
        ordering_attempts,
        rejected_allocation,
        graph,
        program,
        "bounded native search found no checked placement",
    )
