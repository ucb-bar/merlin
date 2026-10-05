"""Merlin-native exploration, extraction, order and placement feedback loop.

This is a selection/allocation stage. It cannot report an executable compiler
result until target-specific scheduling, emission and execution have passed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from time import monotonic
from typing import Any

from .allocate import (
    AllocationResult,
    CandidateGraph,
    StorageBank,
    allocate,
    interference_edges,
    lower_candidate,
    may_prune_interference,
    topological_orders,
)
from .egg_bridge import EGraphTimeout, EGraphUnavailable, Exploration, explore
from .extract import Candidate, ExtractionTimeout, enumerate_candidates
from .model import KernelRequest
from .rules import InstructionDescriptor, RuleProgram, generate_rules
from .verify import check_selection


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
        if any(type(value) is not int or value <= 0 for value in vars(self).values()):
            raise ValueError("all search limits must be positive integers")

    def record(self) -> dict[str, int | str]:
        return {"schema": "merlin.native_search_limits.v1", **vars(self)}

    @classmethod
    def from_record(cls, row: dict[str, Any]) -> SearchLimits:
        if (
            not isinstance(row, dict) or row.get("schema") != "merlin.native_search_limits.v1"
            or set(row) != {"schema", *vars(cls())}
        ):
            raise ValueError("search limits need the v1 schema and every declared field")
        return cls(**{name: row[name] for name in vars(cls())})


@dataclass(frozen=True)
class SearchAblations:
    """Diagnostic-only removals of native search mechanisms.

    Safety checks remain enabled. The current admitted generic rewrite set
    contains a structural identity rule, but no arithmetic reassociation.
    """

    disable_structural_rewrites: bool = False
    one_shot_extraction: bool = False
    disable_alternative_orders: bool = False
    disable_candidate_fallback: bool = False

    def active(self) -> tuple[str, ...]:
        if any(type(value) is not bool for value in vars(self).values()):
            raise ValueError("native search ablations must be booleans")
        return tuple(name for name, enabled in vars(self).items() if enabled)


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
    pruned_orders: int = 0
    check_fingerprint: str = ""
    diagnostic_ablations: tuple[str, ...] = ()

    @property
    def engine(self) -> str:
        return "merlin_native"

    @property
    def diagnostic_only(self) -> bool:
        return bool(self.diagnostic_ablations)


def select_and_allocate(
    request: KernelRequest,
    descriptors: tuple[InstructionDescriptor, ...],
    banks: tuple[StorageBank, ...],
    *,
    bridge: Path,
    fixed_inputs: dict[str, int] | None = None,
    fixed_outputs: tuple[int | None, ...] | None = None,
    limits: SearchLimits = SearchLimits(),
    ablations: SearchAblations = SearchAblations(),
) -> SearchResult:
    active_ablations = ablations.active()

    def finish(result: SearchResult) -> SearchResult:
        return replace(result, diagnostic_ablations=active_ablations)

    deadline = monotonic() + limits.wall_timeout_s
    program = generate_rules(request, descriptors)
    if ablations.disable_structural_rewrites:
        program = replace(
            program,
            rewrites=tuple(rule for rule in program.rewrites if rule.descriptor_name != "<structural>"),
        )
    identity = request.digest()
    remaining = deadline - monotonic()
    if remaining <= 0:
        return finish(SearchResult(
            "search_timeout",
            identity,
            None,
            None,
            None,
            0,
            0,
            0,
            None,
            program,
            "native search deadline expired during rule generation",
        ))
    try:
        graph = explore(
            program,
            bridge=bridge,
            iterations=limits.iterations,
            node_limit=limits.egraph_nodes,
            wall_timeout_s=remaining,
        )
    except EGraphTimeout as exc:
        return finish(SearchResult("search_timeout", identity, None, None, None, 0, 0, 0, None, program, str(exc)))
    except EGraphUnavailable as exc:
        return finish(SearchResult("tool_unavailable", identity, None, None, None, 0, 0, 0, None, program, str(exc)))
    candidate_attempts = 0
    ordering_attempts = 0
    rejected_allocation = 0
    pruned_orders = 0
    seen: set[str] = set()
    inconclusive = False
    unqualified = False
    unqualified_reason = ""

    def timed_out(reason: str) -> SearchResult:
        return finish(SearchResult(
            "search_timeout",
            identity,
            None,
            None,
            None,
            candidate_attempts,
            ordering_attempts,
            rejected_allocation,
            graph,
            program,
            reason,
            pruned_orders,
        ))

    if monotonic() >= deadline:
        return timed_out("native search deadline expired during e-graph exploration")
    for budget in range(1, limits.candidate_nodes + 1):
        iterator = iter(
            enumerate_candidates(
                graph,
                request,
                program,
                node_budget=budget,
                max_candidates=1 if ablations.one_shot_extraction else limits.candidates,
                deadline=deadline,
            )
        )
        while True:
            try:
                candidate = next(iterator)
            except StopIteration:
                break
            except ExtractionTimeout as exc:
                return timed_out(str(exc))
            if monotonic() >= deadline:
                return timed_out("native search deadline expired during candidate extraction")
            digest = candidate.digest()
            if digest in seen:
                continue
            seen.add(digest)
            candidate_attempts += 1
            candidate_graph = lower_candidate(candidate, program)
            # Each candidate owns one fixed base formula: domains, validity,
            # def-use, fixed I/O and geometry. Orders change only its
            # interference edge set in this finite allocation model.
            candidate_base = candidate.digest()
            failed_edges: list[frozenset[tuple[int, int]]] = []
            for order in topological_orders(
                candidate_graph,
                limit=1 if ablations.disable_alternative_orders else limits.orders_per_candidate,
            ):
                remaining = deadline - monotonic()
                if remaining <= 0:
                    return timed_out("native search deadline expired during ordering")
                ordering_attempts += 1
                edges = interference_edges(candidate_graph, order, banks)
                if any(
                    may_prune_interference(
                        candidate_base, candidate_base, prior, edges, failed_status="infeasible_candidate"
                    )
                    for prior in failed_edges
                ):
                    pruned_orders += 1
                    continue
                result = allocate(
                    candidate_graph,
                    order,
                    banks,
                    fixed_inputs=fixed_inputs,
                    fixed_outputs=fixed_outputs,
                    timeout_ms=min(limits.solver_timeout_ms, max(1, int(remaining * 1000))),
                )
                if monotonic() >= deadline:
                    return timed_out("native search deadline expired during allocation")
                if result.status == "feasible":
                    checked = check_selection(
                        request,
                        descriptors,
                        program,
                        graph,
                        candidate,
                        candidate_graph,
                        result,
                        banks,
                        fixed_inputs=fixed_inputs,
                        fixed_outputs=fixed_outputs,
                    )
                    if not checked.valid:
                        return finish(SearchResult(
                            "modeling_failure",
                            identity,
                            candidate,
                            candidate_graph,
                            result,
                            candidate_attempts,
                            ordering_attempts,
                            rejected_allocation,
                            graph,
                            program,
                            checked.reason,
                            pruned_orders,
                        ))
                    if monotonic() >= deadline:
                        return timed_out("native search deadline expired during final selection replay")
                    return finish(SearchResult(
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
                        pruned_orders=pruned_orders,
                        check_fingerprint=checked.fingerprint,
                    ))
                rejected_allocation += 1
                if result.status == "infeasible_candidate":
                    failed_edges.append(edges)
                if result.status == "search_timeout":
                    inconclusive = True
                if result.status == "unqualified_target":
                    unqualified = True
                    unqualified_reason = unqualified_reason or result.reason
                if result.status in {"tool_unavailable", "modeling_failure"}:
                    return finish(SearchResult(
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
                        pruned_orders,
                    ))
            if candidate_attempts >= limits.candidates or (
                candidate_attempts and (ablations.one_shot_extraction or ablations.disable_candidate_fallback)
            ):
                break
        if candidate_attempts >= limits.candidates or (
            candidate_attempts and (ablations.one_shot_extraction or ablations.disable_candidate_fallback)
        ):
            break
    if unqualified:
        status = "unqualified_target"
    elif inconclusive or graph.stop_reason == "time_limit":
        status = "search_timeout"
    elif candidate_attempts >= limits.candidates or graph.stop_reason != "saturated":
        status = "resource_limit"
    elif active_ablations and candidate_attempts:
        status = "resource_limit"
    elif not any(info["kind"] == "instruction" for info in program.symbols.values()):
        status = "unsupported_semantics"
    elif candidate_attempts == 0:
        # A bounded extractor cannot establish that a larger instruction
        # budget has no candidate, even when the current e-graph saturated.
        status = "resource_limit"
    else:
        status = "compile_error"
    return finish(SearchResult(
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
        unqualified_reason if status == "unqualified_target" else "bounded native search found no checked placement",
        pruned_orders,
    ))
