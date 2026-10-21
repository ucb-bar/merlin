"""Merlin-native exploration, extraction, order and placement feedback loop.

This is a selection/allocation stage. It cannot report an executable compiler
result until target-specific scheduling, emission and execution have passed.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from time import monotonic
from typing import Any

from .allocate import (
    AllocationResult,
    CandidateGraph,
    Reservation,
    StorageBank,
    allocate,
    interference_edges,
    lower_candidate,
    may_prune_interference,
    topological_orders,
)
from .egg_bridge import EGraphTimeout, EGraphUnavailable, Exploration, explore
from .extract import Candidate, ExtractionSession, ExtractionTimeout, enumerate_candidates
from .model import KernelRequest
from .rules import ConcatViewDescriptor, InstructionDescriptor, RuleProgram, ViewDescriptor, generate_rules
from .verify import check_selection


@dataclass(frozen=True)
class SearchLimits:
    iterations: int = 8
    egraph_nodes: int = 5000
    candidate_nodes: int = 24
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
            not isinstance(row, dict)
            or row.get("schema") != "merlin.native_search_limits.v1"
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
class MaterializationDecision:
    """Target emission may return a different, independently replayed placement."""

    accepted: bool
    reason: str = ""
    realized_allocation: AllocationResult | None = None


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
    exploration_attempts: int = 0
    rejected_materialization: int = 0

    @property
    def engine(self) -> str:
        return "merlin_native"

    @property
    def diagnostic_only(self) -> bool:
        return bool(self.diagnostic_ablations)


def _work_stages(source_nodes: int, limits: SearchLimits) -> Iterator[tuple[int, int, int]]:
    """Grow exploration and extraction budgets together, ending at the declared limits."""
    iterations = 1
    egraph_nodes = min(limits.egraph_nodes, max(64, 4 * source_nodes))
    candidate_nodes = 1
    while True:
        yield iterations, egraph_nodes, candidate_nodes
        if (
            iterations == limits.iterations
            and egraph_nodes == limits.egraph_nodes
            and candidate_nodes == limits.candidate_nodes
        ):
            return
        iterations = min(limits.iterations, 2 * iterations)
        egraph_nodes = min(limits.egraph_nodes, 2 * egraph_nodes)
        candidate_nodes = min(limits.candidate_nodes, 2 * candidate_nodes)


def select_and_allocate(
    request: KernelRequest,
    descriptors: tuple[InstructionDescriptor, ...],
    banks: tuple[StorageBank, ...],
    *,
    bridge: Path,
    views: tuple[ViewDescriptor | ConcatViewDescriptor, ...] = (),
    fixed_inputs: dict[str, int] | None = None,
    reservations: tuple[Reservation, ...] = (),
    fixed_outputs: tuple[int | None, ...] | None = None,
    limits: SearchLimits = SearchLimits(),
    ablations: SearchAblations = SearchAblations(),
    accept_materialization: Callable[[SearchResult], tuple[bool, str] | MaterializationDecision] | None = None,
) -> SearchResult:
    active_ablations = ablations.active()
    exploration_attempts = 0
    rejected_materialization = 0

    def finish(result: SearchResult) -> SearchResult:
        return replace(
            result,
            diagnostic_ablations=active_ablations,
            exploration_attempts=exploration_attempts,
            rejected_materialization=rejected_materialization,
        )

    unresolved = next((node for node in request.nodes if node.op.startswith("source_")), None)
    if unresolved is not None:
        if unresolved.op == "source_f32_matmul_unqualified":
            reason = f"source f32 matmul {unresolved.id} needs an explicit qualified numerical transform"
        elif unresolved.op == "source_f32_reduce_add_unqualified":
            reason = (
                f"source f32 reduction {unresolved.id} needs a qualified numerical-order policy "
                "or compiled host route"
            )
        else:
            reason = f"source operation {unresolved.id} needs a qualified numerical transform or compiled host route"
        return finish(SearchResult(
            "unsupported_semantics", request.digest(), None, None, None,
            0, 0, 0, None, None, reason,
        ))

    deadline = monotonic() + limits.wall_timeout_s
    program = generate_rules(request, descriptors, views)
    if ablations.disable_structural_rewrites:
        program = replace(
            program,
            rewrites=tuple(rule for rule in program.rewrites if rule.descriptor_name != "<structural>"),
        )
    identity = request.digest()
    graph: Exploration | None = None
    extraction_session: ExtractionSession | None = None
    explored_budget: tuple[int, int] | None = None
    extracted_budget = 0
    candidate_attempts = 0
    ordering_attempts = 0
    rejected_allocation = 0
    pruned_orders = 0
    seen: set[str] = set()
    inconclusive = False
    unqualified = False
    unqualified_reason = ""
    materialization_reason = ""
    order_budget_exhausted = False

    def timed_out(reason: str) -> SearchResult:
        return finish(
            SearchResult(
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
            )
        )

    for iterations, egraph_nodes, candidate_limit in _work_stages(len(request.nodes), limits):
        remaining = deadline - monotonic()
        if remaining <= 0:
            return timed_out("native search deadline expired during exploration")
        if graph is None or (graph.stop_reason != "saturated" and explored_budget != (iterations, egraph_nodes)):
            exploration_attempts += 1
            try:
                graph = explore(
                    program,
                    bridge=bridge,
                    iterations=iterations,
                    node_limit=egraph_nodes,
                    wall_timeout_s=remaining,
                )
            except EGraphTimeout as exc:
                return timed_out(str(exc))
            except EGraphUnavailable as exc:
                return finish(
                    SearchResult(
                        "tool_unavailable",
                        identity,
                        None,
                        None,
                        None,
                        candidate_attempts,
                        ordering_attempts,
                        rejected_allocation,
                        graph,
                        program,
                        str(exc),
                        pruned_orders,
                    )
                )
            explored_budget = (iterations, egraph_nodes)
            extracted_budget = 0
            try:
                extraction_session = ExtractionSession(graph, request, program, deadline)
            except ExtractionTimeout as exc:
                return timed_out(str(exc))
        assert graph is not None
        # Phase 1 chooses the first checked candidate under increasing work
        # bounds; it does not minimize instruction count. Trying every integer
        # between geometric stages repeats expensive incomplete extractions.
        for budget in ((candidate_limit,) if candidate_limit > extracted_budget else ()):
            iterator = iter(
                enumerate_candidates(
                    graph,
                    request,
                    program,
                    node_budget=budget,
                    max_candidates=1 if ablations.one_shot_extraction else limits.candidates,
                    deadline=deadline,
                    session=extraction_session,
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
                order_limit = 1 if ablations.disable_alternative_orders else limits.orders_per_candidate
                orders_seen = 0
                for order in topological_orders(
                    candidate_graph,
                    limit=order_limit + 1,
                ):
                    orders_seen += 1
                    if orders_seen > order_limit:
                        order_budget_exhausted = True
                        break
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
                        reservations=reservations,
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
                            views=views,
                            fixed_inputs=fixed_inputs,
                            reservations=reservations,
                            fixed_outputs=fixed_outputs,
                        )
                        if not checked.valid:
                            return finish(
                                SearchResult(
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
                                )
                            )
                        if monotonic() >= deadline:
                            return timed_out("native search deadline expired during final selection replay")
                        selected = finish(
                            SearchResult(
                                "selected", identity, candidate, candidate_graph, result,
                                candidate_attempts, ordering_attempts, rejected_allocation,
                                graph, program, pruned_orders=pruned_orders,
                                check_fingerprint=checked.fingerprint,
                            )
                        )
                        if accept_materialization is not None:
                            decision = accept_materialization(selected)
                            if isinstance(decision, tuple) and len(decision) == 2:
                                decision = MaterializationDecision(*decision)
                            if (
                                not isinstance(decision, MaterializationDecision)
                                or type(decision.accepted) is not bool
                                or type(decision.reason) is not str
                                or (not decision.accepted and not decision.reason)
                                or (not decision.accepted and decision.realized_allocation is not None)
                            ):
                                raise ValueError("materialization decision needs a boolean and a rejection reason")
                            if monotonic() >= deadline:
                                return timed_out("native search deadline expired during target materialization")
                            replay = check_selection(
                                request, descriptors, program, graph, candidate,
                                candidate_graph, result, banks, views=views, fixed_inputs=fixed_inputs,
                                reservations=reservations, fixed_outputs=fixed_outputs,
                            )
                            if not replay.valid or replay.fingerprint != checked.fingerprint:
                                return finish(replace(
                                    selected, status="modeling_failure",
                                    reason="selected state changed during target materialization: "
                                    + (replay.reason or "check fingerprint differs"),
                                ))
                            if not decision.accepted:
                                rejected_materialization += 1
                                materialization_reason = materialization_reason or decision.reason
                                # This is not an allocation UNSAT witness. Another order or
                                # instruction graph may still lower on the selected target.
                                continue
                            if decision.realized_allocation is not None:
                                realized = decision.realized_allocation
                                if realized.order != result.order or realized.issue_times != result.issue_times:
                                    return finish(replace(
                                        selected, status="modeling_failure",
                                        reason="target materialization changed the selected instruction "
                                        "order or schedule",
                                    ))
                                realized_check = check_selection(
                                    request, descriptors, program, graph, candidate,
                                    candidate_graph, realized, banks, views=views,
                                    fixed_inputs=fixed_inputs, reservations=reservations,
                                    fixed_outputs=fixed_outputs,
                                )
                                if not realized_check.valid:
                                    return finish(replace(
                                        selected, status="modeling_failure",
                                        reason="target materialization has an invalid physical assignment: "
                                        + realized_check.reason,
                                    ))
                                selected = replace(
                                    selected, allocation=realized,
                                    check_fingerprint=realized_check.fingerprint,
                                )
                        return finish(selected)
                    rejected_allocation += 1
                    if result.status == "infeasible_candidate":
                        failed_edges.append(edges)
                    if result.status == "search_timeout":
                        inconclusive = True
                    if result.status == "unqualified_target":
                        unqualified = True
                        unqualified_reason = unqualified_reason or result.reason
                    if result.status in {"tool_unavailable", "modeling_failure"}:
                        return finish(
                            SearchResult(
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
                            )
                        )
                if candidate_attempts >= limits.candidates or (
                    candidate_attempts and (ablations.one_shot_extraction or ablations.disable_candidate_fallback)
                ):
                    break
            if candidate_attempts >= limits.candidates or (
                candidate_attempts and (ablations.one_shot_extraction or ablations.disable_candidate_fallback)
            ):
                break
        extracted_budget = candidate_limit
        if candidate_attempts >= limits.candidates or (
            candidate_attempts and (ablations.one_shot_extraction or ablations.disable_candidate_fallback)
        ):
            break
    assert graph is not None
    if inconclusive or graph.stop_reason == "time_limit":
        status = "search_timeout"
    elif candidate_attempts >= limits.candidates or graph.stop_reason != "saturated" or order_budget_exhausted:
        status = "resource_limit"
    elif active_ablations and candidate_attempts:
        status = "resource_limit"
    elif unqualified or rejected_materialization:
        status = "unqualified_target"
    elif candidate_attempts == 0:
        if not any(info["kind"] == "instruction" for info in program.symbols.values()):
            status = "unsupported_semantics"
        else:
            # A bounded extractor cannot establish that a larger instruction
            # budget has no candidate, even when the current e-graph saturated.
            status = "resource_limit"
    else:
        status = "compile_error"
    return finish(
        SearchResult(
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
            (unqualified_reason or materialization_reason)
            if status == "unqualified_target"
            else "bounded native search found no checked placement",
            pruned_orders,
        )
    )
