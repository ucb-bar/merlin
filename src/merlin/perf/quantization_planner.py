"""Global, target-neutral planning for quantized epilogues and residual regions.

The planner consumes exact source contracts, target-supplied capabilities, host-authorized
accuracy policy, and a complete ordered program graph.  It selects no approximate transformation
without policy/corpus evidence, never converts structural byte deletion into a cycle claim, and
leaves target emission as an explicit downstream obligation.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from math import ceil, isfinite
from typing import Any

from merlin.perf.quantization_contract import (
    AccuracyPolicy,
    EpilogueAdmission,
    ExactEpilogueProof,
    QuantizedEpilogueCandidate,
    QuantizedEpilogueCapability,
    QuantizedEpilogueContract,
    admit_epilogue,
    canonical_sha256,
)


PLAN_SCHEMA = "target_neutral_quantized_region_plan_v1"
RESIDUAL_SCHEMA = "target_neutral_global_quantization_domains_v2"
REGION_SCHEMA = "target_neutral_maximal_accelerator_regions_v1"

# Semantic roles, not implementation paths.  The agent-guidance layer joins these to an immutable,
# host-verified compiler inventory before returning any editable coordinate.
EDIT_REQUIREMENT_ROLES = frozenset({
    "source_epilogue_semantics",
    "global_quant_domain_planner",
    "target_epilogue_emitter",
    "target_residual_region_emitter",
    "target_encoding_and_residency",
})

__all__ = [
    "EDIT_REQUIREMENT_ROLES",
    "PLAN_SCHEMA",
    "REGION_SCHEMA",
    "RESIDUAL_SCHEMA",
    "EpilogueSite",
    "QuantizationDomain",
    "QuantizedProgramEdge",
    "QuantizedProgramNode",
    "QuantizedRegionProblem",
    "ResidualDomainProblem",
    "ResidualDomainSite",
    "plan_quantized_regions",
    "plan_residual_domains",
]


@dataclass(frozen=True)
class QuantizationDomain:
    """One affine integer domain: ``real = scale * (code - zero_point)``."""

    name: str
    scale: float | None
    zero_point: int | None
    qmin: int
    qmax: int
    provenance_sha256: str

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("quantization domain requires a name")
        if self.scale is not None and (not isfinite(self.scale) or self.scale <= 0):
            raise ValueError("quantization scale must be positive finite or unknown")
        if self.qmin >= self.qmax:
            raise ValueError("quantization domain requires qmin < qmax")
        if len(self.provenance_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.provenance_sha256.lower()
        ):
            raise ValueError("quantization domain requires provenance SHA-256")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "scale": self.scale,
            "zero_point": self.zero_point,
            "qmin": self.qmin,
            "qmax": self.qmax,
            "provenance_sha256": self.provenance_sha256,
        }


@dataclass(frozen=True)
class ResidualDomainSite:
    """One exactly binary residual sink and an optional prior residual-history edge."""

    index: int
    site_id: str
    source_operation_ids: tuple[int, ...]
    host_operation_ids: tuple[int, ...]
    element_count: int
    output_domain: str
    predecessor: int | None
    branch_materialization_bytes: int
    carry_materialization_bytes: int | None
    required_capabilities: tuple[str, ...]
    resident_working_set_bytes: int | None
    residual_input_count: int = 2
    alignment_blockers: tuple[str, ...] = ()
    carry_blockers: tuple[str, ...] = ()
    saturation_scope: str = "unproven"
    saturation_corpus_id: str | None = None
    external_input_error_lsb: int = 0
    external_error_provenance_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.index < 0 or not self.site_id.strip() or self.element_count < 1:
            raise ValueError("residual site requires nonnegative index, id, and positive extent")
        if not self.source_operation_ids or any(index < 0 for index in self.source_operation_ids):
            raise ValueError("residual site requires nonnegative source operations")
        if any(index < 0 for index in self.host_operation_ids):
            raise ValueError("residual host operation ids must be nonnegative")
        if self.residual_input_count != 2:
            raise ValueError("global residual alignment currently requires an exactly binary residual")
        if self.branch_materialization_bytes < 0:
            raise ValueError("branch materialization bytes must be an exact nonnegative count")
        if self.predecessor is None and self.carry_materialization_bytes is not None:
            raise ValueError("a root residual cannot carry a predecessor materialization")
        if self.predecessor is not None and (
            self.carry_materialization_bytes is None or self.carry_materialization_bytes < 0
        ):
            raise ValueError("a residual predecessor edge requires exact carry bytes")
        if self.resident_working_set_bytes is not None and self.resident_working_set_bytes <= 0:
            raise ValueError("resident working set must be positive when supplied")
        if self.saturation_scope not in {
            "unproven", "universal_no_saturation", "heldout_no_saturation"
        }:
            raise ValueError("unknown source-quantizer saturation scope")
        if (self.saturation_scope == "heldout_no_saturation") != (
            self.saturation_corpus_id is not None
        ):
            raise ValueError("heldout saturation scope and corpus id disagree")
        if self.external_input_error_lsb < 0:
            raise ValueError("external input error bound must be nonnegative")
        has_provenance = (
            isinstance(self.external_error_provenance_sha256, str)
            and len(self.external_error_provenance_sha256) == 64
            and all(character in "0123456789abcdef"
                    for character in self.external_error_provenance_sha256.lower())
        )
        if (self.external_input_error_lsb > 0) != has_provenance:
            raise ValueError("nonzero external input error requires exact provenance SHA-256")


@dataclass(frozen=True)
class ResidualDomainProblem:
    domains: tuple[QuantizationDomain, ...]
    sites: tuple[ResidualDomainSite, ...]
    available_capabilities: frozenset[str]
    resident_capacity_bytes: int | None

    def __post_init__(self) -> None:
        if self.resident_capacity_bytes is not None and self.resident_capacity_bytes <= 0:
            raise ValueError("resident capacity must be positive when supplied")


@dataclass(frozen=True)
class _ResidualChoice:
    output_error: int
    materialization_bytes: int
    boundary_traffic_bytes: int
    host_operations: int
    decisions: tuple[tuple[int, bool, bool, int], ...]

    @property
    def rank(self) -> tuple[int, int, int, int]:
        return (
            self.materialization_bytes,
            self.boundary_traffic_bytes,
            self.host_operations,
            -self.output_error,
        )


class _UnsupportedResidualTopology(ValueError):
    pass


class _ResidualAccuracyBudgetUnsatisfiable(ValueError):
    pass


def _residual_graph(problem: ResidualDomainProblem) -> tuple[
    dict[str, QuantizationDomain], dict[int, ResidualDomainSite], list[list[int]]
]:
    domains = {domain.name: domain for domain in problem.domains}
    if len(domains) != len(problem.domains):
        raise ValueError("quantization domain names must be unique")
    sites = {site.index: site for site in problem.sites}
    if len(sites) != len(problem.sites):
        raise ValueError("residual site indices must be unique")
    successors: dict[int, list[int]] = {index: [] for index in sites}
    for site in problem.sites:
        if site.output_domain not in domains:
            raise ValueError(f"unknown residual output domain {site.output_domain!r}")
        if site.predecessor is not None:
            if site.predecessor not in sites or site.predecessor >= site.index:
                raise ValueError("residual predecessor must be an earlier topological site")
            successors[site.predecessor].append(site.index)
    if any(len(items) > 1 for items in successors.values()):
        raise _UnsupportedResidualTopology(
            "correlated residual fanout requires a DAG-aware error-budget solver"
        )
    roots = sorted(index for index, site in sites.items() if site.predecessor is None)
    chains: list[list[int]] = []
    seen: set[int] = set()
    for root in roots:
        chain: list[int] = []
        current: int | None = root
        while current is not None:
            chain.append(current)
            seen.add(current)
            current = successors[current][0] if successors[current] else None
        chains.append(chain)
    if seen != set(sites):
        raise ValueError("residual domain graph is not a topological forest")
    return domains, sites, chains


def _carry_bound(
    source: QuantizationDomain, target: QuantizationDomain, upstream_error: int
) -> int | None:
    if source.scale is None or target.scale is None:
        return None
    return int(ceil((upstream_error + 0.5) * source.scale / target.scale))


def _policy_refusals(policy: AccuracyPolicy | None) -> list[str]:
    if policy is None:
        return ["accuracy_policy_not_provided"]
    if not policy.opt_in:
        return ["accuracy_policy_not_opted_in"]
    if not policy.corpora:
        return ["accuracy_corpus_not_provided"]
    return []


def _saturation_refusals(site: ResidualDomainSite, policy: AccuracyPolicy | None) -> list[str]:
    if site.saturation_scope == "universal_no_saturation":
        return []
    if site.saturation_scope == "heldout_no_saturation":
        if policy is None or policy.corpus(site.saturation_corpus_id or "") is None:
            return ["source_saturation_corpus_missing"]
        return []
    return ["source_quantizer_saturation_not_bounded"]


def _capacity_refusal(site: ResidualDomainSite, problem: ResidualDomainProblem) -> str | None:
    if problem.resident_capacity_bytes is None or site.resident_working_set_bytes is None:
        return "resident_capacity_unproven"
    if site.resident_working_set_bytes > problem.resident_capacity_bytes:
        return "resident_capacity_exceeded"
    return None


def _chain_choice(
    chain: Sequence[int],
    domains: Mapping[str, QuantizationDomain],
    sites: Mapping[int, ResidualDomainSite],
    problem: ResidualDomainProblem,
    policy: AccuracyPolicy | None,
) -> _ResidualChoice:
    budget = (
        0 if policy is None or not policy.opt_in
        else policy.budget.max_accumulated_error_lsb
    )
    policy_blockers = _policy_refusals(policy)
    states = [_ResidualChoice(0, 0, 0, 0, ())]
    for position, index in enumerate(chain):
        site = sites[index]
        target = domains[site.output_domain]
        missing_capabilities = sorted(set(site.required_capabilities) - problem.available_capabilities)
        trials: list[_ResidualChoice] = []
        for state in states:
            carry_options = [(False, 0, 0)]
            if position and not policy_blockers:
                predecessor = sites[site.predecessor]  # type: ignore[index]
                source = domains[predecessor.output_domain]
                bound = _carry_bound(source, target, state.output_error)
                if (
                    bound is not None
                    and not site.carry_blockers
                    and not _saturation_refusals(site, policy)
                    and bound <= budget
                ):
                    carry_options.append((True, bound, site.carry_materialization_bytes or 0))
            alignment_options = [False]
            if (
                not policy_blockers
                and budget >= 1
                and not missing_capabilities
                and not site.alignment_blockers
                and _capacity_refusal(site, problem) is None
            ):
                alignment_options.append(True)
            for carry, input_error, carry_bytes in carry_options:
                for alignment in alignment_options:
                    output_error = (
                        site.external_input_error_lsb + input_error + int(alignment)
                    )
                    if output_error > budget:
                        continue
                    alignment_bytes = site.branch_materialization_bytes if alignment else 0
                    eliminated = carry_bytes + alignment_bytes
                    trials.append(_ResidualChoice(
                        output_error=output_error,
                        materialization_bytes=state.materialization_bytes + eliminated,
                        boundary_traffic_bytes=state.boundary_traffic_bytes + eliminated * 2,
                        host_operations=state.host_operations + (
                            len(site.host_operation_ids) if alignment else 0
                        ),
                        decisions=(*state.decisions, (index, carry, alignment, output_error)),
                    ))
        by_error: dict[int, _ResidualChoice] = {}
        for trial in trials:
            incumbent = by_error.get(trial.output_error)
            if incumbent is None or trial.rank > incumbent.rank:
                by_error[trial.output_error] = trial
        states = list(by_error.values())
        if not states:
            raise _ResidualAccuracyBudgetUnsatisfiable(
                f"external input error at residual site {site.site_id!r} exceeds policy budget"
            )
    return max(states, key=lambda choice: choice.rank)


def plan_residual_domains(
    problem: ResidualDomainProblem, policy: AccuracyPolicy | None = None
) -> dict[str, Any]:
    """Select a globally bounded residual rewrite plan, without claiming target emission."""
    try:
        domains, sites, chains = _residual_graph(problem)
    except _UnsupportedResidualTopology as exc:
        reason = "correlated_residual_fanout_requires_dag_solver"
        return {
            "schema": RESIDUAL_SCHEMA,
            "status": "refused_global_topology",
            "policy": None if policy is None else policy.to_dict(),
            "selection": {
                "selected_cross_layer_carries": 0,
                "selected_residual_alignments": 0,
                "lowering_applied": False,
            },
            "sites": [],
            "selected_chains": [],
            "remaining_host_operations": sorted({
                operation for site in problem.sites for operation in site.host_operation_ids
            }),
            "refusal_histogram": {reason: len(problem.sites)},
            "global_refusal": str(exc),
            "claim_boundary": "no approximation or target lowering selected",
        }
    try:
        choices = [_chain_choice(chain, domains, sites, problem, policy) for chain in chains]
    except _ResidualAccuracyBudgetUnsatisfiable as exc:
        reason = "external_input_error_exceeds_accumulated_budget"
        return {
            "schema": RESIDUAL_SCHEMA,
            "status": "refused_accuracy_composition",
            "policy": None if policy is None else policy.to_dict(),
            "selection": {
                "selected_cross_layer_carries": 0,
                "selected_residual_alignments": 0,
                "lowering_applied": False,
            },
            "sites": [],
            "selected_chains": [],
            "remaining_host_operations": sorted({
                operation for site in problem.sites for operation in site.host_operation_ids
            }),
            "refusal_histogram": {reason: 1},
            "global_refusal": str(exc),
            "claim_boundary": "no approximation or target lowering selected",
        }
    selected = {
        index: (carry, alignment, error)
        for choice in choices
        for index, carry, alignment, error in choice.decisions
    }
    policy_reasons = _policy_refusals(policy)
    refusal_histogram: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    for index in sorted(sites):
        site = sites[index]
        carry, alignment, error = selected.get(index, (False, False, 0))
        missing_capabilities = sorted(
            set(site.required_capabilities) - problem.available_capabilities
        )
        alignment_reasons: list[str] = []
        if not alignment:
            alignment_reasons.extend(policy_reasons)
            if (policy is not None and policy.opt_in
                    and policy.budget.max_accumulated_error_lsb < 1):
                alignment_reasons.append("one_code_alignment_error_exceeds_budget")
            if missing_capabilities:
                alignment_reasons.append("missing_target_capability")
            alignment_reasons.extend(site.alignment_blockers)
            if capacity := _capacity_refusal(site, problem):
                alignment_reasons.append(capacity)
            if not alignment_reasons:
                alignment_reasons.append("global_accuracy_budget_reserved_for_larger_region")
        carry_row = None
        if site.predecessor is not None:
            source = domains[sites[site.predecessor].output_domain]
            target = domains[site.output_domain]
            carry_reasons: list[str] = []
            if not carry:
                carry_reasons.extend(policy_reasons)
                carry_reasons.extend(_saturation_refusals(site, policy))
                if source.scale is None or target.scale is None:
                    carry_reasons.append("quantization_scale_ratio_unknown")
                carry_reasons.extend(site.carry_blockers)
                if not carry_reasons:
                    carry_reasons.append("global_accuracy_budget_or_cost_choice")
            carry_row = {
                "predecessor_site": site.predecessor,
                "selected": carry,
                "source_domain": source.name,
                "target_domain": target.name,
                "scale_ratio": (
                    None if source.scale is None or target.scale is None
                    else source.scale / target.scale
                ),
                "bound": "ceil((upstream_code_delta + 0.5) * source_scale / target_scale)",
                "saturation_scope": site.saturation_scope,
                "saturation_corpus_id": site.saturation_corpus_id,
                "materialization_bytes_eliminated_if_selected": (
                    site.carry_materialization_bytes
                ),
                "refusal_reasons": list(dict.fromkeys(carry_reasons)),
            }
            refusal_histogram.update(carry_row["refusal_reasons"])
        alignment_reasons = list(dict.fromkeys(alignment_reasons))
        refusal_histogram.update(alignment_reasons)
        rows.append({
            "index": site.index,
            "site_id": site.site_id,
            "source_operation_ids": list(site.source_operation_ids),
            "host_operation_ids": list(site.host_operation_ids),
            "output_domain": domains[site.output_domain].to_dict(),
            "bounded_output_code_delta": error,
            "external_input_error_lsb": site.external_input_error_lsb,
            "external_error_provenance_sha256": site.external_error_provenance_sha256,
            "carry_from_predecessor": carry_row,
            "alignment": {
                "selected": alignment,
                "local_max_output_code_delta": 1,
                "proof": "abs(round(a)+round(b)-round(a+b))<=1; final clamp is nonexpansive",
                "wide_intermediate": True,
                "branch_clamp_before_add": False,
                "required_capabilities": list(site.required_capabilities),
                "missing_capabilities": missing_capabilities,
                "resident_working_set_bytes": site.resident_working_set_bytes,
                "resident_capacity_bytes": problem.resident_capacity_bytes,
                "capacity_semantics": "target-supplied full live set at this program point",
                "branch_materialization_bytes_eliminated_if_selected": (
                    site.branch_materialization_bytes
                ),
                "refusal_reasons": alignment_reasons,
            },
        })

    rows_by_index = {row["index"]: row for row in rows}
    selected_chains: list[dict[str, Any]] = []
    for chain in chains:
        current: list[int] = []
        for index in chain:
            row = rows_by_index[index]
            if not row["alignment"]["selected"]:
                if current:
                    selected_chains.append({"site_indices": current})
                    current = []
                continue
            carry_selected = bool(
                row["carry_from_predecessor"]
                and row["carry_from_predecessor"]["selected"]
            )
            if current and not carry_selected:
                selected_chains.append({"site_indices": current})
                current = []
            current.append(index)
        if current:
            selected_chains.append({"site_indices": current})
    for chain_index, chain in enumerate(selected_chains):
        chain_rows = [rows_by_index[index] for index in chain["site_indices"]]
        chain.update({
            "chain_index": chain_index,
            "site_ids": [row["site_id"] for row in chain_rows],
            "source_operation_ids": sorted({
                operation for row in chain_rows for operation in row["source_operation_ids"]
            }),
            "maximum_bounded_output_code_delta": max(
                row["bounded_output_code_delta"] for row in chain_rows
            ),
            "retained_internal_quantized_edges": sum(
                bool(row["carry_from_predecessor"] and row["carry_from_predecessor"]["selected"])
                for row in chain_rows
            ),
        })
    selected_carries = sum(
        bool(row["carry_from_predecessor"] and row["carry_from_predecessor"]["selected"])
        for row in rows
    )
    selected_alignments = sum(row["alignment"]["selected"] for row in rows)
    selected_materialization = sum(choice.materialization_bytes for choice in choices)
    selected_traffic = sum(choice.boundary_traffic_bytes for choice in choices)
    selected_host_operations = {
        operation
        for row in rows
        if row["alignment"]["selected"]
        for operation in row["host_operation_ids"]
    }
    all_host_operations = {
        operation for row in rows for operation in row["host_operation_ids"]
    }
    return {
        "schema": RESIDUAL_SCHEMA,
        "status": (
            "analysis_only_exact_default"
            if policy_reasons
            else "selected_bounded_plan"
            if selected_carries or selected_alignments
            else "no_bounded_capable_rewrite"
        ),
        "policy": None if policy is None else policy.to_dict(),
        "domains": [domain.to_dict() for domain in problem.domains],
        "selection": {
            "selected_cross_layer_carries": selected_carries,
            "selected_residual_alignments": selected_alignments,
            "lowering_applied": False,
            "selection_is_target_neutral_rewrite_plan": True,
            "target_adapter_must_revalidate_capability_capacity_and_encoding": True,
        },
        "sites": rows,
        "selected_chains": selected_chains,
        "remaining_host_operations": sorted(all_host_operations - selected_host_operations),
        "cost": {
            "objective": "materialization_bytes_then_boundary_traffic_then_host_operations",
            "selected_materialization_bytes_eliminated": selected_materialization,
            "selected_logical_boundary_traffic_bytes_eliminated": selected_traffic,
            "cycle_prediction_claimed": False,
        },
        "refusal_histogram": dict(sorted(refusal_histogram.items())),
        "accuracy_scope": (
            "exact only" if policy_reasons else
            "globally accumulated output-code bound under explicit policy; heldout saturation "
            "facts remain empirical"
        ),
    }


@dataclass(frozen=True)
class EpilogueSite:
    site_id: str
    source_operation_index: int
    source_contract: QuantizedEpilogueContract
    capability: QuantizedEpilogueCapability
    exact_candidate: QuantizedEpilogueCandidate | None = None
    exact_proof: ExactEpilogueProof | None = None

    def __post_init__(self) -> None:
        if not self.site_id.strip() or self.source_operation_index < 0:
            raise ValueError("epilogue site requires an id and nonnegative source operation")


@dataclass(frozen=True)
class QuantizedProgramNode:
    """One ordered compiler task presented to maximal-region formation."""

    index: int
    source_operation_ids: tuple[int, ...]
    host_operation_ids: tuple[int, ...]
    output_encoding: str | None
    accelerator_eligible: bool
    epilogue_site_id: str | None = None
    residual_site_index: int | None = None
    base_refusal_reasons: tuple[str, ...] = ()
    resident_working_set_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.index < 0 or not self.source_operation_ids:
            raise ValueError("program node requires an ordered index and source ownership")
        if not isinstance(self.accelerator_eligible, bool):
            raise ValueError("program node accelerator eligibility must be explicit")
        if any(index < 0 for index in (*self.source_operation_ids, *self.host_operation_ids)):
            raise ValueError("program operation ids must be nonnegative")
        if self.resident_working_set_bytes is not None and self.resident_working_set_bytes <= 0:
            raise ValueError("node resident working set must be positive when supplied")


@dataclass(frozen=True)
class QuantizedProgramEdge:
    producer: int
    consumer: int
    producer_encoding: str | None
    consumer_encoding: str | None
    value_id: str
    logical_bytes: int
    fanout: int
    residual_carry_site_index: int | None = None
    consumer_error_bound_lsb: int | None = None
    error_bound_provenance_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.producer < 0 or self.consumer <= self.producer:
            raise ValueError("program edge must follow topological node order")
        if not self.value_id.strip() or self.logical_bytes < 0 or self.fanout < 1:
            raise ValueError("program edge requires value identity, bytes, and positive fanout")
        if self.consumer_error_bound_lsb is not None and self.consumer_error_bound_lsb < 0:
            raise ValueError("edge error bound must be nonnegative when supplied")
        has_provenance = (
            isinstance(self.error_bound_provenance_sha256, str)
            and len(self.error_bound_provenance_sha256) == 64
            and all(character in "0123456789abcdef"
                    for character in self.error_bound_provenance_sha256.lower())
        )
        if (self.consumer_error_bound_lsb is not None) != has_provenance:
            raise ValueError("edge error bound and provenance SHA-256 must be supplied together")


@dataclass(frozen=True)
class QuantizedRegionProblem:
    normalized_source_sha256: str
    epilogues: tuple[EpilogueSite, ...]
    residuals: ResidualDomainProblem | None
    nodes: tuple[QuantizedProgramNode, ...]
    edges: tuple[QuantizedProgramEdge, ...]
    resident_capacity_bytes: int | None
    unassigned_host_operation_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if len(self.normalized_source_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.normalized_source_sha256.lower()
        ):
            raise ValueError("quantized program requires normalized source SHA-256")
        if self.resident_capacity_bytes is not None and self.resident_capacity_bytes <= 0:
            raise ValueError("program resident capacity must be positive when supplied")


def _requirement(role: str, reasons: Sequence[str], *, status: str = "required") -> dict[str, Any]:
    if role not in EDIT_REQUIREMENT_ROLES:
        raise ValueError(f"unknown quantized-region edit requirement {role!r}")
    return {"role": role, "status": status, "reason_codes": sorted(set(reasons))}


def plan_quantized_regions(
    problem: QuantizedRegionProblem, policy: AccuracyPolicy | None = None
) -> dict[str, Any]:
    """Admit epilogues, solve residuals, and form maximal legal accelerator regions."""
    nodes = {node.index: node for node in problem.nodes}
    if len(nodes) != len(problem.nodes) or set(nodes) != set(range(len(nodes))):
        raise ValueError("program nodes must have unique dense topological indices")
    epilogues = {site.site_id: site for site in problem.epilogues}
    if len(epilogues) != len(problem.epilogues):
        raise ValueError("epilogue site ids must be unique")
    source_owners = [operation for node in problem.nodes for operation in node.source_operation_ids]
    host_owners = [operation for node in problem.nodes for operation in node.host_operation_ids]
    if len(source_owners) != len(set(source_owners)):
        raise ValueError("program nodes must uniquely own every supplied source operation")
    if (len(host_owners) != len(set(host_owners))
            or set(host_owners).intersection(problem.unassigned_host_operation_ids)):
        raise ValueError("program nodes and unassigned list must uniquely own host operations")
    referenced_epilogues = [
        node.epilogue_site_id for node in problem.nodes if node.epilogue_site_id is not None
    ]
    if set(referenced_epilogues) != set(epilogues) or len(referenced_epilogues) != len(epilogues):
        raise ValueError("every epilogue site must be referenced by exactly one program node")
    for node in problem.nodes:
        if node.epilogue_site_id is not None:
            site = epilogues[node.epilogue_site_id]
            if site.source_operation_index not in node.source_operation_ids:
                raise ValueError("epilogue source producer is not owned by its program node")
    residual_sites = (
        {} if problem.residuals is None else
        {site.index: site for site in problem.residuals.sites}
    )
    referenced_residuals = [
        node.residual_site_index for node in problem.nodes
        if node.residual_site_index is not None
    ]
    if (set(referenced_residuals) != set(residual_sites)
            or len(referenced_residuals) != len(residual_sites)):
        raise ValueError("every residual site must be referenced by exactly one program node")
    if (problem.residuals is not None
            and problem.residuals.resident_capacity_bytes != problem.resident_capacity_bytes):
        raise ValueError("residual and program resident-capacity facts disagree")
    for node in problem.nodes:
        if node.residual_site_index is not None:
            site = residual_sites[node.residual_site_index]
            if not set(site.source_operation_ids).issubset(node.source_operation_ids):
                raise ValueError("residual source operations are not owned by their program node")
            if not set(site.host_operation_ids).issubset(node.host_operation_ids):
                raise ValueError("residual host operations are not owned by their program node")
    fanouts = Counter((edge.producer, edge.value_id) for edge in problem.edges)
    for edge in problem.edges:
        if edge.producer not in nodes or edge.consumer not in nodes:
            raise ValueError("program edge names an unknown node")
        if edge.fanout != fanouts[(edge.producer, edge.value_id)]:
            raise ValueError("program edge fanout disagrees with the complete supplied graph")
        if nodes[edge.producer].output_encoding != edge.producer_encoding:
            raise ValueError("program edge producer encoding disagrees with its node")
        if (edge.residual_carry_site_index is not None
                and edge.residual_carry_site_index not in residual_sites):
            raise ValueError("program edge names an unknown residual carry site")

    admissions: dict[str, EpilogueAdmission] = {}
    for site in problem.epilogues:
        admissions[site.site_id] = admit_epilogue(
            site_id=site.site_id,
            normalized_source_sha256=problem.normalized_source_sha256,
            source_operation_index=site.source_operation_index,
            source_contract=site.source_contract,
            capability=site.capability,
            policy=policy,
            exact_candidate=site.exact_candidate,
            exact_proof=site.exact_proof,
        )
    epilogue_node_admissions = {
        node.index: admissions[node.epilogue_site_id]
        for node in problem.nodes if node.epilogue_site_id is not None
    }
    composition_refusals_by_node: dict[int, list[str]] = {}
    residual_problem = problem.residuals
    if residual_problem is not None:
        adjusted_sites = []
        residual_consumers = {
            node.residual_site_index: node.index
            for node in problem.nodes if node.residual_site_index is not None
        }
        for site in residual_problem.sites:
            consumer = residual_consumers[site.index]
            external_error = site.external_input_error_lsb
            provenance = []
            approximate_producers: list[int] = []
            if site.external_error_provenance_sha256 is not None:
                provenance.append(site.external_error_provenance_sha256)
            for edge in problem.edges:
                if edge.consumer != consumer:
                    continue
                admission = epilogue_node_admissions.get(edge.producer)
                if admission is None or not admission.selected or admission.exact:
                    continue
                if edge.consumer_error_bound_lsb is None:
                    composition_refusals_by_node.setdefault(edge.producer, []).append(
                        "downstream_error_composition_unproven"
                    )
                    continue
                approximate_producers.append(edge.producer)
                external_error += edge.consumer_error_bound_lsb
                provenance.append(edge.error_bound_provenance_sha256 or "")
            if (policy is not None and policy.opt_in
                    and external_error > policy.budget.max_accumulated_error_lsb):
                for producer in approximate_producers:
                    composition_refusals_by_node.setdefault(producer, []).append(
                        "downstream_accumulated_error_budget_exceeded"
                    )
                # These upstream approximations are now refused, so solve the residual from the
                # exact fallback rather than letting a rejected path poison the legal plan.
                external_error = site.external_input_error_lsb
                provenance = (
                    [] if site.external_error_provenance_sha256 is None
                    else [site.external_error_provenance_sha256]
                )
            adjusted_sites.append(replace(
                site,
                external_input_error_lsb=external_error,
                external_error_provenance_sha256=(
                    None if external_error == 0 else canonical_sha256({
                        "consumer_residual_site": site.index,
                        "component_error_provenance_sha256": provenance,
                        "composition": "sum of input L-infinity code bounds in consumer domain",
                    })
                ),
            ))
        residual_problem = replace(residual_problem, sites=tuple(adjusted_sites))
    residual = (
        None if residual_problem is None else plan_residual_domains(residual_problem, policy)
    )
    residual_rows = {
        row["index"]: row for row in (() if residual is None else residual.get("sites", ()))
    }

    node_rows: list[dict[str, Any]] = []
    refusal_histogram: Counter[str] = Counter()
    selected_nodes: set[int] = set()
    for index, node in sorted(nodes.items()):
        reasons = list(node.base_refusal_reasons)
        reasons.extend(composition_refusals_by_node.get(index, ()))
        if not node.accelerator_eligible:
            reasons.append("node_not_accelerator_eligible")
        if node.accelerator_eligible and not node.output_encoding:
            reasons.append("physical_output_encoding_unknown")
        admission = None
        if node.epilogue_site_id is not None:
            admission = admissions.get(node.epilogue_site_id)
            if admission is None:
                reasons.append("epilogue_site_missing")
            elif not admission.selected:
                reasons.extend(admission.receipt.get("reason_codes", ()))
        residual_row = None
        if node.residual_site_index is not None:
            residual_row = residual_rows.get(node.residual_site_index)
            if residual_row is None:
                reasons.append("residual_site_missing_or_globally_refused")
            elif not residual_row["alignment"]["selected"]:
                reasons.extend(residual_row["alignment"]["refusal_reasons"])
        if node.accelerator_eligible:
            if (node.resident_working_set_bytes is None
                    or problem.resident_capacity_bytes is None):
                reasons.append("node_resident_capacity_unproven")
            elif node.resident_working_set_bytes > problem.resident_capacity_bytes:
                reasons.append("node_resident_capacity_exceeded")
        if (admission is not None and admission.selected and not admission.exact
                and residual_row is not None and residual_row["alignment"]["selected"]):
            reasons.append("combined_epilogue_residual_error_composition_unproven")
        reasons = list(dict.fromkeys(reasons))
        if not reasons:
            selected_nodes.add(index)
        refusal_histogram.update(reasons)
        node_rows.append({
            "index": index,
            "source_operation_ids": list(node.source_operation_ids),
            "host_operation_ids_eliminated_if_lowered": list(node.host_operation_ids),
            "selected_for_accelerator_region": not reasons,
            "accelerator_eligible_before_quantized_gates": node.accelerator_eligible,
            "output_encoding": node.output_encoding,
            "resident_working_set_bytes": node.resident_working_set_bytes,
            "resident_capacity_bytes": problem.resident_capacity_bytes,
            "epilogue_site_id": node.epilogue_site_id,
            "epilogue_admission": (
                None if admission is None else {
                    "selected": admission.selected,
                    "exact": admission.exact,
                    "proof_scope": admission.proof_scope,
                    "max_observed_error_lsb": admission.max_observed_error_lsb,
                }
            ),
            "residual_site_index": node.residual_site_index,
            "refusal_reasons": reasons,
        })

    residual_carries = {
        row["index"]: bool(
            row.get("carry_from_predecessor")
            and row["carry_from_predecessor"].get("selected")
        )
        for row in residual_rows.values()
    }
    edge_rows: list[dict[str, Any]] = []
    incompatible_consumers: set[int] = set()
    direct_edges = 0
    eliminated_boundary_bytes = 0
    for edge in problem.edges:
        reasons: list[str] = []
        if edge.producer not in selected_nodes or edge.consumer not in selected_nodes:
            reasons.append("endpoint_not_selected_for_accelerator_region")
        if not edge.producer_encoding or not edge.consumer_encoding:
            reasons.append("physical_encoding_unknown")
        elif edge.producer_encoding != edge.consumer_encoding:
            reasons.append("physical_encoding_transition_required")
        if (
            edge.residual_carry_site_index is not None
            and not residual_carries.get(edge.residual_carry_site_index, False)
        ):
            reasons.append("quantized_residual_carry_not_selected")
        reasons = list(dict.fromkeys(reasons))
        compatible = not reasons
        if compatible:
            direct_edges += 1
            eliminated_boundary_bytes += edge.logical_bytes * 2
        elif edge.producer in selected_nodes and edge.consumer in selected_nodes:
            incompatible_consumers.add(edge.consumer)
        refusal_histogram.update(
            reason for reason in reasons
            if reason != "endpoint_not_selected_for_accelerator_region"
        )
        edge_rows.append({
            "producer": edge.producer,
            "consumer": edge.consumer,
            "value_id": edge.value_id,
            "logical_bytes": edge.logical_bytes,
            "fanout": edge.fanout,
            "producer_encoding": edge.producer_encoding,
            "consumer_encoding": edge.consumer_encoding,
            "direct_internal_boundary": compatible,
            "residual_carry_site_index": edge.residual_carry_site_index,
            "consumer_error_bound_lsb": edge.consumer_error_bound_lsb,
            "error_bound_provenance_sha256": edge.error_bound_provenance_sha256,
            "refusal_reasons": reasons,
        })

    regions: list[dict[str, Any]] = []
    current: list[int] = []
    for index in range(len(nodes)):
        if index not in selected_nodes or index in incompatible_consumers:
            if current:
                regions.append({"node_indices": current})
                current = []
            if index not in selected_nodes:
                continue
        current.append(index)
    if current:
        regions.append({"node_indices": current})
    for region_index, region in enumerate(regions):
        region_nodes = set(region["node_indices"])
        internal_edges = [
            row for row in edge_rows
            if row["producer"] in region_nodes and row["consumer"] in region_nodes
            and row["direct_internal_boundary"]
        ]
        rows = [node_rows[index] for index in region["node_indices"]]
        region.update({
            "region_index": region_index,
            "source_operation_ids": sorted({
                operation for row in rows for operation in row["source_operation_ids"]
            }),
            "host_operation_ids_eliminated_if_lowered": sorted({
                operation
                for row in rows
                for operation in row["host_operation_ids_eliminated_if_lowered"]
            }),
            "internal_boundaries": internal_edges,
            "numeric_admission": sorted({
                row["epilogue_admission"]["proof_scope"]
                for row in rows if row["epilogue_admission"] is not None
            }),
        })
    selected_host_ops = {
        operation
        for row in node_rows
        if row["selected_for_accelerator_region"]
        for operation in row["host_operation_ids_eliminated_if_lowered"]
    }
    all_host_ops = {
        *problem.unassigned_host_operation_ids,
        *(operation for row in node_rows for operation in row["host_operation_ids_eliminated_if_lowered"]),
    }

    requirements: list[dict[str, Any]] = []
    selected_epilogues = [admission for admission in admissions.values() if admission.selected]
    if selected_epilogues:
        requirements.append(_requirement(
            "target_epilogue_emitter", ["selected_epilogue_plan_not_yet_lowered"]
        ))
    if residual is not None and residual["selection"]["selected_residual_alignments"]:
        requirements.append(_requirement(
            "target_residual_region_emitter", ["selected_residual_plan_not_yet_lowered"]
        ))
    encoding_reasons = [
        reason
        for row in edge_rows
        for reason in row["refusal_reasons"]
        if reason in {"physical_encoding_unknown", "physical_encoding_transition_required"}
    ]
    if encoding_reasons:
        requirements.append(_requirement("target_encoding_and_residency", encoding_reasons))
        requirements.append(_requirement("global_quant_domain_planner", encoding_reasons))
    node_encoding_reasons = [
        reason for row in node_rows for reason in row["refusal_reasons"]
        if reason in {"physical_output_encoding_unknown", "node_resident_capacity_unproven",
                      "node_resident_capacity_exceeded"}
    ]
    if node_encoding_reasons and not any(
            requirement["role"] == "target_encoding_and_residency"
            for requirement in requirements):
        requirements.append(_requirement(
            "target_encoding_and_residency", node_encoding_reasons
        ))
    composition_reasons = [
        reason for reasons in composition_refusals_by_node.values() for reason in reasons
    ]
    if composition_reasons and not any(
            requirement["role"] == "global_quant_domain_planner"
            for requirement in requirements):
        requirements.append(_requirement(
            "global_quant_domain_planner", composition_reasons
        ))
    if any("exact_candidate_not_provided" in admission.receipt.get("reason_codes", ())
           for admission in admissions.values()):
        requirements.append(_requirement(
            "source_epilogue_semantics", ["exact_epilogue_candidate_or_proof_missing"]
        ))
    protected_blockers = []
    if policy is None:
        protected_blockers.append("accuracy_policy_not_provided")
    elif not policy.opt_in:
        protected_blockers.append("accuracy_policy_not_opted_in")

    selected_source_ops = {
        operation
        for row in node_rows
        if row["selected_for_accelerator_region"]
        for operation in row["source_operation_ids"]
    }
    return {
        "schema": PLAN_SCHEMA,
        "status": "selected_framework_plan" if selected_nodes else "no_legal_accelerator_region",
        "normalized_source_sha256": problem.normalized_source_sha256,
        "policy": None if policy is None else policy.to_dict(),
        "epilogue_admissions": {
            site_id: admission.receipt for site_id, admission in sorted(admissions.items())
        },
        "residual_domain_plan": residual,
        "nodes": node_rows,
        "edges": edge_rows,
        "accelerator_region_formation": {
            "schema": REGION_SCHEMA,
            "maximal_region_count": len(regions),
            "regions": regions,
            "selected_node_count": len(selected_nodes),
            "selected_source_operation_count": len(selected_source_ops),
            "direct_encoding_compatible_boundary_count": direct_edges,
            "logical_boundary_bytes_eliminated_if_lowered": eliminated_boundary_bytes,
            "remaining_host_operation_ids": sorted(all_host_ops - selected_host_ops),
            "remaining_host_operation_count": len(all_host_ops - selected_host_ops),
            "objective": [
                "maximize legally admitted accelerator source operations",
                "maximize direct encoding-compatible accelerator boundaries",
                "minimize residual host operations",
            ],
            "selection_scope": "complete supplied topological program graph",
        },
        "refusal_histogram": dict(sorted(refusal_histogram.items())),
        "protected_policy_blockers": protected_blockers,
        "agent_edit_requirements": requirements,
        "lowering_applied": False,
        "remaining_target_emitter_boundary": (
            "consume selected epilogue/residual plans, revalidate physical encoding, full-live-set "
            "capacity and target capability, emit the region, then bind structural and numeric receipts"
        ),
        "claims": {
            "exact_default_when_policy_missing": True,
            "cycle_prediction": "UNMEASURED",
            "structural_bytes_are_performance": False,
            "local_heldout_error_is_end_to_end_accuracy": False,
        },
    }
