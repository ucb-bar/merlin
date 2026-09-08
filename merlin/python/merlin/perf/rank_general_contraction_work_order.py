"""Seal rank-general integer-contraction source sites for one static portfolio iteration.

The input is the immutable, complete-graph analysis record already produced by the global
performance harness.  This module neither reparses a model nor asks a target what it supports.
It joins four independent facts already bound into that record:

* the source MAC observer's verified affine domain and yielded accumulator recurrence;
* the captured graph's exact integer operand/result buffers;
* the global plan's source/artifact identities; and
* statically verified source-operation ownership by an emitted task.

Only a rank-general contraction still owned by a host task becomes an authoring site.  Existing
non-host placement and incomplete observations remain visible, but confer no edit assignment.
There are deliberately no workload names, target constants, simulators, timing estimates, or
performance claims in this policy.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
from argparse import ArgumentParser
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from merlin.benchharness import hash_tree
from merlin.perf.compiler_edit_scope import validate_edit_contract, validate_mechanism_catalog


MECHANISM_ID = "t01_02_rank_general_integer_contraction_offload"
INVENTORY_SCHEMA = "portfolio_rank_general_integer_contraction_inventory_v1"

_MEMBER_IDENTITY_FIELDS = frozenset({
    "analysis", "capsule", "capsule_sha256", "full_model_simulation_allowed",
    "required_lanes", "required_tiers", "role",
})
_PIN_FIELDS = (
    "source_sha256", "lowered_sha256", "command_buffer_sha256", "compiler_sha256",
    "logical_dispatch_digest", "plan_digest", "target_facts_sha256",
    "host_verifier_policy_sha256",
)
_MAC_PROOF = "one proved yielded MAC per static affine-domain point"

# These are compiler semantic surface identities, not target or workload identities.  Optional
# surrounding surfaces let one coherent implementation carry geometry through partitioning,
# absorption, and scheduling.  The four required surfaces are the fail-closed minimum.
_REQUIRED_SURFACES = frozenset({
    "integer_contraction_geometry",
    "contraction_affine_map_recognition",
    "rank_general_integer_contraction",
    "contraction_lane_reclassification",
})
_MECHANISM_SURFACES = _REQUIRED_SURFACES | frozenset({
    "global_partition",
    "contraction_placement",
    "global_model_partition_and_batched_contraction",
    "batched_contraction_capacity_guard",
    "contraction_absorbed_source_ops",
    "exclusive_contraction_absorption",
    "ranked_contraction_route",
    "pipeline_issue",
})


def _digest(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(body.encode()).hexdigest()


def _raw_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _pin(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _integer_dtype(value: Any) -> bool:
    if not isinstance(value, str) or len(value) < 2 or value[0] != "i" or value[1] == "0":
        return False
    return all("0" <= character <= "9" for character in value[1:])


def _portfolio_identity(identities: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "schema": "full_model_optimization_portfolio_v1",
        "members": [dict(identity) for identity in identities],
        "selection": "multi_model_pareto_without_invented_static_cycle_total",
        "execution": "bounded_host_admitted_analysis_with_deterministic_record_order",
        "holdout_policy": "separate_post_authoring_evaluation",
        "micro_graphs": "smoke_and_mechanism_calibration_only",
    }


def _ordered_analyses(record: Mapping[str, Any], problems: list[str]
                      ) -> tuple[list[dict[str, Any]], list[Mapping[str, Any]]]:
    """Resolve `/analysis` plus inline members while checking the static portfolio identity."""
    candidate_sha256 = record.get("candidate_sha256")
    portfolio = record.get("portfolio")
    if record.get("schema") != "global_perf_iteration_v1":
        problems.append("record is not a global_perf_iteration_v1")
    if not _pin(candidate_sha256):
        problems.append("iteration candidate identity is not an exact SHA-256 digest")
    if not isinstance(portfolio, Mapping):
        problems.append("iteration portfolio is missing")
        return [], []
    if (portfolio.get("schema") != "full_model_portfolio_iteration_v1"
            or portfolio.get("candidate_sha256") != candidate_sha256
            or portfolio.get("full_model_simulation_allowed") is not False
            or portfolio.get("selection")
            != "multi_model_pareto_without_invented_static_cycle_total"):
        problems.append("iteration portfolio violates the static-only protocol")
    members = portfolio.get("members")
    if not isinstance(members, list) or not members:
        problems.append("ordered portfolio members are unavailable")
        return [], []
    if (portfolio.get("members_total") != len(members)
            or portfolio.get("members_ready") != len(members)):
        problems.append("portfolio is not a complete ready ordered member set")

    identities: list[dict[str, Any]] = []
    analyses: list[Mapping[str, Any]] = []
    for index, member in enumerate(members):
        if not isinstance(member, Mapping):
            problems.append(f"portfolio member {index} is malformed")
            continue
        identity = member.get("identity")
        if not isinstance(identity, Mapping) or set(identity) != _MEMBER_IDENTITY_FIELDS:
            problems.append(f"portfolio member {index} lacks an exact identity")
            continue
        role = "primary" if index == 0 else "training"
        if (identity.get("role") != role
                or identity.get("analysis") != "full_graph_compile_and_static_only"
                or identity.get("full_model_simulation_allowed") is not False
                or not isinstance(identity.get("capsule"), str) or not identity["capsule"]
                or not _pin(identity.get("capsule_sha256"))):
            problems.append(f"portfolio member {index} identity violates the ordered protocol")
        for field in ("required_lanes", "required_tiers"):
            values = identity.get(field)
            if (not isinstance(values, list)
                    or any(not isinstance(value, str) or not value for value in values)
                    or len(values) != len(set(values))):
                problems.append(f"portfolio member {index} has malformed {field}")
        if member.get("status") != "completed":
            problems.append(f"portfolio member {index} analysis is not completed")
        if index == 0:
            if member.get("analysis_ref") != "/analysis" or "analysis" in member:
                problems.append("primary member does not resolve uniquely through /analysis")
            analysis = record.get("analysis")
        else:
            if member.get("analysis_ref") is not None or not isinstance(member.get("analysis"), Mapping):
                problems.append(f"portfolio member {index} lacks one unique embedded analysis")
            analysis = member.get("analysis")
        if not isinstance(analysis, Mapping):
            problems.append(f"portfolio member {index} analysis is missing")
            analysis = {}
        workload = {field: identity.get(field) for field in (
            "capsule", "capsule_sha256", "required_lanes", "required_tiers")}
        if analysis.get("workload") != workload:
            problems.append(f"portfolio member {index} analysis workload differs from its identity")
        if analysis.get("candidate_sha256") != candidate_sha256:
            problems.append(f"portfolio member {index} analysis belongs to another candidate")
        identities.append(dict(identity))
        analyses.append(analysis)

    declared = portfolio.get("portfolio_sha256")
    if (len(identities) != len(members) or not _pin(declared)
            or _digest(_portfolio_identity(identities)) != declared):
        problems.append("ordered member identities do not match the exact portfolio hash")
    return identities, analyses


def _analysis_parts(analysis: Mapping[str, Any], candidate_sha256: str
                    ) -> tuple[dict[str, Any], list[str]]:
    """Validate artifact/graph/plan identities before returning source-local structures."""
    problems: list[str] = []
    diagnostics = analysis.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        return {}, ["analysis diagnostics are missing"]
    plan = diagnostics.get("verified_global_plan_emission")
    graph = diagnostics.get("captured_logical_graph")
    evidence = diagnostics.get("task_instruction_evidence")
    placement_pair = diagnostics.get("model_contraction_placement")
    emission = analysis.get("emission")
    if not all(isinstance(value, Mapping) for value in (
            plan, graph, evidence, placement_pair, emission)):
        return {}, ["analysis lacks plan, graph, placement, task, or emission evidence"]
    tasks = evidence.get("candidate")
    placement = placement_pair.get("candidate")
    program = graph.get("dispatch_program")
    if not all(isinstance(value, Mapping) for value in (tasks, placement, program)):
        return {}, ["analysis lacks candidate placement, task ownership, or graph payload"]
    binding = tasks.get("binding")
    if not isinstance(binding, Mapping):
        return {}, ["task ownership binding is missing"]

    if plan.get("status") != "verified":
        problems.append("global plan emission is not verified")
    if graph.get("schema") != "captured_global_graph_v1" or graph.get("status") != "verified":
        problems.append("captured logical graph is not verified")
    if (tasks.get("schema") != "task_instruction_evidence_v1"
            or tasks.get("status") != "static_ownership_verified"
            or tasks.get("declared_source_plan_status") != "verified"):
        problems.append("candidate source-task ownership is not statically verified")
    if (placement.get("schema") != "model_contraction_placement_v1"
            or not isinstance(placement.get("contractions"), list)
            or not isinstance(placement.get("unresolved"), list)):
        problems.append("candidate contraction placement evidence is incomplete")
    if placement.get("conflicting_regions") not in (None, []):
        problems.append("candidate contraction placement has conflicting region lanes")
    for field in _PIN_FIELDS:
        if not _pin(binding.get(field)):
            problems.append(f"task ownership binding lacks exact {field}")
    expected = {
        "source_sha256": plan.get("source_sha256"),
        "lowered_sha256": plan.get("candidate_lowered_sha256"),
        "command_buffer_sha256": plan.get("candidate_command_buffer_sha256"),
        "compiler_sha256": candidate_sha256,
        "logical_dispatch_digest": plan.get("logical_dispatch_digest"),
        "plan_digest": plan.get("plan_digest"),
    }
    for field, value in expected.items():
        if not _pin(value) or binding.get(field) != value:
            problems.append(f"plan and task ownership disagree on {field}")
    if (analysis.get("candidate_sha256") != candidate_sha256
            or graph.get("source_sha256") != binding.get("source_sha256")
            or graph.get("logical_dispatch_digest") != binding.get("logical_dispatch_digest")
            or emission.get("candidate_command_buffer_sha256") != binding.get("command_buffer_sha256")
            or emission.get("candidate_lowered_sha256") != binding.get("lowered_sha256")):
        problems.append("analysis artifact identities disagree with the verified task binding")
    if _digest(program) != graph.get("logical_dispatch_digest"):
        problems.append("logical dispatch payload does not match its bound digest")

    nodes, buffers, task_rows = program.get("nodes"), program.get("buffers"), tasks.get("tasks")
    if not isinstance(nodes, list) or not isinstance(buffers, Mapping):
        problems.append("captured graph nodes or buffers are unavailable")
        nodes, buffers = [], {}
    if not isinstance(task_rows, list):
        problems.append("source-task ownership rows are unavailable")
        task_rows = []
    if (graph.get("nodes") != len(nodes) or plan.get("source_operations") != len(nodes)
            or plan.get("tasks") != len(task_rows)):
        problems.append("graph, plan, and task counts disagree")

    owners: dict[int, Mapping[str, Any]] = {}
    for task_index, task in enumerate(task_rows):
        if (not isinstance(task, Mapping) or task.get("task_index") != task_index
                or not isinstance(task.get("declared_task_kind"), str)
                or not task["declared_task_kind"]
                or not isinstance(task.get("source_op_indices"), list)):
            problems.append("source-task ownership row is malformed")
            continue
        indices = task["source_op_indices"]
        if indices != sorted(set(indices)):
            problems.append(f"task {task_index} source operation IDs are not sorted and unique")
        for source_index in indices:
            if type(source_index) is not int or not 0 <= source_index < len(nodes):
                problems.append(f"task {task_index} owns an absent source operation")
            elif source_index in owners:
                problems.append(f"source operation {source_index} has multiple owners")
            else:
                owners[source_index] = task
    missing_owners = set(range(len(nodes))) - set(owners)
    if missing_owners:
        problems.append("source-task ownership does not cover every graph node exactly once")

    return {
        "plan": plan, "graph": graph, "program": program, "nodes": nodes, "buffers": buffers,
        "placement": placement, "binding": binding, "owners": owners,
        "missing_owners": missing_owners,
    }, sorted(set(problems))


def _site(row: Mapping[str, Any], parts: Mapping[str, Any]) -> tuple[dict[str, Any] | None,
                                                                    list[str]]:
    reasons: list[str] = []
    index = row.get("source_op_index")
    parallel, reduction = row.get("parallel"), row.get("reduction")
    if (not isinstance(parallel, list) or len(parallel) < 2
            or not isinstance(reduction, list) or not reduction
            or any(type(value) is not int or value <= 0 for value in [*parallel, *reduction])):
        reasons.append("rank evidence lacks exact positive parallel/reduction dimensions")
    if row.get("mac_status") != "derived" or row.get("mac_basis") != _MAC_PROOF:
        reasons.extend([
            "affine-map evidence is not a proved static source domain",
            "accumulator evidence is not a proved yielded multiply-add recurrence",
        ])
    if type(index) is not int or not 0 <= index < len(parts["nodes"]):
        return None, sorted(set([*reasons, "source operation index is absent from the exact graph"]))
    node = parts["nodes"][index]
    if not isinstance(node, Mapping) or node.get("kind") != "dispatch":
        reasons.append("source operation is not an exact captured dispatch")
        node = {}
    provenance = node.get("prov") if isinstance(node.get("prov"), Mapping) else {}
    if (provenance.get("prov.family") != "contraction"
            or provenance.get("prov.region_id") != row.get("region")):
        reasons.append("source graph and contraction observer region identities disagree")
    inputs, outputs = node.get("inputs"), node.get("outputs")
    if (not isinstance(inputs, list) or len(inputs) < 2 or not isinstance(outputs, list)
            or len(outputs) != 1):
        reasons.append("accumulator evidence lacks exact operand/result edges")
        inputs, outputs = [], []
    buffers = parts["buffers"]
    input_rows = [buffers.get(name) for name in inputs]
    output_row = buffers.get(outputs[0]) if outputs else None
    if (any(not isinstance(value, Mapping) for value in input_rows)
            or not isinstance(output_row, Mapping)):
        reasons.append("accumulator evidence references absent graph buffers")
        input_rows, output_row = [], {}
    dtypes = [value.get("dtype") for value in input_rows]
    accumulator_dtype = output_row.get("dtype")
    if (len(dtypes) < 2 or any(not _integer_dtype(dtype) for dtype in dtypes[:2])
            or not _integer_dtype(accumulator_dtype)
            or any(dtype != accumulator_dtype for dtype in dtypes[2:])):
        reasons.append("accumulator evidence lacks exact integer operands/result/initializer")
    owner = parts["owners"].get(index)
    if not isinstance(owner, Mapping):
        reasons.append("ownership evidence has no unique source-task owner")
    if reasons:
        return None, sorted(set(reasons))

    extent_product = math.prod([*parallel, *reduction])
    if type(row.get("macs")) is not int or row["macs"] != extent_product:
        return None, ["rank evidence and declared static MAC domain disagree"]
    site = {
        "source_operation_id": index,
        "source_region_id": row["region"],
        "source_operation": row.get("op"),
        "parallel_extents": list(parallel),
        "reduction_extents": list(reduction),
        "iteration_rank": len(parallel) + len(reduction),
        "operand_dtypes": dtypes[:2],
        "accumulator_evidence": {
            "status": "proved", "dtype": accumulator_dtype,
            "source": "yielded_integer_mac_recurrence",
        },
        "map_evidence": {
            "status": "proved", "source": "static_affine_domain_observer",
            "symbols": "none", "domain_macs": extent_product,
        },
        "ownership_evidence": {
            "status": "proved", "task_index": owner["task_index"],
            "declared_task_kind": owner["declared_task_kind"],
            "task_binding_sha256": _digest({field: parts["binding"][field]
                                            for field in _PIN_FIELDS}),
        },
        "declared_lane": row.get("lane"),
        "proof_scope": (
            "source rank/maps/integer recurrence and exact static task ownership; authoring site only"
        ),
        "not_proven": [
            "target eligibility", "replacement semantic equivalence", "emitted work deletion",
            "runtime correctness", "cycle improvement",
        ],
    }
    site["site_binding_sha256"] = _digest(site)
    return site, []


def _member_inventory(analysis: Mapping[str, Any], identity: Mapping[str, Any],
                      candidate_sha256: str, member_index: int) -> dict[str, Any]:
    parts, problems = _analysis_parts(analysis, candidate_sha256)
    member = {
        "member_index": member_index,
        "identity": dict(identity),
        "identity_sha256": _digest(identity),
        "analysis_location": "/analysis" if member_index == 0 else
                             f"/portfolio/members/{member_index}/analysis",
        "analysis_sha256": _digest(analysis),
        "status": "not_ready",
        "source_operation_ids": [],
        "host_rejected": [],
        "already_offloaded": [],
        "uncaptured": [],
        "refusals": [],
        "problems": problems,
    }
    if not parts:
        member["inventory_sha256"] = _digest(member)
        return member

    observed: set[int] = set()
    for raw in parts["placement"].get("contractions", []):
        if not isinstance(raw, Mapping):
            member["uncaptured"].append({
                "source_operation_id": None, "refusal_class": "uncaptured_contraction",
                "reasons": ["contraction placement row is malformed"],
            })
            continue
        index = raw.get("source_op_index")
        if type(index) is int:
            observed.add(index)
        site, reasons = _site(raw, parts)
        if site is None:
            member["uncaptured"].append({
                "source_operation_id": index,
                "refusal_class": "uncaptured_contraction",
                "reasons": reasons,
            })
            continue
        if site["iteration_rank"] <= 3:
            member["refusals"].append({
                "source_operation_id": index,
                "refusal_class": "canonical_rank",
                "reasons": ["operation does not have leading or additional contraction dimensions"],
                "site": site,
            })
        elif site["ownership_evidence"]["declared_task_kind"] == "host":
            member["host_rejected"].append(site)
        else:
            member["already_offloaded"].append(site)

    for raw in parts["placement"].get("unresolved", []):
        index = raw.get("source_op_index") if isinstance(raw, Mapping) else None
        if type(index) is int:
            observed.add(index)
        reason = raw.get("mac_basis") if isinstance(raw, Mapping) else None
        member["uncaptured"].append({
            "source_operation_id": index,
            "refusal_class": "uncaptured_contraction",
            "reasons": [reason if isinstance(reason, str) and reason else
                        "contraction observer left the source domain unresolved"],
        })

    # Provenance alone never creates an opportunity.  It only keeps an explicitly role-tagged source
    # contraction visible when the structural observer could not bind maps/recurrence evidence.
    for index, node in enumerate(parts["nodes"]):
        provenance = node.get("prov") if isinstance(node, Mapping) else None
        if (index not in observed and isinstance(provenance, Mapping)
                and provenance.get("prov.family") == "contraction"
                and provenance.get("prov.role") == "contraction"):
            member["uncaptured"].append({
                "source_operation_id": index,
                "refusal_class": "uncaptured_contraction",
                "reasons": [
                    "explicit source contraction role has no complete affine-domain MAC witness"],
            })

    # A contradictory ownership denominator invalidates every otherwise eligible assignment.
    if parts["missing_owners"]:
        retained = []
        for site in member["host_rejected"]:
            retained.append({
                "source_operation_id": site["source_operation_id"],
                "refusal_class": "uncaptured_contraction",
                "reasons": ["ownership evidence is incomplete for the full source graph"],
            })
        member["uncaptured"].extend(retained)
        member["host_rejected"] = []
    member["host_rejected"].sort(key=lambda row: row["source_operation_id"])
    member["already_offloaded"].sort(key=lambda row: row["source_operation_id"])
    member["uncaptured"].sort(key=lambda row: (
        -1 if row["source_operation_id"] is None else row["source_operation_id"], row["reasons"]))
    member["refusals"].sort(key=lambda row: row["source_operation_id"])
    member["source_operation_ids"] = [row["source_operation_id"]
                                      for row in member["host_rejected"]]
    member["status"] = (
        "eligible_sites_bound" if member["source_operation_ids"] and not problems
        else "no_eligible_sites" if not problems else "not_ready"
    )
    member["bindings"] = {field: parts["binding"][field] for field in _PIN_FIELDS}
    member["inventory_sha256"] = _digest(member)
    return member


def inventory_portfolio_rank_general_contractions(
        record: Mapping[str, Any], *, iteration_record_sha256: str | None = None,
) -> dict[str, Any]:
    """Derive exact per-member source sites from one ordered static analysis iteration."""
    result = {
        "schema": INVENTORY_SCHEMA,
        "mechanism_id": MECHANISM_ID,
        "status": "not_ready",
        "candidate_sha256": record.get("candidate_sha256") if isinstance(record, Mapping) else None,
        "portfolio_sha256": None,
        "iteration_record_sha256": iteration_record_sha256,
        "iteration_payload_sha256": _digest(record) if isinstance(record, Mapping) else None,
        "ordered_portfolio": [],
        "members": [],
        "problems": [],
        "proof_scope": "static source-site assignment only",
        "not_proven": [
            "target eligibility", "semantic equivalence", "emitted work deletion",
            "runtime correctness", "performance improvement",
        ],
    }
    if not isinstance(record, Mapping):
        result["problems"] = ["iteration record is missing"]
        result["sha256"] = _digest(result)
        return result
    if iteration_record_sha256 is not None and not _pin(iteration_record_sha256):
        result["problems"].append("iteration_record_sha256 is not an exact SHA-256 digest")
    identities, analyses = _ordered_analyses(record, result["problems"])
    portfolio = record.get("portfolio") if isinstance(record.get("portfolio"), Mapping) else {}
    result["portfolio_sha256"] = portfolio.get("portfolio_sha256")
    result["ordered_portfolio"] = identities
    if len(identities) != len(analyses) or not identities:
        result["sha256"] = _digest(result)
        return result
    for index, (identity, analysis) in enumerate(zip(identities, analyses, strict=True)):
        result["members"].append(_member_inventory(
            analysis, identity, record["candidate_sha256"], index))
    if any(member["status"] == "not_ready" for member in result["members"]):
        result["problems"].append("one or more portfolio member inventories are not ready")
    opportunities = sum(len(member["source_operation_ids"]) for member in result["members"])
    result["status"] = (
        "ready_for_work_order" if not result["problems"] and opportunities else
        "no_eligible_sites" if not result["problems"] else "not_ready"
    )
    result["problems"] = sorted(set(result["problems"]))
    result["sha256"] = _digest(result)
    return result


def _embedded_edit_contract(record: Mapping[str, Any], analyses: Sequence[Mapping[str, Any]],
                            candidate: Path) -> dict[str, Any]:
    binding = record.get("cross_run_static_analysis_binding")
    authority = binding.get("compiler_edit_authority") if isinstance(binding, Mapping) else None
    if (not isinstance(authority, Mapping)
            or authority.get("schema") != "host_frozen_compiler_edit_authority_v1"
            or authority.get("source_pins_checked") is not True):
        raise ValueError("iteration lacks frozen compiler edit authority")
    contract = authority.get("contract")
    if (not isinstance(contract, Mapping) or _digest(contract) != authority.get(
            "contract_document_sha256")):
        raise ValueError("iteration compiler edit authority changed")
    for analysis in analyses:
        # Per-member guidance may intentionally omit helper extensions.  It is descriptive and
        # cannot grant authority.  The enforced compiler-edit receipt must still point at the one
        # host-frozen contract carried by the cross-run binding.
        scope = analysis.get("compiler_edit_scope")
        if (not isinstance(scope, Mapping) or scope.get("status") != "allowed"
                or scope.get("contract_sha256") != contract.get("sha256")):
            raise ValueError("portfolio member is not bound to the frozen compiler edit authority")
    return validate_edit_contract(contract, candidate)


def _definition_kinds(source: str) -> dict[str, str]:
    result: dict[str, str] = {}

    def visit(nodes: Sequence[ast.stmt], parents: tuple[str, ...] = ()) -> None:
        for node in nodes:
            if isinstance(node, ast.ClassDef):
                name = ".".join((*parents, node.name))
                result[name] = "class"
                visit(node.body, (*parents, node.name))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = ".".join((*parents, node.name))
                result[name] = "method" if parents else "function"

    visit(ast.parse(source).body)
    return result


def _mechanism_catalog(contract: Mapping[str, Any], candidate: Path) -> dict[str, Any]:
    selected = [row for row in contract.get("existing_symbols", [])
                if isinstance(row, Mapping) and row.get("surface_id") in _MECHANISM_SURFACES]
    found = {row.get("surface_id") for row in selected}
    if _REQUIRED_SURFACES - found:
        raise ValueError("edit contract lacks required rank-general contraction surfaces")
    selectors: list[dict[str, Any]] = []
    paths: set[str] = set()
    seen: set[tuple[str, str, str | None]] = set()
    for row in selected:
        path, symbol = row.get("path"), row.get("symbol")
        if not isinstance(path, str) or not isinstance(symbol, str):
            raise ValueError("rank-general compiler surface identity is malformed")
        source = candidate / path
        if source.is_symlink() or not source.is_file():
            raise ValueError("rank-general compiler surface source is missing")
        kind = _definition_kinds(source.read_text()).get(symbol)
        if kind not in {"class", "function", "method"}:
            raise ValueError("rank-general compiler surface does not resolve to an exact AST unit")
        key = (kind, path, symbol)
        if key not in seen:
            selectors.append({"kind": kind, "path": path, "symbol": symbol})
            seen.add(key)
            paths.add(path)
    # Imports are semantic units in the one-mechanism audit.  Owning them only in files that also
    # contain an exact selected surface permits required type/helper imports without widening paths.
    for path in sorted(paths):
        selectors.append({"kind": "imports", "path": path})
    for extension in contract.get("helper_extensions", []):
        if (isinstance(extension, Mapping)
                and set(extension.get("surface_ids", [])) & _MECHANISM_SURFACES):
            selectors.append({"kind": "helper", "directory": extension["directory"]})
    selectors.sort(key=lambda row: (row["kind"], row.get("path", row.get("directory", "")),
                                    row.get("symbol", "")))
    catalog = {
        "schema": "compiler_mechanism_catalog_v1",
        "contract_sha256": contract["sha256"],
        "mechanisms": [{"id": MECHANISM_ID, "selectors": selectors}],
    }
    catalog["sha256"] = _digest(catalog)
    return validate_mechanism_catalog(catalog, candidate, contract)


def build_rank_general_mechanism_documents(
        record: Mapping[str, Any], *, candidate: Path, expected_candidate_sha256: str,
        iteration_record_sha256: str,
) -> dict[str, Any]:
    """Build, but do not write, the exact contract/catalog/inventory/work-order documents."""
    candidate = Path(candidate)
    if (not candidate.is_absolute() or candidate.resolve() != candidate or candidate.is_symlink()
            or not candidate.is_dir() or not _pin(expected_candidate_sha256)
            or hash_tree(candidate)["sha256"] != expected_candidate_sha256
            or record.get("candidate_sha256") != expected_candidate_sha256):
        raise ValueError("explicit candidate path/hash does not match the static iteration")
    if not _pin(iteration_record_sha256):
        raise ValueError("static iteration requires an exact raw SHA-256 digest")
    protocol_problems: list[str] = []
    _identities, analyses = _ordered_analyses(record, protocol_problems)
    if protocol_problems:
        raise ValueError("static iteration portfolio is not exact: " + "; ".join(protocol_problems))
    contract = _embedded_edit_contract(record, analyses, candidate)
    inventory = inventory_portfolio_rank_general_contractions(
        record, iteration_record_sha256=iteration_record_sha256)
    if inventory["status"] != "ready_for_work_order":
        raise ValueError("static iteration has no fully evidenced rank-general host opportunity")
    catalog = _mechanism_catalog(contract, candidate)
    site_bindings = []
    for member in inventory["members"]:
        bindings = member["bindings"]
        site_bindings.append({
            "capsule": member["identity"]["capsule"],
            "capsule_sha256": member["identity"]["capsule_sha256"],
            "compiler_sha256": bindings["compiler_sha256"],
            "source_sha256": bindings["source_sha256"],
            "plan_digest": bindings["plan_digest"],
            "candidate_command_buffer_sha256": bindings["command_buffer_sha256"],
            "candidate_lowered_sha256": bindings["lowered_sha256"],
            "status": member["status"],
            "source_operation_ids": member["source_operation_ids"],
            "chains": member["host_rejected"],
            "inventory": {
                "schema": INVENTORY_SCHEMA,
                "member_index": member["member_index"],
                "analysis_location": member["analysis_location"],
                "analysis_sha256": member["analysis_sha256"],
                "member_inventory_sha256": member["inventory_sha256"],
                "portfolio_inventory_sha256": inventory["sha256"],
                "iteration_record_sha256": iteration_record_sha256,
                "already_offloaded_count": len(member["already_offloaded"]),
                "uncaptured_count": len(member["uncaptured"]),
            },
        })
    work_order = {
        "schema": "host_prepared_mechanism_work_order_v1",
        "status": "ready_for_authoring",
        "mechanism_id": MECHANISM_ID,
        "catalog_sha256": catalog["sha256"],
        "contract_sha256": contract["sha256"],
        "initial_candidate_sha256": expected_candidate_sha256,
        "round_start_candidate_sha256": expected_candidate_sha256,
        "portfolio_sha256": inventory["portfolio_sha256"],
        "ordered_portfolio": inventory["ordered_portfolio"],
        "source_operation_ids": [],
        "portfolio_site_bindings": site_bindings,
    }
    work_order["sha256"] = _digest(work_order)
    return {"edit_contract": contract, "catalog": catalog,
            "inventory": inventory, "work_order": work_order}


def _write_once(path: Path, document: Mapping[str, Any]) -> str:
    raw = (json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(raw)
    path.chmod(0o444)
    return _raw_sha256(raw)


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(
        description="Seal one rank-general integer-contraction Phase-2 catalog/work order")
    parser.add_argument("iteration", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--iteration-sha256", required=True)
    parser.add_argument("--candidate-sha256", required=True)
    args = parser.parse_args(argv)
    iteration, candidate, output = args.iteration, args.candidate, args.output
    if (not iteration.is_absolute() or iteration.resolve() != iteration or iteration.is_symlink()
            or not iteration.is_file() or iteration.stat().st_mode & 0o222):
        parser.error("iteration must be an absolute, read-only, non-symlink regular file")
    raw = iteration.read_bytes()
    if not _pin(args.iteration_sha256) or _raw_sha256(raw) != args.iteration_sha256:
        parser.error("iteration raw SHA-256 pin does not match")
    try:
        record = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        parser.error(f"iteration is not valid JSON: {exc}")
    try:
        documents = build_rank_general_mechanism_documents(
            record, candidate=candidate, expected_candidate_sha256=args.candidate_sha256,
            iteration_record_sha256=args.iteration_sha256)
        if hash_tree(candidate)["sha256"] != args.candidate_sha256:
            raise ValueError("candidate changed while deriving the sealed work order")
        output.mkdir(mode=0o755, parents=False, exist_ok=False)
        names = {
            "compiler_edit_contract.json": documents["edit_contract"],
            "mechanism_catalog.json": documents["catalog"],
            "mechanism_work_order.json": documents["work_order"],
            "source_site_inventory.json": documents["inventory"],
        }
        artifacts = {name: {"sha256": _write_once(output / name, document)}
                     for name, document in names.items()}
        receipt = {
            "schema": "sealed_rank_general_contraction_work_order_receipt_v1",
            "mechanism_id": MECHANISM_ID,
            "candidate_sha256": args.candidate_sha256,
            "iteration_record_sha256": args.iteration_sha256,
            "inventory_sha256": documents["inventory"]["sha256"],
            "catalog_sha256": documents["catalog"]["sha256"],
            "work_order_sha256": documents["work_order"]["sha256"],
            "artifacts": dict(artifacts),
        }
        receipt["sha256"] = _digest(receipt)
        receipt_raw_sha = _write_once(output / "receipt.json", receipt)
        artifacts["receipt.json"] = {"sha256": receipt_raw_sha}
    except (FileExistsError, OSError, ValueError) as exc:
        parser.error(str(exc))
    report = {
        "status": "ready_for_authoring",
        "mechanism_id": MECHANISM_ID,
        "output": str(output),
        "opportunity_counts": [len(member["source_operation_ids"])
                               for member in documents["inventory"]["members"]],
        "already_offloaded_counts": [len(member["already_offloaded"])
                                     for member in documents["inventory"]["members"]],
        "uncaptured_counts": [len(member["uncaptured"])
                              for member in documents["inventory"]["members"]],
        "artifacts": artifacts,
    }
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
