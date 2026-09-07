"""Find exact source sites for host pointwise/materialization work deletion.

This is an inventory, not a fusion proof.  It joins the host-reconstructed logical DAG to the
independently verified source-task ownership record and emits source-operation indices only when
the graph, plan, command buffer, lowered artifact, compiler, and host policy identities agree.
It never consults workload names or aggregate operation counts.
"""
from __future__ import annotations

import hashlib
import json
from argparse import ArgumentParser
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any


SCHEMA = "host_epilogue_site_inventory_v1"
PORTFOLIO_SCHEMA = "portfolio_host_epilogue_site_inventory_v1"
MECHANISM_ID = "t01_01_exact_host_epilogue_materialization_deletion"

_PIN_FIELDS = (
    "source_sha256",
    "lowered_sha256",
    "command_buffer_sha256",
    "compiler_sha256",
    "logical_dispatch_digest",
    "plan_digest",
    "target_facts_sha256",
    "host_verifier_policy_sha256",
)
_POINTWISE_CATEGORIES = frozenset({"elementwise", "quantize_requant"})
_MATERIALIZATION_CATEGORIES = frozenset({"alloc", "fill_init", "layout_copy", "layout_view"})
_MEMBER_IDENTITY_FIELDS = frozenset({
    "analysis", "capsule", "capsule_sha256", "full_model_simulation_allowed",
    "required_lanes", "required_tiers", "role",
})


def _digest(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(body.encode()).hexdigest()


def _pin(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _mapping(value: Any, path: str, missing: list[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        missing.append(path)
        return {}
    return value


def _analysis(record: Mapping[str, Any], missing: list[str]) -> Mapping[str, Any]:
    if record.get("schema") in {
            "host_owned_whole_model_emission_analysis_v2",
            "host_owned_whole_model_emission_analysis_v1",
    }:
        return record
    return _mapping(record.get("analysis"), "analysis", missing)


def _semantic_site(node: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """Classify only from the host graph's explicit provenance or structural view op."""
    from merlin.llvmlower.op_profile import resolve_category

    kind = node.get("kind")
    if kind == "view":
        op = node.get("op")
        if not isinstance(op, str) or not op:
            return None, "view operation name is missing"
        category, source = resolve_category({"mlir_op": op})
        if category not in _MATERIALIZATION_CATEGORIES:
            return None, None
        if (type(node.get("regions")) is not int or node["regions"] < 0
                or not isinstance(node.get("captures"), list)):
            return None, "view capture/region semantics are incomplete"
        if node["regions"]:
            return None, "region-carrying view semantics require exact source-body inspection"
        return {
            "site_kind": "materialization",
            "category": category,
            "classification_source": source,
            "operation": op,
        }, None

    if kind != "dispatch":
        return None, "node kind is neither an explicit dispatch nor view"
    provenance = node.get("prov")
    if not isinstance(provenance, Mapping):
        return None, "dispatch provenance is missing"
    family, operation = provenance.get("prov.family"), provenance.get("prov.op")
    if not isinstance(family, str) or not family or not isinstance(operation, str) or not operation:
        return None, "dispatch lacks both explicit prov.family and prov.op semantic labels"
    by_family = resolve_category({"family": family})
    by_operation = resolve_category({"op": operation})
    if by_family[1] == "unknown":
        return None, "prov.family does not resolve to a known semantic category"
    # The family vocabulary is deliberately coarser and more stable than frontend operation
    # spellings.  An operation spelling absent from the shared table is still useful exact evidence
    # when its explicit family is known; it is retained as a label, never guessed from the symbol.
    if by_operation[1] != "unknown" and by_family[0] != by_operation[0]:
        return None, "prov.family and prov.op resolve to conflicting semantic categories"
    if by_family[0] not in _POINTWISE_CATEGORIES:
        return None, None
    if (type(node.get("regions")) is not int or node["regions"] != 0
            or node.get("captures") != []):
        return None, "dispatch call-site capture/region semantics are incomplete"
    return {
        "site_kind": "pointwise",
        "category": by_family[0],
        "classification_source": [by_family[1], "prov.op_label"],
        "operation": operation,
        "family": family,
        "region_id": provenance.get("prov.region_id"),
    }, None


def _refusal_reasons(index: int, node: Mapping[str, Any], *, uses: Mapping[str, list[int]],
                     results: set[str], owner: Mapping[int, Mapping[str, Any]],
                     relevant: Mapping[int, Mapping[str, Any]], nodes: list[Mapping[str, Any]]) -> list[str]:
    reasons: list[str] = []
    outputs = node.get("outputs", [])
    if not outputs:
        return ["source operation has no explicit output buffer"]
    for output in outputs:
        consumers = uses.get(output, [])
        if output in results:
            reasons.append(f"buffer {output} is a full-program result boundary")
        if not consumers:
            reasons.append(f"buffer {output} has no recorded consumer")
        elif len(consumers) != 1:
            reasons.append(f"buffer {output} has fanout {len(consumers)}; one-use edge refused")
        else:
            consumer = consumers[0]
            if owner[consumer]["task_index"] != owner[index]["task_index"]:
                reasons.append(
                    f"buffer {output} crosses task {owner[index]['task_index']}"
                    f"->{owner[consumer]['task_index']}")
            elif consumer not in relevant:
                reasons.append(
                    f"buffer {output} enters non-pointwise/materialization operation "
                    f"{consumer} ({nodes[consumer].get('op')})")
    return sorted(set(reasons)) or ["no one-use pointwise/materialization neighbour"]


def inventory_host_epilogue_sites(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return candidate source IDs and fail-closed refusals from one exact analysis record.

    ``ready_for_source_site_binding`` means the IDs are safe inputs to an authoring work order.  It
    does *not* mean a rewrite is semantically valid: exact source bodies, arithmetic order, shape,
    aliasing, and reduced numerical witnesses remain mandatory before any change is retained.
    """
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "mechanism_id": MECHANISM_ID,
        "status": "not_ready",
        "source_operation_ids": [],
        "chains": [],
        "refusals": [],
        "missing_fields": [],
        "problems": [],
        "proof_scope": (
            "exact logical-DAG source indices joined to verified host task ownership and "
            "artifact identities; candidate discovery only"
        ),
        "not_proven": [
            "source-body arithmetic equivalence",
            "legal fusion or materialization deletion",
            "changed emitted work",
            "cycle improvement",
        ],
    }
    if not isinstance(record, Mapping):
        result["missing_fields"] = ["record"]
        return result

    missing: list[str] = []
    analysis = _analysis(record, missing)
    diagnostics = _mapping(analysis.get("diagnostics"), "analysis.diagnostics", missing)
    plan = _mapping(diagnostics.get("verified_global_plan_emission"),
                    "analysis.diagnostics.verified_global_plan_emission", missing)
    graph = _mapping(diagnostics.get("captured_logical_graph"),
                     "analysis.diagnostics.captured_logical_graph", missing)
    program = _mapping(graph.get("dispatch_program"),
                       "analysis.diagnostics.captured_logical_graph.dispatch_program", missing)
    task_evidence = _mapping(diagnostics.get("task_instruction_evidence"),
                             "analysis.diagnostics.task_instruction_evidence", missing)
    task_summary = _mapping(task_evidence.get("candidate"),
                            "analysis.diagnostics.task_instruction_evidence.candidate", missing)
    binding = _mapping(task_summary.get("binding"),
                       "analysis.diagnostics.task_instruction_evidence.candidate.binding", missing)
    emission = _mapping(analysis.get("emission"), "analysis.emission", missing)
    result["missing_fields"] = sorted(set(missing))
    if missing:
        return result

    problems: list[str] = []
    if plan.get("status") != "verified":
        problems.append("global plan emission is not verified")
    if graph.get("schema") != "captured_global_graph_v1" or graph.get("status") != "verified":
        problems.append("captured logical graph is not verified")
    if (task_summary.get("schema") != "task_instruction_evidence_v1"
            or task_summary.get("status") != "static_ownership_verified"
            or task_summary.get("declared_source_plan_status") != "verified"):
        problems.append("candidate source-task ownership is not statically verified")
    for field in _PIN_FIELDS:
        if not _pin(binding.get(field)):
            problems.append(f"task ownership binding lacks exact {field}")

    expected = {
        "source_sha256": plan.get("source_sha256"),
        "lowered_sha256": plan.get("candidate_lowered_sha256"),
        "command_buffer_sha256": plan.get("candidate_command_buffer_sha256"),
        "compiler_sha256": plan.get("candidate_sha256"),
        "logical_dispatch_digest": plan.get("logical_dispatch_digest"),
        "plan_digest": plan.get("plan_digest"),
    }
    for field, value in expected.items():
        if not _pin(value) or binding.get(field) != value:
            problems.append(f"plan and task ownership disagree on {field}")
    if graph.get("source_sha256") != binding.get("source_sha256"):
        problems.append("logical graph and plan disagree on source_sha256")
    if graph.get("logical_dispatch_digest") != binding.get("logical_dispatch_digest"):
        problems.append("logical graph and plan disagree on logical_dispatch_digest")
    if emission.get("candidate_command_buffer_sha256") != binding.get("command_buffer_sha256"):
        problems.append("analysis emission and plan disagree on command_buffer_sha256")
    if emission.get("candidate_lowered_sha256") != binding.get("lowered_sha256"):
        problems.append("analysis emission and plan disagree on lowered_sha256")
    for candidate in (analysis.get("candidate_sha256"), record.get("candidate_sha256")):
        if candidate is not None and candidate != binding.get("compiler_sha256"):
            problems.append("analysis/iteration candidate identity disagrees with verified plan")

    nodes = program.get("nodes")
    buffers = program.get("buffers")
    results = program.get("results")
    tasks = task_summary.get("tasks")
    if not isinstance(nodes, list):
        problems.append("logical graph nodes are unavailable")
        nodes = []
    if not isinstance(buffers, Mapping):
        problems.append("logical graph buffers are unavailable")
        buffers = {}
    if not isinstance(results, list) or any(not isinstance(item, str) for item in results):
        problems.append("logical graph results are unavailable")
        results = []
    if not isinstance(tasks, list):
        problems.append("source-task ownership rows are unavailable")
        tasks = []
    if graph.get("nodes") != len(nodes) or plan.get("source_operations") != len(nodes):
        problems.append("source operation counts do not equal the exact logical graph")
    if plan.get("tasks") != len(tasks):
        problems.append("plan task count does not equal the exact ownership rows")
    if graph.get("logical_dispatch_digest") != _digest(program):
        problems.append("logical dispatch payload does not match its bound digest")

    for name, buffer in buffers.items():
        if (not isinstance(name, str) or not name or not isinstance(buffer, Mapping)
                or buffer.get("id") != name or not isinstance(buffer.get("shape"), list)
                or any(type(extent) is not int or extent < 0 for extent in buffer.get("shape", []))
                or not isinstance(buffer.get("dtype"), str) or not buffer.get("dtype")
                or buffer.get("kind") not in {"arg", "const", "intermediate"}):
            problems.append(f"logical graph buffer {name!r} lacks an exact structural contract")

    owner: dict[int, Mapping[str, Any]] = {}
    task_ids: list[int] = []
    for task in tasks:
        if not isinstance(task, Mapping):
            problems.append("malformed source-task ownership row")
            continue
        task_index, kind, indices = (task.get("task_index"), task.get("declared_task_kind"),
                                     task.get("source_op_indices"))
        if (type(task_index) is not int or task_index < 0 or not isinstance(kind, str) or not kind
                or not isinstance(indices, list) or not indices):
            problems.append("malformed source-task ownership row")
            continue
        task_ids.append(task_index)
        if indices != sorted(set(indices)):
            problems.append(f"task {task_index} source operations are not sorted and unique")
        for index in indices:
            if type(index) is not int or not 0 <= index < len(nodes):
                problems.append(f"task {task_index} owns an absent source operation")
            elif index in owner:
                problems.append(f"source operation {index} has multiple task owners")
            else:
                owner[index] = {"task_index": task_index, "kind": kind}
    if task_ids != list(range(len(tasks))):
        problems.append("task indices do not uniquely enumerate source execution order")
    if set(owner) != set(range(len(nodes))):
        problems.append("source-task ownership does not cover every logical graph node exactly once")

    producer: dict[str, int] = {}
    uses: dict[str, list[int]] = defaultdict(list)
    for index, node in enumerate(nodes):
        if not isinstance(node, Mapping):
            problems.append(f"logical graph node {index} is malformed")
            continue
        inputs, outputs = node.get("inputs"), node.get("outputs")
        if (not isinstance(inputs, list) or not isinstance(outputs, list)
                or any(not isinstance(value, str) for value in [*inputs, *outputs])):
            problems.append(f"logical graph node {index} has malformed buffer edges")
            continue
        for value in inputs:
            if value not in buffers:
                problems.append(f"logical graph node {index} reads absent buffer {value}")
            uses[value].append(index)
        for value in outputs:
            if value not in buffers:
                problems.append(f"logical graph node {index} writes absent buffer {value}")
            if value in producer:
                problems.append(f"logical graph buffer {value} has multiple producers")
            producer[value] = index
    for index, node in enumerate(nodes):
        if not isinstance(node, Mapping):
            continue
        for value in node.get("inputs", []):
            if value in producer and producer[value] >= index:
                problems.append(f"logical graph edge for {value} is not in source order")

    result["problems"] = sorted(set(problems))
    if problems:
        return result

    relevant: dict[int, dict[str, Any]] = {}
    semantic_refusals: dict[int, str] = {}
    non_host_refusals: dict[int, dict[str, Any]] = {}
    for index, node in enumerate(nodes):
        site, reason = _semantic_site(node)
        if site is None:
            if reason is not None and node.get("kind") == "dispatch" and owner[index]["kind"] == "host":
                semantic_refusals[index] = reason
            continue
        if owner[index]["kind"] != "host":
            non_host_refusals[index] = site
            continue
        relevant[index] = site

    parent = {index: index for index in relevant}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left != right:
            parent[max(left, right)] = min(left, right)

    edges: list[dict[str, Any]] = []
    for source in sorted(relevant):
        for output in nodes[source]["outputs"]:
            consumers = uses.get(output, [])
            if len(consumers) != 1:
                continue
            destination = consumers[0]
            if (destination in relevant
                    and owner[source]["task_index"] == owner[destination]["task_index"]):
                union(source, destination)
                edges.append({"buffer": output, "producer": source, "consumer": destination})

    components: dict[int, list[int]] = defaultdict(list)
    for index in relevant:
        components[find(index)].append(index)
    candidate_members: set[int] = set()
    candidates: list[dict[str, Any]] = []
    for members in sorted((sorted(value) for value in components.values()), key=lambda row: row[0]):
        member_set = set(members)
        internal = [edge for edge in edges
                    if edge["producer"] in member_set and edge["consumer"] in member_set]
        pointwise = [index for index in members if relevant[index]["site_kind"] == "pointwise"]
        if len(members) < 2 or not pointwise or not internal:
            continue
        candidate_members.update(members)
        inputs: list[dict[str, Any]] = []
        outputs: list[dict[str, Any]] = []
        for index in members:
            for buffer in nodes[index]["inputs"]:
                source = producer.get(buffer)
                if source not in member_set:
                    inputs.append({
                        "buffer": buffer,
                        "producer_source_operation_id": source,
                        "producer_task_index": owner[source]["task_index"] if source is not None else None,
                        "buffer_kind": buffers[buffer].get("kind"),
                        "shape": buffers[buffer].get("shape"),
                        "dtype": buffers[buffer].get("dtype"),
                    })
            for buffer in nodes[index]["outputs"]:
                consumers = uses.get(buffer, [])
                outside = [consumer for consumer in consumers if consumer not in member_set]
                if outside or buffer in results or not consumers:
                    outputs.append({
                        "buffer": buffer,
                        "consumer_source_operation_ids": outside,
                        "consumer_task_indices": sorted({owner[item]["task_index"] for item in outside}),
                        "fanout": len(consumers),
                        "full_program_result": buffer in results,
                        "shape": buffers[buffer].get("shape"),
                        "dtype": buffers[buffer].get("dtype"),
                    })
        site_rows = [{"source_operation_id": index, **relevant[index]} for index in members]
        obligations = [
            "inspect and clone exact source bodies before rewriting; provenance labels are not arithmetic equivalence",
            "preserve source operation order, tensor shape/dtype, aliases, and every listed boundary value",
        ]
        if any(len(nodes[index]["inputs"]) > 1 for index in pointwise):
            obligations.append("preserve every multi-input operand and its original ordering")
        if any(row["fanout"] > 1 for row in outputs):
            obligations.append("preserve all fanout consumers and materialized lifetime")
        candidate = {
            "source_operation_ids": members,
            "pointwise_source_operation_ids": pointwise,
            "materialization_source_operation_ids": [
                index for index in members if relevant[index]["site_kind"] == "materialization"],
            "task_index": owner[members[0]]["task_index"],
            "sites": site_rows,
            "one_use_edges": sorted(internal, key=lambda row: (row["producer"], row["consumer"], row["buffer"])),
            "input_boundaries": sorted(
                inputs,
                key=lambda row: (
                    row["buffer"],
                    -1
                    if row["producer_source_operation_id"] is None
                    else row["producer_source_operation_id"],
                ),
            ),
            "output_boundaries": sorted(outputs, key=lambda row: row["buffer"]),
            "semantic_obligations": obligations,
        }
        candidate["site_binding_sha256"] = _digest(candidate)
        candidates.append(candidate)

    refusals: list[dict[str, Any]] = []
    for index, reason in sorted(semantic_refusals.items()):
        refusals.append({"source_operation_id": index, "refusal_class": "semantic",
                         "reasons": [reason]})
    for index, site in sorted(non_host_refusals.items()):
        refusals.append({
            "source_operation_id": index,
            "refusal_class": "execution_boundary",
            "semantic_site": site,
            "reasons": [f"operation is owned by non-host task kind {owner[index]['kind']!r}"],
        })
    for index in sorted(set(relevant) - candidate_members):
        refusals.append({
            "source_operation_id": index,
            "refusal_class": "fanout_or_boundary",
            "semantic_site": relevant[index],
            "reasons": _refusal_reasons(
                index, nodes[index], uses=uses, results=set(results), owner=owner,
                relevant=relevant, nodes=nodes),
        })

    source_ids = sorted(candidate_members)
    result.update(
        status="ready_for_source_site_binding" if source_ids else "no_candidate_chains",
        bindings={field: binding[field] for field in _PIN_FIELDS},
        logical_dispatch_payload_sha256=_digest(program),
        source_operation_ids=source_ids,
        chains=candidates,
        refusals=sorted(refusals, key=lambda row: row["source_operation_id"]),
    )
    result["inventory_sha256"] = _digest(result)
    return result


def _portfolio_identity(identities: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Reconstruct the public ordered portfolio identity carried by an iteration."""
    return {
        "schema": "full_model_optimization_portfolio_v1",
        "members": [dict(identity) for identity in identities],
        "selection": "multi_model_pareto_without_invented_static_cycle_total",
        "execution": "bounded_host_admitted_analysis_with_deterministic_record_order",
        "holdout_policy": "separate_post_authoring_evaluation",
        "micro_graphs": "smoke_and_mechanism_calibration_only",
    }


def inventory_portfolio_host_epilogue_sites(
        record: Mapping[str, Any], *, iteration_record_sha256: str | None = None,
) -> dict[str, Any]:
    """Inventory every exactly ordered member of one global-performance iteration.

    The first member must resolve through the protocol's ``/analysis`` alias; subsequent members
    must carry their own analysis.  The identities are order-sensitive and checked against the
    iteration's canonical portfolio hash before any source IDs are exposed.
    """
    result: dict[str, Any] = {
        "schema": PORTFOLIO_SCHEMA,
        "mechanism_id": MECHANISM_ID,
        "status": "not_ready",
        "candidate_sha256": None,
        "portfolio_sha256": None,
        "iteration_record_sha256": iteration_record_sha256,
        "iteration_payload_sha256": None,
        "ordered_member_identities": [],
        "members": [],
        "problems": [],
        "proof_scope": (
            "ordered complete-model portfolio of exact per-member logical-DAG/source-task site "
            "inventories; authoring source-site selection only"
        ),
        "not_proven": [
            "source-body arithmetic equivalence",
            "legal cross-operation fusion or materialization deletion",
            "positive emitted-work delta",
            "per-model or aggregate cycle improvement",
        ],
    }
    problems: list[str] = []
    if not isinstance(record, Mapping):
        problems.append("global performance iteration record is missing")
    elif record.get("schema") != "global_perf_iteration_v1":
        problems.append("record is not a global_perf_iteration_v1")
    if iteration_record_sha256 is not None and not _pin(iteration_record_sha256):
        problems.append("iteration_record_sha256 is not an exact SHA-256 digest")
    if problems:
        result["problems"] = problems
        result["portfolio_site_inventory_sha256"] = _digest(result)
        return result

    result["iteration_payload_sha256"] = _digest(record)
    candidate_sha256 = record.get("candidate_sha256")
    portfolio = record.get("portfolio")
    if not _pin(candidate_sha256):
        problems.append("iteration candidate identity is not an exact SHA-256 digest")
    if not isinstance(portfolio, Mapping):
        problems.append("iteration portfolio is missing")
        portfolio = {}
    if portfolio.get("schema") != "full_model_portfolio_iteration_v1":
        problems.append("iteration portfolio schema is unsupported")
    if portfolio.get("candidate_sha256") != candidate_sha256:
        problems.append("iteration and portfolio candidate identities disagree")
    if portfolio.get("full_model_simulation_allowed") is not False:
        problems.append("portfolio does not explicitly prohibit full-model simulation")
    if portfolio.get("selection") != "multi_model_pareto_without_invented_static_cycle_total":
        problems.append("portfolio selection contract is unsupported")
    members = portfolio.get("members")
    if not isinstance(members, list) or not members:
        problems.append("ordered portfolio members are unavailable")
        members = []
    if (portfolio.get("members_total") != len(members)
            or portfolio.get("members_ready") != len(members)):
        problems.append("portfolio is not a complete ready ordered member set")

    identities: list[Mapping[str, Any]] = []
    analyses: list[Mapping[str, Any]] = []
    primary_analysis = record.get("analysis")
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
                or not isinstance(identity.get("capsule"), str) or not identity.get("capsule")
                or not _pin(identity.get("capsule_sha256"))):
            problems.append(f"portfolio member {index} identity violates the ordered protocol")
        for field in ("required_lanes", "required_tiers"):
            values = identity.get(field)
            strings = (isinstance(values, list)
                       and all(isinstance(value, str) and value for value in values))
            if not strings or len(values) != len(set(values)):
                problems.append(f"portfolio member {index} has malformed {field}")
        if member.get("status") != "completed":
            problems.append(f"portfolio member {index} analysis is not completed")
        if index == 0:
            if member.get("analysis_ref") != "/analysis" or "analysis" in member:
                problems.append("primary member does not resolve uniquely through /analysis")
            analysis = primary_analysis
        else:
            if member.get("analysis_ref") is not None or not isinstance(member.get("analysis"), Mapping):
                problems.append(f"portfolio member {index} lacks one unique embedded analysis")
            analysis = member.get("analysis")
        if not isinstance(analysis, Mapping):
            problems.append(f"portfolio member {index} analysis is missing")
            analysis = {}
        expected_workload = {
            field: identity.get(field)
            for field in ("capsule", "capsule_sha256", "required_lanes", "required_tiers")
        }
        if analysis.get("workload") != expected_workload:
            problems.append(f"portfolio member {index} analysis workload differs from its identity")
        if analysis.get("candidate_sha256") != candidate_sha256:
            problems.append(f"portfolio member {index} analysis belongs to another candidate")
        identities.append(identity)
        analyses.append(analysis)

    declared_portfolio_sha256 = portfolio.get("portfolio_sha256")
    if (len(identities) != len(members) or not _pin(declared_portfolio_sha256)
            or _digest(_portfolio_identity(identities)) != declared_portfolio_sha256):
        problems.append("ordered member identities do not match the exact portfolio hash")

    result.update(
        candidate_sha256=candidate_sha256,
        portfolio_sha256=declared_portfolio_sha256,
        ordered_member_identities=[dict(identity) for identity in identities],
        problems=sorted(set(problems)),
    )
    if problems:
        result["portfolio_site_inventory_sha256"] = _digest(result)
        return result

    member_rows: list[dict[str, Any]] = []
    inventory_problems: list[str] = []
    for index, (identity, analysis) in enumerate(zip(identities, analyses, strict=True)):
        inventory = inventory_host_epilogue_sites(analysis)
        if inventory.get("status") not in {"ready_for_source_site_binding", "no_candidate_chains"}:
            inventory_problems.append(f"member {index} exact site inventory is not ready")
        member_rows.append({
            "member_index": index,
            "identity": dict(identity),
            "identity_sha256": _digest(identity),
            "analysis_location": "/analysis" if index == 0 else f"/portfolio/members/{index}/analysis",
            "analysis_sha256": _digest(analysis),
            "site_inventory_status": inventory.get("status"),
            "site_inventory_sha256": inventory.get("inventory_sha256"),
            "bindings": inventory.get("bindings"),
            "source_operation_ids": inventory.get("source_operation_ids", []),
            "chains": inventory.get("chains", []),
            "refusals": inventory.get("refusals", []),
            "inventory_problems": inventory.get("problems", []),
            "inventory_missing_fields": inventory.get("missing_fields", []),
        })
    result["members"] = member_rows
    result["problems"] = inventory_problems
    result["status"] = (
        "ready_for_portfolio_source_site_binding" if not inventory_problems else "not_ready")
    result["portfolio_site_inventory_sha256"] = _digest(result)
    return result


def _raw_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Emit an exact host-epilogue portfolio source-site inventory")
    parser.add_argument("iteration", type=Path, help="global_perf_iteration_v1 JSON record")
    parser.add_argument("output", type=Path, help="new immutable inventory JSON path")
    args = parser.parse_args(argv)
    if args.iteration.is_symlink() or not args.iteration.is_file():
        parser.error("iteration must be an existing non-symlink regular file")
    source_bytes = args.iteration.read_bytes()
    try:
        record = json.loads(source_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        parser.error(f"iteration is not valid JSON: {exc}")
    result = inventory_portfolio_host_epilogue_sites(
        record, iteration_record_sha256=_raw_sha256(source_bytes))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_bytes = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    try:
        with args.output.open("xb") as stream:
            stream.write(output_bytes)
    except FileExistsError:
        parser.error("output already exists; immutable inventories are never overwritten")
    print(json.dumps({
        "status": result["status"],
        "output": str(args.output.resolve()),
        "output_sha256": _raw_sha256(output_bytes),
        "portfolio_site_inventory_sha256": result["portfolio_site_inventory_sha256"],
        "member_counts": [{
            "member_index": row["member_index"],
            "source_operation_ids": len(row["source_operation_ids"]),
            "chains": len(row["chains"]),
        } for row in result["members"]],
    }, sort_keys=True))
    return 0 if result["status"] == "ready_for_portfolio_source_site_binding" else 2


if __name__ == "__main__":
    raise SystemExit(main())
