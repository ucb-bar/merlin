"""Typed operation obligations and conditional lane crossings, not dispatch decisions."""

from __future__ import annotations

import copy

from merlin.development.targetgen.frontend_trace import join_frontend_trace
from merlin.development.targetgen.operation_numerics import operation_numerical_contracts


def _precision(entry: dict, operation: dict | None, compute_by_operation: dict) -> dict:
    row = entry["observed_signature"]
    operands, results = row.get("ordered_operand_types"), row.get("ordered_result_types")
    unknowns = []
    if operation is None:
        unknowns.append("exact typed SSA graph unavailable")
    if not isinstance(operands, list) or not isinstance(results, list):
        unknowns.append("ordered operand/result precision observations unavailable")
    compute = compute_by_operation.get(operation["operation_id"], []) if operation is not None else []
    accumulators = row.get("accumulator_dtypes")
    if row.get("semantic_family") == "contraction" and not accumulators:
        unknowns.append("accumulator precision not observed")
    return {
        "status": "unknown" if unknowns else "resolved",
        "ordered_operand_types": copy.deepcopy(operands),
        "ordered_result_types": copy.deepcopy(results),
        "ordered_storage_types": [value["type"] for value in operation["operands"]] if operation else None,
        "compute_types": compute if operation else None,
        "accumulator_types": copy.deepcopy(accumulators),
        "result_types": [value["type"] for value in operation["results"]] if operation else None,
        "policy": "preserve observed typed SSA; no implicit widening, narrowing or cross-lane format substitution",
        "unknowns": unknowns,
    }


def _validate_graph(application: dict, graph: dict | None) -> dict | None:
    if graph is None:
        return None
    if (
        graph.get("schema") != "merlin.application_graph.v1"
        or graph.get("capture_sha256") != application["capture_sha256"]
    ):
        raise ValueError("application graph is not bound to the selected capture")
    normalization = application.get("capture_normalization") or {}
    if normalization.get("output_sha256") is not None and normalization["output_sha256"] != graph.get(
        "normalized_mlir_sha256"
    ):
        raise ValueError("application graph normalized byte identity disagrees with the inventory")
    normalized = graph.get("normalized_graph") or {}
    operations = normalized.get("operations") or []
    if normalized.get("n_operations") != application["n_operations"] or len(operations) != application["n_operations"]:
        raise ValueError("application graph operation count disagrees with the inventory")
    by_ordinal = {operation["ordinal"]: operation for operation in operations}
    if set(by_ordinal) != set(range(application["n_operations"])):
        raise ValueError("application graph operation ordinals are duplicated or incomplete")
    for row in application["signatures"]:
        for ordinal in row["ordinals"]:
            operation = by_ordinal[ordinal]
            if operation["mlir_operation"] != row["mlir_operation"]:
                raise ValueError("application graph operation identity differs from its exact inventory ordinal")
            for values, key in (
                (operation["operands"], "ordered_operand_types"),
                (operation["results"], "ordered_result_types"),
            ):
                if [{"shape": value["shape"], "dtype": value["dtype"]} for value in values] != row.get(key):
                    raise ValueError("application graph ordered typed SSA differs from its inventory signature")
    return normalized


def build_application_completeness(
    application: dict,
    entries: list[dict],
    *,
    application_graph: dict | None = None,
    frontend_trace: dict | None = None,
    software_spec: dict | None = None,
    host_capabilities: dict | None = None,
) -> dict:
    """Build inputs for coverage admission; consumers select placement separately."""
    normalized = _validate_graph(application, application_graph)
    operations = normalized["operations"] if normalized else []
    by_ordinal = {operation["ordinal"]: operation for operation in operations}
    by_id = {operation["operation_id"]: operation for operation in operations}
    compute_by_operation = {}
    for operation in operations:
        if (
            not operation["mlir_operation"].startswith(("arith.", "math."))
            or operation["mlir_operation"] == "arith.constant"
        ):
            continue
        computation = {
            "operation_id": operation["operation_id"],
            "mlir_operation": operation["mlir_operation"],
            "operand_types": [value["type"] for value in operation["operands"]],
            "result_types": [value["type"] for value in operation["results"]],
        }
        ancestor = operation
        while ancestor is not None and ancestor["mlir_operation"] not in {"builtin.module", "func.func"}:
            compute_by_operation.setdefault(ancestor["operation_id"], []).append(computation)
            ancestor = by_id.get(ancestor.get("parent_operation_id"))
    source = join_frontend_trace(frontend_trace, application_graph, capture_sha256=application["capture_sha256"])
    obligations, by_operation, compute_by_id = [], {}, {}
    for signature_index, entry in enumerate(entries):
        if entry["observed_signature"]["disposition"] in {"structural", "component"}:
            continue
        for ordinal in entry["ordinals"]:
            operation = by_ordinal.get(ordinal)
            operation_id = operation["operation_id"] if operation else None
            trace = source["normalized_operations"].get(str(ordinal)) or {}
            precision = _precision(entry, operation, compute_by_operation)
            support = entry["observed_signature"]["disposition"] == "support_required"
            if not support:
                precision["numerical_contracts"] = operation_numerical_contracts(
                    entry, software_spec, host_capabilities
                )
            accelerator = entry["accelerator_admission"]
            host = entry["host_admission"]
            choices = [
                name
                for name, decision in (("accelerator", accelerator), ("host", host))
                if decision["status"] == "admitted" and decision.get("reviewed")
            ]
            obligation = {
                "id": operation_id or f"unknown:{entry['application']}:{ordinal}",
                "operation_ids": [operation_id] if operation_id else [],
                "mlir_ordinals": [ordinal],
                "signature_index": signature_index,
                "observed_signature": copy.deepcopy(entry["observed_signature"]),
                "source_operation_ids": trace.get("original_node_ids") or [],
                "source_identity_scope": {
                    "application": entry["application"],
                    "trace_document_sha256": source.get("trace_document_sha256"),
                    "original_graph_sha256": (source.get("graphs", {}).get("original") or {}).get("sha256"),
                },
                "quantized_source_operation_ids": trace.get("quantized_node_ids") or [],
                "prepared_source_operation_ids": trace.get("prepared_node_ids") or [],
                "status": "resolved" if operation is not None and precision["status"] == "resolved" else "unknown",
                "role": "support_lowering" if support else "compute_placement",
                "placement": "unselected",
                "accelerator_admission": accelerator,
                "host_admission": host,
                "capability": {"status": "unknown", "reason": "placement is not selected"},
                "precision": precision,
                "required_placement_choices": [] if support else choices,
                "lowering_status": "unverified",
            }
            if support:
                # Typed source observations are not a lowering receipt. The compiler
                # must later bind an actual lowered program and shape/value mapping
                # to this exact operation before whole-program coverage can pass.
                source_values = [*operation["operands"], *operation["results"]] if operation else []
                obligation["support_lowering_evidence"] = {
                    "status": "not_available",
                    "source_capture_sha256": application["capture_sha256"],
                    "source_operation_id": operation_id,
                    "operand_types": precision["ordered_storage_types"],
                    "result_types": precision["result_types"],
                    "operand_shapes": [value.get("shape") for value in operation["operands"]] if operation else None,
                    "result_shapes": [value.get("shape") for value in operation["results"]] if operation else None,
                    "source_shape_status": (
                        "static"
                        if operation
                        and all(
                            isinstance(value.get("shape"), list)
                            and all(type(dimension) is int and dimension >= 0 for dimension in value["shape"])
                            for value in source_values
                        )
                        else "dynamic_or_unknown"
                    ),
                    "reason": "no selected compiler lowering and typed shape/value preservation receipt",
                }
            obligations.append(obligation)
            if operation_id:
                by_operation[operation_id] = obligation
                if not support:
                    compute_by_id[operation_id] = obligation
    transfers = []
    transfer_by_edge = {}
    for edge in normalized["edges"] if normalized else []:
        producer = compute_by_id.get(edge.get("producer_operation_id"))
        consumer = compute_by_id.get(edge.get("consumer_operation_id"))
        if producer is None or consumer is None:
            continue
        # This is a conditional candidate edge, not evidence of actual dispatch.
        transfer = {
            **copy.deepcopy(edge),
            "producer": producer["id"],
            "consumer": consumer["id"],
            "source_type": edge["type"],
            "result_type": edge["type"],
            "operand_dtype": edge["dtype"],
            "result_dtype": edge["dtype"],
            # An SSA edge carries one value: its layout, observed from its type, is both ends' layout.
            "operand_layout": edge.get("layout"),
            "result_layout": edge.get("layout"),
            "producer_signature": producer["observed_signature"],
            "consumer_signature": consumer["observed_signature"],
            "status": "conditional",
            "placement": "unselected",
            "conversion": {
                "status": "not_applicable",
                "reason": "SSA edge preserves the exact value type; casts are separate operation obligations",
            },
            "condition": "required only if selected producer and consumer lanes differ",
            "transfer_support": {"status": "unknown", "reason": "no actual lane boundary selected"},
        }
        transfers.append(transfer)
        transfer_by_edge[edge["id"]] = transfer["id"]
    # Preserve the *whole* normalized graph denominator. Structural operations
    # and nested region components have no independent dispatch decision, but
    # dropping them (or their SSA uses) would make a shortened obligation list
    # look complete to the Phase 1 handoff.
    entry_by_ordinal = {ordinal: entry for entry in entries for ordinal in entry["ordinals"]}
    graph_nodes = []
    for operation in operations:
        ordinal = operation["ordinal"]
        entry = entry_by_ordinal.get(ordinal)
        disposition = (entry or {}).get("observed_signature", {}).get("disposition", "unknown")
        graph_nodes.append(
            {
                "operation_id": operation["operation_id"],
                "ordinal": ordinal,
                "mlir_operation": operation["mlir_operation"],
                "disposition": disposition,
                "accounting": (
                    "non_independent_compute"
                    if disposition in {"structural", "component"}
                    else "support_lowering_obligation"
                    if disposition == "support_required"
                    else "placement_obligation"
                ),
                "obligation_id": by_operation.get(operation["operation_id"], {}).get("id"),
                "parent_operation_id": operation.get("parent_operation_id"),
            }
        )
    graph_edges = (
        [
            {
                "id": edge["id"],
                "consumer_operation_id": edge["consumer_operation_id"],
                "producer_operation_id": edge.get("producer_operation_id"),
                "value_id": edge["value_id"],
                "type": edge["type"],
                "accounting": "conditional_transfer"
                if edge["id"] in transfer_by_edge
                else "support_dependency"
                if edge.get("producer_operation_id") in by_operation
                and edge.get("consumer_operation_id") in by_operation
                and (
                    edge.get("producer_operation_id") not in compute_by_id
                    or edge.get("consumer_operation_id") not in compute_by_id
                )
                else "block_argument_or_non_independent_endpoint",
                "transfer_id": transfer_by_edge.get(edge["id"]),
            }
            for edge in normalized["edges"]
        ]
        if normalized
        else []
    )
    dynamic_values = []
    for operation in operations:
        for value in [*operation["operands"], *operation["results"]]:
            if any(type(dim) is not int or dim < 0 for dim in value.get("shape") or []):
                dynamic_values.append(value["value_id"])
    for block in normalized["blocks"] if normalized else []:
        for value in block["arguments"]:
            if any(type(dim) is not int or dim < 0 for dim in value.get("shape") or []):
                dynamic_values.append(value["value_id"])
    return {
        "source_trace": source,
        "operation_graph_status": "complete" if normalized else "unknown",
        "graph_accounting": {
            "status": "accounted" if normalized else "unknown",
            "normalized_mlir_sha256": normalized.get("mlir_sha256") if normalized else None,
            "n_operations": len(graph_nodes) if normalized else None,
            "n_edges": len(graph_edges) if normalized else None,
            "nodes": graph_nodes,
            "edges": graph_edges,
            "shape_domain": {
                "status": "unknown" if dynamic_values else "static" if normalized else "unknown",
                "dynamic_value_ids": sorted(set(dynamic_values)),
                "reason": "dynamic extent requires a selected guard and range proof"
                if dynamic_values
                else "all observed ranked tensor extents are static"
                if normalized
                else "exact typed graph unavailable",
            },
            "scope": (
                "every normalized MLIR operation and SSA operand use; block argument binding "
                "and support-lowering dependencies are not independent transfer endpoints"
            ),
        },
        "operation_obligations": obligations,
        "transfer_obligations": transfers,
        "dispatch_status": "not_selected",
        "lowering_status": "unverified",
    }
