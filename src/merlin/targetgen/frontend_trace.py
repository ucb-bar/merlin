"""Verify frontend stage identities against exact captured MLIR bytes and typed SSA."""

from __future__ import annotations

import hashlib
import json
from collections import Counter

from merlin.common import strict_json
from merlin.common.digest import is_sha256

# Complete frontend graphs can exceed the ordinary authority-record budget.
# This capacity changes parsing only; callers still bind the complete bytes.
MAX_FRONTEND_TRACE_BYTES = 256 * 1024 * 1024


def loads_frontend_trace(raw: bytes | str):
    """Parse one complete trace with strict duplicate/finite checks and a fixed byte bound."""
    return strict_json.loads(raw, max_bytes=MAX_FRONTEND_TRACE_BYTES)


_CALLS = {"call_function", "call_module", "call_method"}
_STRUCTURAL = {"builtin.module", "func.func", "func.return", "linalg.yield", "scf.yield"}


def _digest(document: dict) -> str:
    return hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _graph(snapshot: dict | None, stage: str, errors: list[str]) -> dict:
    unknown = {
        "status": "unknown",
        "call_count": None,
        "nodes": {},
        "calls": set(),
        "input_dtypes": {},
        "by_target": {},
    }
    if not isinstance(snapshot, dict) or snapshot.get("status") != "complete":
        errors.append(f"{stage} graph is unavailable")
        return unknown
    if snapshot.get("schema") != "m2m.frontend_graph.v1" or snapshot.get("stage") != stage:
        errors.append(f"{stage} graph schema or stage identity is invalid")
        return unknown
    if not is_sha256(snapshot.get("sha256")) or snapshot["sha256"] != _digest(
        {key: value for key, value in snapshot.items() if key != "sha256"}
    ):
        errors.append(f"{stage} graph content SHA256 disagrees")
        return unknown
    nodes = snapshot.get("nodes")
    if not isinstance(nodes, list) or any(
        not isinstance(node, dict) or not isinstance(node.get("id"), str) for node in nodes
    ):
        errors.append(f"{stage} graph node roster is invalid")
        return unknown
    indexed = {node["id"]: node for node in nodes}
    if len(indexed) != len(nodes):
        errors.append(f"{stage} graph node identities are duplicated")
        return unknown
    calls = {node["id"] for node in nodes if node.get("op") in _CALLS}
    if type(snapshot.get("call_count")) is not int or snapshot["call_count"] != len(calls):
        errors.append(f"{stage} graph call count disagrees with its node roster")
        return unknown
    values = {
        result["id"]: result
        for node in nodes
        for result in node.get("results") or []
        if isinstance(result, dict) and isinstance(result.get("id"), str)
    }
    input_dtypes: dict[str, list[str]] = {}
    for edge in snapshot.get("edges") or []:
        if (
            not isinstance(edge, dict)
            or edge.get("producer_node_id") not in indexed
            or edge.get("consumer_node_id") not in indexed
        ):
            errors.append(f"{stage} graph has an edge without known node endpoints")
            return unknown
        value_id = edge.get("producer_value_id")
        if value_id is not None:
            result = values.get(value_id)
            if result is None or result.get("dtype") != edge.get("dtype") or result.get("shape") != edge.get("shape"):
                errors.append(f"{stage} graph typed edge disagrees with its producer value")
                return unknown
        if isinstance(edge.get("dtype"), str):
            input_dtypes.setdefault(edge["consumer_node_id"], []).append(edge["dtype"])
    return {
        "status": "verified",
        "call_count": len(calls),
        "nodes": indexed,
        "calls": calls,
        "input_dtypes": {identity: sorted(dtypes) for identity, dtypes in input_dtypes.items()},
        "sha256": snapshot["sha256"],
        "runtime_versions": snapshot.get("runtime_versions"),
        "by_target": dict(sorted(Counter(indexed[identity].get("target") for identity in calls).items())),
    }


def _unresolved_call(graph: dict, identity: str) -> dict:
    node = graph["nodes"][identity]
    return {
        "node_id": identity,
        "op": node.get("op"),
        "target": node.get("target"),
        "input_dtypes": graph["input_dtypes"].get(identity, []),
        "result_dtypes": [result.get("dtype") for result in node.get("results") or []],
    }


def original_operation_semantics(trace: dict) -> tuple[str, tuple[str, ...]]:
    """Validate a selected original graph and project only its operation roster.

    Shapes, graph size and invocation counts remain source audit information;
    this projection supplies no workload-derived input or objective weights.
    """
    if not isinstance(trace, dict) or trace.get("schema") != "m2m.frontend_trace.v1":
        raise ValueError("semantic basis requires an original frontend trace")
    errors = []
    snapshots = trace.get("graphs")
    if not isinstance(snapshots, dict):
        raise ValueError("semantic basis requires a selected original frontend graph")
    snapshot = snapshots.get("original")
    try:
        graph = _graph(snapshot, "original", errors)
    except (TypeError, KeyError) as exc:
        raise ValueError("original semantic graph is malformed") from exc
    if errors or graph["status"] != "verified":
        raise ValueError("original semantic graph is invalid: " + "; ".join(errors))
    operations = tuple(sorted(graph["by_target"]))
    if not operations or any(not isinstance(op, str) or not op for op in operations):
        raise ValueError("original semantic graph requires concrete operation targets")
    if snapshot.get("by_target") is not None and snapshot["by_target"] != graph["by_target"]:
        raise ValueError("original semantic graph operation roster disagrees with its call nodes")
    return graph["sha256"], operations


def _recorded_unresolved_status(transition: dict, field: str, computed: set[str]) -> str:
    recorded = transition.get(field)
    if recorded is None:
        return "not_reported"
    if not isinstance(recorded, list) or any(not isinstance(identity, str) for identity in recorded):
        return "mismatch"
    return "matched" if len(recorded) == len(set(recorded)) and set(recorded) == computed else "mismatch"


def join_frontend_trace(trace: dict | None, application_graph: dict | None, *, capture_sha256: str) -> dict:
    """Return exact stage counts and digest-bound per-operation correspondence.

    Program names, FQNs, operator-family labels and matching row counts never
    establish identity. Missing original/quantized snapshots remain unknown.
    """
    base = {
        "status": "unknown",
        "counting_unit": "static_captured_call_sites",
        "original_invocation_count": None,
        "quantized_invocation_count": None,
        "prepared_invocation_count": None,
        "graphs": {},
        "transition_obligations": [],
        "normalized_operations": {},
        "raw_mlir_correspondence": {"status": "unknown"},
        "prepared_lowering_obligations": {"status": "unknown", "unresolved_calls": []},
        "normalization_correspondence": {"status": "unknown"},
    }
    if trace is None:
        return {**base, "errors": ["no selected original/quantized frontend trace"]}
    if not isinstance(trace, dict) or trace.get("schema") != "m2m.frontend_trace.v1":
        return {**base, "status": "invalid", "errors": ["unsupported frontend trace schema"]}
    errors = []
    snapshots = trace.get("graphs") or {}
    graphs = {stage: _graph(snapshots.get(stage), stage, errors) for stage in ("original", "quantized", "prepared")}
    stages_valid = all(graph["status"] == "verified" for graph in graphs.values())
    for stage, graph in graphs.items():
        base[f"{stage}_invocation_count"] = graph["call_count"]
        base["graphs"][stage] = {
            key: value for key, value in graph.items() if key not in {"nodes", "calls", "input_dtypes"}
        }
    relations = trace.get("transformations")
    seen_transitions = set()
    if not isinstance(relations, list):
        errors.append("frontend transformation correspondence is absent")
        relations = []
    for transition in relations:
        source, destination = transition.get("from_stage"), transition.get("to_stage")
        if (source, destination) not in {("original", "quantized"), ("quantized", "prepared")}:
            errors.append("unknown frontend stage transition")
            continue
        seen_transitions.add((source, destination))
        consumed, produced = set(), set()
        for relation in transition.get("relations") or []:
            sources, destinations = relation.get("source_ids") or [], relation.get("destination_ids") or []
            if not set(sources) <= set(graphs[source]["nodes"]) or not set(destinations) <= set(
                graphs[destination]["nodes"]
            ):
                errors.append(f"{source} -> {destination} references unknown source identities")
            consumed.update(sources)
            produced.update(destinations)
        source_calls = graphs[source]["calls"] - consumed
        destination_calls = graphs[destination]["calls"] - produced
        graph_verified = graphs[source]["status"] == graphs[destination]["status"] == "verified"
        reported = [
            _recorded_unresolved_status(transition, field, computed)
            for field, computed in (
                ("unresolved_source_ids", source_calls),
                ("unresolved_destination_ids", destination_calls),
            )
        ]
        producer_status = (
            "not_checked"
            if not graph_verified
            else "mismatch"
            if "mismatch" in reported
            else "not_reported"
            if "not_reported" in reported
            else "matched"
        )
        if producer_status == "mismatch":
            errors.append(f"{source} -> {destination} producer unresolved call IDs disagree with relation roster")
        base["transition_obligations"].append(
            {
                "from_stage": source,
                "to_stage": destination,
                "status": "unknown_graph"
                if not graph_verified
                else "complete"
                if transition.get("status") == "complete" and not source_calls and not destination_calls
                else "unresolved",
                "producer_unresolved_ids_status": producer_status,
                "unresolved_source_calls": [
                    _unresolved_call(graphs[source], identity) for identity in sorted(source_calls)
                ]
                if graph_verified
                else [],
                "unresolved_destination_calls": [
                    _unresolved_call(graphs[destination], identity) for identity in sorted(destination_calls)
                ]
                if graph_verified
                else [],
                "scope": "uncovered selected call sites; neither elimination nor semantic equivalence is inferred",
            }
        )
        if (
            transition.get("status") != "complete"
            or not graphs[source]["calls"] <= consumed
            or not graphs[destination]["calls"] <= produced
        ):
            errors.append(f"{source} -> {destination} call-site correspondence is incomplete")
    if seen_transitions != {("original", "quantized"), ("quantized", "prepared")}:
        errors.append("original -> quantized -> prepared transition roster is incomplete")
    mlir = trace.get("mlir") or {}
    if not isinstance(application_graph, dict) or application_graph.get("schema") != "merlin.application_graph.v1":
        errors.append("selected typed MLIR graph is unavailable")
    elif (
        application_graph.get("capture_sha256") != capture_sha256
        or mlir.get("sha256") != capture_sha256
        or mlir.get("bytes") != application_graph.get("capture_bytes")
    ):
        errors.append("frontend trace does not correspond to the exact selected capture bytes")
    else:
        raw = application_graph.get("capture_graph") or {}
        raw_operations, recorded = raw.get("operations") or [], mlir.get("operations")
        prepared_nodes = graphs["prepared"]["nodes"]
        origin_nodes = {**graphs["original"]["nodes"], **graphs["quantized"]["nodes"]}
        roster_valid = isinstance(recorded, list) and len(recorded) == len(raw_operations)
        if roster_valid:
            for ordinal, (observed, record) in enumerate(zip(raw_operations, recorded, strict=True)):
                expected_role = observed.get("trace_role") or (
                    "structural" if observed["mlir_operation"] in _STRUCTURAL else "unresolved"
                )
                if (
                    record.get("ordinal") != ordinal
                    or observed.get("ordinal") != ordinal
                    or record.get("operation") != observed["mlir_operation"]
                    or record.get("operand_types") != [value["type"] for value in observed["operands"]]
                    or record.get("result_types") != [value["type"] for value in observed["results"]]
                    or record.get("source_node_ids") != observed.get("source_node_ids")
                    or record.get("origin_node_ids") != observed.get("origin_node_ids")
                    or record.get("role") != expected_role
                    or not set(record.get("source_node_ids") or []) <= set(prepared_nodes)
                    or not set(record.get("origin_node_ids") or []) <= set(origin_nodes)
                ):
                    roster_valid = False
                    break
        if roster_valid:
            base["raw_mlir_correspondence"] = {
                "status": "verified",
                "sha256": capture_sha256,
                "bytes": mlir["bytes"],
                "n_operations": len(recorded),
            }
            mapped = {
                identity: [record["ordinal"] for record in recorded if identity in record["source_node_ids"]]
                for identity in graphs["prepared"]["calls"]
            }
            correspondence = mlir.get("source_correspondence")
            correspondence_valid = True
            unresolved = set()
            if not isinstance(correspondence, list) or len(correspondence) != len(mapped):
                correspondence_valid = False
            else:
                seen = set()
                for receipt in correspondence:
                    if not isinstance(receipt, dict):
                        correspondence_valid = False
                        break
                    identity = receipt.get("node_id")
                    if identity not in mapped or identity in seen or receipt.get("mlir_ordinals") != mapped[identity]:
                        correspondence_valid = False
                        break
                    seen.add(identity)
                    # An identity decomposition can emit a cast while retaining
                    # its explicit alias disposition. Its exact operation roster
                    # is still checked above; alias targets and result indices
                    # need the same validation with or without emitted operations.
                    if receipt.get("status") == "alias":
                        if (
                            receipt.get("alias_of_node_id") not in prepared_nodes
                            or type(receipt.get("result_index")) is not int
                            or receipt["result_index"] < 0
                        ):
                            correspondence_valid = False
                            break
                    elif mapped[identity]:
                        if receipt.get("status") != "lowered":
                            correspondence_valid = False
                            break
                    elif receipt.get("status") == "eliminated":
                        if not receipt.get("reason"):
                            correspondence_valid = False
                            break
                    elif receipt.get("status") == "unresolved":
                        unresolved.add(identity)
                    else:
                        correspondence_valid = False
                        break
            base["prepared_lowering_obligations"] = {
                "status": "unknown" if not correspondence_valid else "unresolved" if unresolved else "verified",
                "unresolved_calls": [_unresolved_call(graphs["prepared"], identity) for identity in sorted(unresolved)]
                if correspondence_valid
                else [],
            }
            if not correspondence_valid:
                errors.append("prepared call-site lowering receipt disagrees with exact MLIR source identities")
            elif unresolved:
                errors.append("prepared call-site lowering correspondence is incomplete")
            normalized = application_graph.get("normalization_correspondence") or {}
            normalization_valid = normalized.get("status") in {"identity", "serialization_equivalent"}
            base["normalization_correspondence"] = {
                **normalized,
                "status": "verified" if normalization_valid else "unknown",
            }
            if normalization_valid:
                base["normalized_operations"] = {
                    str(record["ordinal"]): {
                        "prepared_node_ids": record["source_node_ids"],
                        "origin_node_ids": record["origin_node_ids"],
                        "original_node_ids": [
                            identity
                            for identity in record["origin_node_ids"]
                            if identity in graphs["original"]["nodes"]
                        ],
                        "quantized_node_ids": [
                            identity
                            for identity in record["origin_node_ids"]
                            if identity in graphs["quantized"]["nodes"]
                        ],
                    }
                    for record in recorded
                }
            else:
                errors.append("normalization has no exact per-operation source correspondence")
        else:
            errors.append("frontend trace MLIR roster differs from exact parsed SSA/type/source identities")
    if trace.get("status") != "complete" or trace.get("blockers"):
        errors.extend(trace.get("blockers") or ["producer reports incomplete frontend trace"])
    base["status"] = "complete" if not errors and stages_valid else "partial"
    return {
        **base,
        "errors": sorted(set(errors)),
        "trace_document_sha256": _digest(trace),
        "scope": "exact selected static frontend call sites; not dynamic runtime invocation counts",
    }
