"""Bind application obligations to an actual admitted cohort, not its source pool.

This is a pre-compiler completeness check. It does not execute a capsule, certify
hardware, or turn declared numerical semantics into a compiler correctness claim.
All observations travel with the corpus and are independently snapshot-bound.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import yaml
from merlin.common.digest import is_sha256, sha256_bytes
from merlin.development.testgen.witness_basis import build_witness_basis


# ---- coverage_commitment -----------------------------------------------------------------------

INPUT_SCHEMA = "merlin.phase0.coverage_inputs.v1"


SCHEMA = "merlin.phase0.coverage_commitment.v2"


READINESS_SCHEMA = "merlin.phase0.corpus_readiness.v1"


INPUT_PATH = Path("_phase0/coverage-inputs.json")


_SIGNATURE_FIELDS = (
    "mlir_operation",
    "callee",
    "semantic_family",
    "ordered_operand_types",
    "ordered_result_types",
    "indexing_maps",
    "iterator_types",
    "body_operations",
    "accumulator_dtypes",
    "contraction_shape",
)


def _json(document) -> bytes:
    return json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(document) -> str:
    return sha256_bytes(_json(document))


def selected_inputs(selection, *, accounting: dict, prohibited_roles=()) -> dict:
    """Project only selected immutable documents; never resolve live references."""
    from merlin.development.testgen.spec_drift import from_selection

    requirement = next(
        (yaml.safe_load(row.content) for row in selection.source_snapshots if row.role == "conformance-spec"), None
    )
    return {
        "schema": INPUT_SCHEMA,
        "target": selection.target,
        "accounting": copy.deepcopy(accounting),
        "software_spec": selection.software_spec,
        # The authored spec compared field by field with the selected fact-derived capability.
        "spec_fact_drift": from_selection(selection, prohibited_roles=prohibited_roles),
        "capability_contract": selection.contract,
        "conformance": requirement,
        # Preserve the exact selected bytes so memory coverage can be recomputed
        # without asking a live target registry for a possibly different artifact.
        "raw_facts_utf8": selection.raw_facts.decode("utf-8") if selection.raw_facts is not None else None,
        "evidence": {"status": selection.status, "raw_facts_sha256": selection.raw_facts_sha256},
        "qualification": "selected inputs only; hardware, compilation and numerical execution remain independent",
    }


def write_inputs(corpus: Path, document: dict) -> dict:
    """Write the generated corpus-local input artifact, preserving immutable retries."""
    path = corpus / INPUT_PATH
    if path.is_symlink() or path.parent.is_symlink():
        raise ValueError("coverage inputs cannot be written through filesystem indirection")
    raw = _json(document) + b"\n"
    if path.exists() and path.read_bytes() != raw:
        raise ValueError("coverage inputs changed; generate a fresh corpus instead of replacing its commitments")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(raw)
    return {"path": INPUT_PATH.as_posix(), "sha256": sha256_bytes(raw), "schema": INPUT_SCHEMA}


def read_inputs(corpus: Path, manifest: dict | None = None) -> dict | None:
    """Read a manifest-bound sidecar; historical absence is unknown, never success."""
    if manifest is None:
        manifest_path = corpus / "MANIFEST.yaml"
        if not manifest_path.exists():
            return None
        if manifest_path.is_symlink():
            raise ValueError("corpus manifest may not be indirect")
        manifest = yaml.safe_load(manifest_path.read_bytes())
    record = (manifest or {}).get("coverage_inputs")
    if record is None:
        return None
    if not isinstance(record, dict) or record.get("path") != INPUT_PATH.as_posix():
        raise ValueError("corpus coverage input record is malformed or escapes its corpus")
    path = corpus / INPUT_PATH
    if path.is_symlink() or path.parent.is_symlink() or not path.is_file():
        raise ValueError("corpus coverage inputs are absent or indirect")
    raw = path.read_bytes()
    if sha256_bytes(raw) != record.get("sha256"):
        raise ValueError("corpus coverage input bytes differ from the derivation commitment")
    document = json.loads(raw)
    if not isinstance(document, dict) or document.get("schema") != INPUT_SCHEMA:
        raise ValueError("unsupported corpus coverage input schema")
    return document


def requires_workload_coverage(descriptor, inputs: dict | None) -> bool:
    """Either authored roster or selected source roster makes the gate mandatory."""
    return bool((getattr(descriptor, "workload_spec", None) or {}).get("applications")) or bool(
        ((inputs or {}).get("accounting") or {}).get("applications")
    )


def signature_identity(signature: dict) -> str | None:
    """Exact observed typed/structural signature, not a guessed frontend alias."""
    if not isinstance(signature, dict) or not isinstance(signature.get("mlir_operation"), str):
        return None
    if not all(isinstance(signature.get(key), list) for key in ("ordered_operand_types", "ordered_result_types")):
        return None
    return _digest({key: signature.get(key) for key in _SIGNATURE_FIELDS})


def _placement(obligation: dict) -> str | None:
    choices = []
    for lane, key in (("accelerator", "accelerator_admission"), ("host", "host_admission")):
        admission = obligation.get(key) or {}
        if admission.get("status") == "admitted" and admission.get("reviewed") is True:
            choices.append(lane)
    selected = obligation.get("placement")
    if selected in choices:
        return selected
    return choices[0] if len(choices) == 1 else None


def _correspondence_complete(value) -> bool:
    return isinstance(value, dict) and value.get("status") in {
        "complete",
        "verified",
        "identity",
        "serialization_equivalent",
    }


def _source_complete(trace: dict) -> bool:
    counts = [
        trace.get(name)
        for name in ("original_invocation_count", "quantized_invocation_count", "prepared_invocation_count")
    ]
    return (
        trace.get("status") == "complete"
        and all(type(count) is int and count >= 0 for count in counts)
        and _correspondence_complete(trace.get("raw_mlir_correspondence"))
        and _correspondence_complete(trace.get("normalization_correspondence"))
    )


def _framework_versions_complete(application: dict) -> bool:
    """Catalog and all source stages must belong to the same observed runtime.

    Never infer a missing stage version from another stage, or let a historical
    catalog without a successful observation qualify a whole-workload cohort.
    """
    record = (application.get("framework_universe") or {}).get("source_graph_version_match") or {}
    version = record.get("catalog_version")
    sources = record.get("source_graph_versions")
    return (
        record.get("status") == "matched"
        and record.get("catalog_status") == "available"
        and isinstance(version, str)
        and bool(version)
        and isinstance(sources, dict)
        and all(sources.get(stage) == version for stage in ("original", "quantized", "prepared"))
    )


def attestation_failure(application: dict) -> str | None:
    """Why this application's capture has no admitted fresh sealed execution, or None.

    The attestation must pass the issuer gate (which re-reads its bytes) AND name
    exactly this application's selected capture and materialized receipt bytes.
    """
    from merlin.development.testgen.capture_execution_attestation import (
        AttestationNotVerified,
        require_verified_execution,
    )

    attestation = application.get("capture_execution_attestation") or {}
    capture = attestation.get("capture") or {}
    materialized = application.get("capture_receipt") or {}
    try:
        require_verified_execution(attestation)
        if (
            capture.get("model_sha256") != application.get("capture_sha256")
            or capture.get("receipt_sha256") != materialized.get("receipt_sha256")
            or not is_sha256(capture.get("model_sha256"))
            or not is_sha256(capture.get("receipt_sha256"))
        ):
            raise AttestationNotVerified("verified issuer does not bind selected capture and receipt bytes")
    except AttestationNotVerified as exc:
        return str(exc)
    return None


def _graph_totality(application: dict, completeness: dict) -> tuple[dict, list[str]]:
    """Check that no normalized node or SSA use vanished from the handoff.

    The selected graph is already capture/digest-bound by operation accounting.
    This check is a closed partition of that graph, not a compiler proof.
    """
    graph = completeness.get("graph_accounting") or {}
    nodes, edges = graph.get("nodes"), graph.get("edges")
    obligations = completeness.get("operation_obligations") or []
    transfers = completeness.get("transfer_obligations") or []
    reasons = []
    if (
        completeness.get("operation_graph_status") != "complete"
        or graph.get("status") != "accounted"
        or not is_sha256(graph.get("normalized_mlir_sha256"))
        or not isinstance(nodes, list)
        or not isinstance(edges, list)
        or graph.get("n_operations") != len(nodes)
        or graph.get("n_edges") != len(edges)
        or graph.get("n_operations") != application.get("n_mlir_operations")
    ):
        reasons.append("exact normalized graph roster or capture-bound counts are absent")
        return graph, reasons
    identity = application.get("operation_graph_identity") or {}
    node_roster = [
        {key: row.get(key) for key in ("operation_id", "ordinal", "mlir_operation", "parent_operation_id")}
        for row in nodes
        if isinstance(row, dict)
    ]
    edge_roster = [
        {key: row.get(key) for key in ("id", "consumer_operation_id", "producer_operation_id", "value_id", "type")}
        for row in edges
        if isinstance(row, dict)
    ]
    if (
        identity.get("normalized_mlir_sha256") != graph["normalized_mlir_sha256"]
        or identity.get("n_operations") != len(nodes)
        or identity.get("n_edges") != len(edges)
        or identity.get("node_roster_sha256") != hashlib.sha256(_json(node_roster)).hexdigest()
        or identity.get("edge_roster_sha256") != hashlib.sha256(_json(edge_roster)).hexdigest()
    ):
        reasons.append("operation or SSA-use roster differs from the selected normalized capture graph")
    node_ids = [row.get("operation_id") for row in nodes if isinstance(row, dict)]
    ordinals = [row.get("ordinal") for row in nodes if isinstance(row, dict)]
    if (
        len(node_ids) != len(nodes)
        or len(set(node_ids)) != len(nodes)
        or len(set(ordinals)) != len(nodes)
        or set(ordinals) != set(range(len(nodes)))
        or any(not isinstance(identity, str) or not identity.startswith("mlir:") for identity in node_ids)
    ):
        reasons.append("normalized operation identities or ordinals are incomplete")
    inventory = {}
    for signature in application.get("signatures") or []:
        row = signature.get("observed_signature") or {}
        for ordinal in signature.get("ordinals") or []:
            if ordinal in inventory:
                reasons.append("inventory operation ordinal is duplicated")
            inventory[ordinal] = (row.get("mlir_operation"), row.get("disposition"))
    if set(inventory) != set(range(len(nodes))):
        reasons.append("selected operation inventory is not graph-total")
    obligation_index = {}
    for obligation in obligations:
        ids, positions = obligation.get("operation_ids") or [], obligation.get("mlir_ordinals") or []
        if len(ids) != 1 or len(positions) != 1 or ids[0] in obligation_index:
            reasons.append("operation obligation is not one-to-one with a normalized operation")
        elif isinstance(ids[0], str):
            obligation_index[ids[0]] = (obligation.get("id"), positions[0], obligation.get("role"))
    for node in nodes:
        if not isinstance(node, dict):
            reasons.append("normalized graph contains a malformed operation row")
            continue
        identity, ordinal = node.get("operation_id"), node.get("ordinal")
        if inventory.get(ordinal) != (node.get("mlir_operation"), node.get("disposition")):
            reasons.append("normalized graph node differs from the exact inventory disposition")
        noncompute = node.get("disposition") in {"structural", "component"}
        if noncompute:
            if node.get("accounting") != "non_independent_compute" or node.get("obligation_id") is not None:
                reasons.append("structural or nested operation is not explicitly accounted")
        else:
            support = node.get("disposition") == "support_required"
            expected_accounting = "support_lowering_obligation" if support else "placement_obligation"
            expected_role = "support_lowering" if support else "compute_placement"
            if node.get("accounting") != expected_accounting or obligation_index.get(identity) != (
                node.get("obligation_id"),
                ordinal,
                expected_role,
            ):
                reasons.append("normalized operation lacks its exact role-specific obligation")
    if set(obligation_index) != {
        node.get("operation_id")
        for node in nodes
        if isinstance(node, dict) and node.get("disposition") not in {"structural", "component"}
    }:
        reasons.append("operation obligation roster has missing or foreign operation identities")
    edge_ids = [edge.get("id") for edge in edges if isinstance(edge, dict)]
    if len(edge_ids) != len(edges) or len(set(edge_ids)) != len(edges):
        reasons.append("normalized SSA-use identities are incomplete or duplicated")
    transfer_index = {row.get("id"): row for row in transfers if isinstance(row, dict)}
    if len(transfer_index) != len(transfers):
        reasons.append("conditional transfer identities are duplicated")
    expected_transfers = set()
    known_nodes = set(node_ids)
    for edge in edges:
        if not isinstance(edge, dict):
            reasons.append("normalized graph contains a malformed SSA-use row")
            continue
        producer, consumer = edge.get("producer_operation_id"), edge.get("consumer_operation_id")
        if consumer not in known_nodes or (producer is not None and producer not in known_nodes):
            reasons.append("SSA use has an unknown operation endpoint")
        if not isinstance(edge.get("value_id"), str) or not isinstance(edge.get("type"), str):
            reasons.append("SSA use lacks its typed value identity")
        if edge.get("accounting") == "conditional_transfer":
            transfer = transfer_index.get(edge.get("transfer_id"))
            expected_transfers.add(edge.get("transfer_id"))
            if transfer is None or any(
                transfer.get(key) != edge.get(key)
                for key in ("id", "value_id", "type", "producer_operation_id", "consumer_operation_id")
            ):
                reasons.append("typed SSA use differs from its transfer obligation")
        elif edge.get("accounting") == "support_dependency":
            source = obligation_index.get(producer)
            destination = obligation_index.get(consumer)
            if (
                edge.get("transfer_id") is not None
                or source is None
                or destination is None
                or "support_lowering" not in {source[2], destination[2]}
            ):
                reasons.append("support dependency has no exact support-lowering endpoint")
        elif edge.get("accounting") == "block_argument_or_non_independent_endpoint" and edge.get("transfer_id") is None:
            if producer in obligation_index and consumer in obligation_index:
                reasons.append("independent operation SSA use has no conditional transfer or support dependency")
        else:
            reasons.append("SSA use has no recognized accounting disposition")
    if expected_transfers != set(transfer_index):
        reasons.append("conditional transfer roster has missing or foreign SSA uses")
    shape = graph.get("shape_domain") or {}
    if shape.get("status") != "static" or shape.get("dynamic_value_ids") != []:
        reasons.append("dynamic or unknown shape domain has no selected guard and range proof")
    return graph, sorted(set(reasons))


def _conformance_blockers(coverage: dict | None) -> list[dict]:
    if not isinstance(coverage, dict):
        return [{"component": "conformance", "reason": "admitted-cohort conformance was not measured"}]
    blockers = []

    def visit(value, prefix):
        if not isinstance(value, dict):
            return
        if value.get("uncovered"):
            blockers.append(
                {
                    "component": prefix,
                    "reason": "required capsule coverage is missing",
                    "uncovered": copy.deepcopy(value["uncovered"]),
                }
            )
        for key in ("unreadable_capsules", "capsules_unreadable"):
            if value.get(key):
                blockers.append({"component": prefix, "reason": "coverage contains unreadable capsules"})
        if value.get("status") == "not_measured":
            blockers.append({"component": prefix, "reason": "required coverage axis was not measured"})
        for key, child in value.items():
            if key != "application_demands" and isinstance(child, dict):
                visit(child, f"{prefix}.{key}")

    visit(coverage, "conformance")
    return blockers
