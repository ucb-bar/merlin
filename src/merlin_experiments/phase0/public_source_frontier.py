"""Explicit source-only publications for fast, unscored public diagnostics."""

from pathlib import Path

PROTOCOL = "merlin.public_source_frontier.v1"
SCHEMA = "merlin.public_source_frontier_publication.v1"
SCOPE = "complete sealed source arithmetic and exact references only; target effects and timing pending"


def arithmetic(entry, cap, directory, *, protocol):
    from . import connected_integer_qualification as Q

    if (
        protocol != PROTOCOL
        or cap.get("kind") != "model"
        or cap.get("label") not in {"public", "dev"}
        or cap.get("source_role") != "materialized_performance_capture"
        or "model_qualification" in cap
        or "integer_partial_sum_bound" in cap
    ):
        raise ValueError("source frontier requires an explicit unqualified complete public integer source")
    selection = entry.get("source_integer_proof")
    proof = Q.source_proof(cap, directory, selection=selection)
    return dict(
        schema=SCHEMA,
        protocol=protocol,
        source_integer_proof=selection,
        source_proof=proof,
        target_effects="PENDING",
        formal_admission=False,
        promotion_authority="NONE",
        scope=SCOPE,
    )


def verify_written(cap, directory):
    """Recompute arithmetic from the PUBLIC-verified complete original bytes."""
    import hashlib
    import json

    import numpy as np

    from merlin.common import mlir_query as mq

    from . import connected_integer_bounds as B
    from . import connected_integer_qualification as Q
    from .sealed_generation import verified_capture_failure

    record = cap.get("public_source_frontier")
    if (
        type(record) is not dict
        or set(record)
        != {
            "schema",
            "protocol",
            "source_integer_proof",
            "source_proof",
            "target_effects",
            "formal_admission",
            "promotion_authority",
            "scope",
        }
        or record["schema"] != SCHEMA
        or record["protocol"] != PROTOCOL
        or record["target_effects"] != "PENDING"
        or record["formal_admission"] is not False
        or record["promotion_authority"] != "NONE"
        or "model_qualification" in cap
        or "integer_partial_sum_bound" in cap
        or record["scope"] != SCOPE
    ):
        raise ValueError("source frontier cannot claim target qualification or change its original pending scope")
    reason = verified_capture_failure(cap, require_attestation=True)
    if reason is not None:
        raise ValueError(reason)
    directory = Path(directory)
    with np.load(directory / "source-capture-inputs.npz", allow_pickle=False) as inputs:
        intervals = [
            (int(inputs["in" + str(i)].min()), int(inputs["in" + str(i)].max())) for i in range(len(cap["inputs"]))
        ]
    selection = record["source_integer_proof"]
    options = {}
    if selection == Q.AFFINE_COPY_SELECTION:
        from .connected_integer_affine_copy import PROTOCOL as COPY

        options["affine_copy_protocol"] = COPY
    elif selection != Q.SELECTION:
        raise ValueError("source frontier has an unknown source arithmetic selection")
    derived = B.derive(mq.parse(directory / "capsule.interface.mlir"), intervals, **options)
    trace = directory / "frontend-trace.json"
    original = Q.original_use_def_semantics(json.loads(trace.read_bytes()))
    proof = record["source_proof"]
    metadata = {
        "source_sha256",
        "interface_sha256",
        "frontend_trace_sha256",
        "original_graph_sha256",
        "captured_input_sha256",
        "captured_golden_sha256",
        "source_mode",
        "selection_sha256",
        "source_scope",
    }
    if (
        type(proof) is not dict
        or any(proof.get(key) != value for key, value in derived.items())
        or set(proof) != set(derived) | metadata
        or proof.get("source_sha256") != cap["materialized_capture"]["capture_sha256"]
        or proof.get("captured_input_sha256")
        != hashlib.sha256((directory / "source-capture-inputs.npz").read_bytes()).hexdigest()
        or proof.get("captured_golden_sha256")
        != hashlib.sha256((directory / "source-capture-golden.json").read_bytes()).hexdigest()
        or proof.get("interface_sha256")
        != hashlib.sha256((directory / "capsule.interface.mlir").read_bytes()).hexdigest()
        or proof.get("frontend_trace_sha256") != hashlib.sha256(trace.read_bytes()).hexdigest()
        or proof.get("original_graph_sha256") != original.graph_sha256
        or proof.get("source_mode") != Q.native_integer.MODE
        or proof.get("selection_sha256") != cap["capture_execution_attestations"][0]["selection"]["sha256"]
        or proof.get("source_scope")
        != "original sealed integral source, exact inputs and every output; executable target still separate"
    ):
        raise ValueError("source frontier arithmetic differs from its complete original source")
    return record


def publish(entry, binding, destination, *, protocol, operator_exclusion):
    """Use original staging, source writer and every non-target finishing check."""
    from . import staging, writer

    if protocol != PROTOCOL or getattr(binding, "target", None) != "source_only":
        raise ValueError("source frontier publication requires explicit source-only diagnostic selection")

    def write(root):
        written = writer._write_capsule_source(entry, binding, root)
        return writer._finish_capsule(entry, binding, root, written, diagnostic_source_protocol=protocol)

    return staging.write_staged(
        write, destination, member=entry["cat"] + "/" + entry["name"], operator_exclusion=operator_exclusion
    )
