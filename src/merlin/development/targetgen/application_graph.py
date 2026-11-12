"""Exact typed SSA capture graphs, kept separate from legacy demand-inventory bytes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from merlin.compiler.ir.graph import _program_graph

SCHEMA = "merlin.application_graph.v1"


def application_graph_inventory(path: str | Path) -> dict:
    """Observe raw and canonically normalized capture bytes without touching v1 inventory.

    Ordinals only have meaning within their digest-bound program. A normalizer
    changing the program does not establish a raw-to-normalized operation mapping.
    """
    from merlin.common import mlir_query as mq
    from merlin.compiler.frontend.capture_normalization import normalize_capture_mlir

    raw = Path(path).read_bytes()
    text = raw.decode("utf-8")
    raw_digest = hashlib.sha256(raw).hexdigest()
    normalized, normalization = normalize_capture_mlir(text)
    normalized_bytes = normalized.encode("utf-8")
    normalized_digest = hashlib.sha256(normalized_bytes).hexdigest()
    capture_graph = _program_graph(mq.parse(text), raw_digest)
    normalized_graph = _program_graph(mq.parse(normalized), normalized_digest)

    def comparable(graph: dict) -> str:
        identities = {
            "operation_id",
            "parent_operation_id",
            "consumer_operation_id",
            "producer_operation_id",
            "owner_operation_id",
            "block_id",
            "value_id",
            "id",
            "successor_block_ids",
        }

        def normalize(value, key=None):
            if isinstance(value, dict):
                return {field: normalize(member, field) for field, member in value.items() if field != "mlir_sha256"}
            if isinstance(value, list):
                return [normalize(member, key) for member in value]
            if isinstance(value, str) and key in identities:
                return value.replace(f"mlir:{graph['mlir_sha256']}:", "mlir:program:")
            return value

        return json.dumps(normalize(graph), sort_keys=True)

    correspondence = comparable(capture_graph) == comparable(normalized_graph)
    return {
        "schema": SCHEMA,
        "capture_sha256": raw_digest,
        "capture_bytes": len(raw),
        "normalized_mlir_sha256": normalized_digest,
        "normalized_mlir_bytes": len(normalized_bytes),
        "normalization": normalization,
        "capture_graph": capture_graph,
        "normalized_graph": normalized_graph,
        "normalization_correspondence": {
            "status": "identity"
            if raw_digest == normalized_digest
            else "serialization_equivalent"
            if correspondence
            else "unknown",
            "reason": "byte-identical programs"
            if raw_digest == normalized_digest
            else "exact ordered SSA, operation attributes/properties and control-flow graph agree"
            if correspondence
            else "normalization receipt binds program bytes, not per-operation rewrite correspondence",
        },
    }
