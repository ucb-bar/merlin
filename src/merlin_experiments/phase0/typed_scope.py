"""Exact typed source instances behind derived Phase 0 adjacency requirements.

These records preserve what the selected capture actually says. They do not
assign host/device placement, assert fusion, or qualify a Phase 2 performance
cohort. The raw ``scope.required`` axis remains a separate source-demand census.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


def typed_required_instances(scope: dict, inventory: dict) -> dict:
    """Bind every required chain occurrence to raw MLIR operation IDs and SSA types."""
    from merlin.common import mlir_query as mq
    from merlin.targetgen.model_coverage import region_ops
    from merlin.targetgen.scope_census import chains

    required = scope.get("required") or []
    wanted = {row["signature"]: row["occurrences"] for row in required}
    if len(wanted) != len(required) or any(type(count) is not int or count < 1 for count in wanted.values()):
        raise ValueError("scope requirement has duplicate or invalid occurrence counts")
    instances = []
    observed: Counter[str] = Counter()
    for label, application in sorted((inventory.get("applications") or {}).items()):
        path = Path(application["capture_source_path"])
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"selected scope capture is missing or symlinked: {label}")
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != application["capture_sha256"]:
            raise ValueError(f"capture bytes changed while deriving typed scope: {label}")
        graph = (application.get("operation_graph") or {}).get("capture_graph") or {}
        if graph.get("mlir_sha256") != digest:
            raise ValueError(f"source graph differs from selected capture bytes: {label}")
        module = mq.parse(raw.decode("utf-8"))
        operations = list(mq.walk(module))
        graph_rows = graph.get("operations") or []
        if len(operations) != len(graph_rows):
            raise ValueError(f"source operation graph membership changed: {label}")
        ordinals = {id(op): index for index, op in enumerate(operations)}
        regions = region_ops(module)
        for chain in chains(module, max_length=64):
            if chain.signature not in wanted:
                continue
            observed[chain.signature] += 1
            selected = []
            for region_index, family in zip(chain.indices, chain.families, strict=True):
                op = regions[region_index]
                ordinal = ordinals[id(op)]
                source = graph_rows[ordinal]
                operation_id = f"mlir:{digest}:{ordinal}"
                if (
                    source.get("ordinal") != ordinal
                    or source.get("operation_id") != operation_id
                    or source.get("mlir_operation") != mq.op_name(op)
                ):
                    raise ValueError(f"source operation identity changed: {label}/{ordinal}")
                provenance = source.get("provenance") or {}
                selected.append(
                    {
                        "region_index": region_index,
                        "operation_id": operation_id,
                        "ordinal": ordinal,
                        "mlir_operation": source["mlir_operation"],
                        "source_op": provenance.get("prov.op") or source["mlir_operation"],
                        "semantic_family": family,
                        "source_node_ids": source.get("source_node_ids") or [],
                        "origin_node_ids": source.get("origin_node_ids") or [],
                        "operands": source.get("operands") or [],
                        "results": source.get("results") or [],
                    }
                )
            edges = []
            for producer, consumer in zip(selected, selected[1:]):
                crossing = [
                    operand for operand in consumer["operands"]
                    if operand.get("producer_operation_id") == producer["operation_id"]
                ]
                if not crossing:
                    raise ValueError(f"source chain lost its typed SSA edge: {label}/{chain.signature}")
                edges.extend(
                    {
                        "producer_operation_id": producer["operation_id"],
                        "consumer_operation_id": consumer["operation_id"],
                        "value_id": operand["value_id"],
                        "operand_index": operand["index"],
                        "type": operand["type"],
                        "shape": operand["shape"],
                        "dtype": operand["dtype"],
                    }
                    for operand in crossing
                )
            operation_ids = [row["operation_id"] for row in selected]
            identity = json.dumps({"capture_sha256": digest, "operation_ids": operation_ids}, sort_keys=True)
            instances.append(
                {
                    "instance_id": hashlib.sha256(identity.encode()).hexdigest(),
                    "application": label,
                    "capture_sha256": digest,
                    "capture_receipt_sha256": (application.get("capture_receipt") or {}).get("receipt_sha256"),
                    "signature": chain.signature,
                    "operation_ids": operation_ids,
                    "regions": selected,
                    "edges": edges,
                }
            )
    if dict(observed) != wanted:
        raise ValueError(f"typed scope occurrences differ from raw requirement: {dict(observed)} != {wanted}")
    return {
        "schema": "merlin.phase0.typed_scope_instances.v1",
        "status": "typed_source_only",
        "instances": instances,
        "qualification": "exact source adjacency and types; no host/device placement, compiler lowering or execution",
    }
