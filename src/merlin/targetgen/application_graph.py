"""Exact typed SSA capture graphs, kept separate from legacy demand-inventory bytes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

SCHEMA = "merlin.application_graph.v1"


def _program_graph(module, digest: str) -> dict:
    from merlin.common import mlir_query as mq

    operations = list(mq.walk(module))
    operation_ids = {id(op): f"mlir:{digest}:{ordinal}" for ordinal, op in enumerate(operations)}
    values, blocks, block_ids = {}, [], {}

    from merlin.targetgen.access_observations import type_layout

    def typed(value) -> dict:
        shape, dtype = mq.type_shape_dtype(value.type)
        return {"type": str(value.type), "shape": shape, "dtype": dtype, "layout": type_layout(value.type)}

    for op in operations:
        operation_id = operation_ids[id(op)]
        for index, value in enumerate(op.results):
            values[value] = {
                "value_id": f"{operation_id}:result:{index}",
                "producer_operation_id": operation_id,
                "result_index": index,
                **typed(value),
            }
        for region_index, region in enumerate(op.regions):
            for block_index, block in enumerate(region.blocks):
                block_id = f"{operation_id}:region:{region_index}:block:{block_index}"
                block_ids[id(block)] = block_id
                arguments = []
                for argument_index, value in enumerate(block.args):
                    identity = {
                        "value_id": f"{block_id}:argument:{argument_index}",
                        "producer_operation_id": None,
                        "block_id": block_id,
                        "argument_index": argument_index,
                        **typed(value),
                    }
                    values[value] = identity
                    arguments.append(identity)
                blocks.append(
                    {
                        "block_id": block_id,
                        "owner_operation_id": operation_id,
                        "region_index": region_index,
                        "block_index": block_index,
                        "arguments": arguments,
                    }
                )
    rows, edges = [], []
    for ordinal, op in enumerate(operations):
        operation_id = operation_ids[id(op)]
        operands = []
        for operand_index, value in enumerate(op.operands):
            if value not in values:
                raise ValueError(f"capture graph {operation_id} operand {operand_index} has no SSA producer")
            operand = {"index": operand_index, **values[value]}
            operands.append(operand)
            edges.append(
                {
                    "id": f"{operation_id}:operand:{operand_index}",
                    "consumer_operation_id": operation_id,
                    "operand_index": operand_index,
                    **values[value],
                }
            )
        parent = op.parent_op()

        def source_ids(key: str) -> list[str]:
            from xdsl.dialects.builtin import ArrayAttr, StringAttr

            attribute = op.attributes.get(key)
            if attribute is None:
                return []
            if not isinstance(attribute, ArrayAttr) or any(not isinstance(item, StringAttr) for item in attribute):
                raise ValueError(f"capture graph {operation_id} has malformed source identity metadata")
            return [item.data for item in attribute]

        rows.append(
            {
                "operation_id": operation_id,
                "ordinal": ordinal,
                "mlir_operation": mq.op_name(op),
                "parent_operation_id": operation_ids.get(id(parent)) if parent is not None else None,
                "attributes": {key: str(value) for key, value in sorted(op.attributes.items())},
                "properties": {key: str(value) for key, value in sorted(op.properties.items())},
                "successor_block_ids": [block_ids[id(block)] for block in op.successors],
                "source_node_ids": source_ids("prov.source_node_ids"),
                "origin_node_ids": source_ids("prov.origin_node_ids"),
                "trace_role": mq.attr_str(op, "prov.trace_role"),
                "operands": operands,
                "results": [{"index": index, **values[value]} for index, value in enumerate(op.results)],
                "provenance": dict(sorted(mq.provenance(op).items())),
            }
        )
    return {
        "mlir_sha256": digest,
        "n_operations": len(rows),
        "operations": rows,
        "blocks": blocks,
        "edges": edges,
        "scope": "exact parsed SSA uses; block-argument binding is not an inferred runtime transfer",
    }


def application_graph_inventory(path: str | Path) -> dict:
    """Observe raw and canonically normalized capture bytes without touching v1 inventory.

    Ordinals only have meaning within their digest-bound program. A normalizer
    changing the program does not establish a raw-to-normalized operation mapping.
    """
    return application_graph_inventory_from_bytes(Path(path).read_bytes())


def application_graph_inventory_from_bytes(raw: bytes) -> dict:
    """:func:`application_graph_inventory` of exactly these capture bytes: a function of ``raw`` alone."""
    from merlin.common import mlir_query as mq
    from merlin.frontends.capture_normalization import normalize_capture_mlir

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
