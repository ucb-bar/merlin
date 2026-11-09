"""Typed source SSA identities and conservative layout observations.

These observations provide no memory placement, execution or effect authority.
"""

from __future__ import annotations


def type_layout(value_type) -> str | None:
    """``row_major_contiguous`` for an unencoded tensor or identity-layout memref, else ``None``.

    The same rule judges one SSA value (a transfer edge) and every operand of an operation, so a
    transfer and the operations on either side of it are described in one vocabulary.
    """
    kind = type(value_type).__name__
    if kind == "TensorType":
        encoding = getattr(value_type, "encoding", None)
        return None if encoding is not None and type(encoding).__name__ != "NoneAttr" else "row_major_contiguous"
    if kind == "MemRefType":
        layout = getattr(value_type, "layout", None)
        return None if layout is not None and type(layout).__name__ != "NoneAttr" else "row_major_contiguous"
    return None



def _program_graph(module, digest: str) -> dict:
    from merlin.common import mlir_query as mq

    operations = list(mq.walk(module))
    operation_ids = {id(op): f"mlir:{digest}:{ordinal}" for ordinal, op in enumerate(operations)}
    values, blocks, block_ids = {}, [], {}

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

