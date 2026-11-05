"""Public, target-neutral authoring checks for ``mixed_program_plan_v1``.

The inventory counts every direct operation in the unique model entry block except
``func.return``. In particular constants, empty tensors, splats and fills have
indices: the payload-only linalg reader deliberately omits those operations.
These checks establish syntax and source/plan consistency, not device execution,
semantic equivalence, eligibility, or numerical correctness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import threading
from .linalg_iface import make_linalg_context
IR_LOCK = threading.Lock()

OUTPUT_WRITER_OWNERSHIP_FINDING = "output writer lacks exact source-result task ownership"


def _attribute_text(op, name: str) -> str | None:
    from xdsl.dialects.builtin import StringAttr

    for table in (op.attributes, getattr(op, "properties", {}) or {}):
        value = table.get(name)
        if isinstance(value, StringAttr):
            return value.data
    return None


def _parsed_source(source: bytes):
    from xdsl.parser import Parser

    text = source.decode("utf-8")
    with IR_LOCK:
        module = Parser(make_linalg_context(), text).parse_module()
    functions = [op for op in module.body.block.ops if op.name == "func.func" and op.body.blocks]
    if len(functions) != 1 or len(functions[0].body.blocks) != 1:
        raise ValueError("source requires one unique single-block function definition")
    entry = _attribute_text(functions[0], "sym_name")
    if not entry:
        raise ValueError("source function has no symbol name")
    block = functions[0].body.blocks[0]
    operations = [op for op in block.ops if op.name != "func.return"]
    returns = [op for op in block.ops if op.name == "func.return"]
    if len(returns) != 1 or block.last_op is not returns[0]:
        raise ValueError("source requires one terminating func.return")
    # Parsing alone accepts a return whose count or types disagree with the
    # declared function result ABI. Verify that exact boundary before exposing
    # its ordered value identities; this grants no body/effect equivalence.
    returns[0].verify()
    return entry, block, operations, returns[0]


def _tensor_type(value) -> dict[str, Any]:
    from xdsl.dialects.builtin import TensorType

    typ = value.type
    if isinstance(typ, TensorType):
        return {"shape": [int(d) for d in typ.get_shape()], "dtype": str(typ.get_element_type())}
    return {"type": str(typ)}


def source_operation_inventory(source: bytes | str) -> dict[str, Any]:
    """Inventory exact UTF-8 source bytes, including direct initialization operations."""
    raw = source.encode("utf-8") if isinstance(source, str) else source
    entry, block, operations, returned = _parsed_source(raw)
    owners = {
        value: {"op_index": i, "result_index": j}
        for i, op in enumerate(operations)
        for j, value in enumerate(op.results)
    }
    owners.update({value: {"arg_index": i} for i, value in enumerate(block.args)})

    def value_record(value):
        return {**_tensor_type(value), "source": owners.get(value)}

    rows = []
    for i, op in enumerate(operations):
        rows.append(
            {
                "source_op_index": i,
                "operation": op.name
                if op.name != "builtin.unregistered"
                else (_attribute_text(op, "op_name__") or op.name),
                "region_id": _attribute_text(op, "prov.region_id"),
                "operands": [value_record(value) for value in op.operands],
                "results": [_tensor_type(value) for value in op.results],
            }
        )
    return {
        "schema": "mixed_source_operation_inventory_v1",
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "entry": entry,
        "source_op_count": len(rows),
        "arguments": [_tensor_type(value) for value in block.args],
        "operations": rows,
        "returns": [value_record(value) for value in returned.operands],
    }

