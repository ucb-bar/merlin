"""Typed source identities for non-function mathematical input representations.

This is structural provenance, never an equivalence proof. The trusted judge
derives its input from the original selected capsule; submitted inventories do
not confer ownership. Operation indices are expression order for numerical JSON
and non-declaration operation order for the module-level interface grammar.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def _inventory(raw, arguments, operations, returns, entry):
    return {"schema": "mixed_source_operation_inventory_v1",
            "source_sha256": hashlib.sha256(raw).hexdigest(), "entry": entry,
            "source_op_count": len(operations), "arguments": arguments,
            "operations": operations, "returns": returns}


def numerical_inventory(raw: bytes) -> dict[str, Any]:
    """Number exact ``numerical_program.v1`` bytes, preserving typed SSA edges."""
    def unique_object(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate numerical-program object key")
            value[key] = item
        return value

    doc = json.loads(raw, object_pairs_hook=unique_object,
                     parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite numerical-program value")))
    if (not isinstance(doc, dict) or set(doc) != {"schema", "inputs", "expressions", "outputs"}
            or doc["schema"] != "numerical_program.v1"
            or any(not isinstance(doc[k], list) for k in ("inputs", "expressions", "outputs"))):
        raise ValueError("expected complete numerical_program.v1 document")

    def typ(row):
        shape, dtype = row.get("shape"), row.get("dtype")
        if (not isinstance(shape, list) or not shape or any(type(d) is not int or d <= 0 for d in shape)
                or not isinstance(dtype, str) or dtype not in {"i8", "i32"}):
            raise ValueError("invalid numerical-program tensor type")
        return {"shape": shape, "dtype": dtype}

    values, names, arguments, operations = {}, set(), [], []

    def bind(ident, record):
        if not isinstance(ident, str) or not ident or ident in values:
            raise ValueError("duplicate or invalid numerical-program value identity")
        values[ident] = record

    for i, row in enumerate(doc["inputs"]):
        if not isinstance(row, dict) or set(row) != {"id", "name", "shape", "dtype"}:
            raise ValueError("invalid numerical-program input record")
        name = row["name"]
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("duplicate or invalid numerical-program input name")
        names.add(name)
        value_type = typ(row)
        arguments.append(value_type)
        bind(row["id"], {**value_type, "source": {"arg_index": i}})
    arities = {"matmul": 2, "conv2d": 2, "add_bias": 2, "scaled_add": 2,
               "scale_round": 1, "round_shift_right": 1, "relu": 1, "max_pool": 1, "cast": 1}
    for i, row in enumerate(doc["expressions"]):
        if (not isinstance(row, dict) or set(row) - {"id", "operator", "operands", "shape", "dtype", "parameters"}
                or not {"id", "operator", "operands", "shape", "dtype"} <= set(row)):
            raise ValueError("invalid numerical-program expression record")
        op, args = row["operator"], row["operands"]
        if (not isinstance(op, str) or op not in arities or not isinstance(args, list)
                or len(args) != arities[op] or any(not isinstance(v, str) or v not in values for v in args)
                or ("parameters" in row and not isinstance(row["parameters"], dict))):
            raise ValueError("unsupported expression or undefined numerical-program operand")
        value_type = typ(row)
        operations.append({"source_op_index": i, "operation": "numerical_program." + op,
                           "region_id": None, "operands": [values[v] for v in args],
                           "results": [value_type]})
        bind(row["id"], {**value_type, "source": {"op_index": i, "result_index": 0}})
    returns, output_names = [], set()
    for row in doc["outputs"]:
        if not isinstance(row, dict) or set(row) != {"name", "value", "shape", "dtype"}:
            raise ValueError("invalid numerical-program output record")
        name, value = row["name"], row["value"]
        if (not isinstance(name, str) or not name or name in output_names
                or not isinstance(value, str) or value not in values or typ(row) != {k: values[value][k] for k in ("shape", "dtype")}):
            raise ValueError("invalid numerical-program output identity or type")
        output_names.add(name)
        returns.append(values[value])
    if not returns:
        raise ValueError("numerical program requires an output")
    return _inventory(raw, arguments, operations, returns, "numerical_program")


def interface_inventory(raw: bytes) -> dict[str, Any]:
    """Use the interface owner grammar, then typed generic SSA parsing.

    Tensor declarations form the ordered entry arguments; all remaining source
    operations, including lifetime operations, have indices. Published tensor
    results form the ordered returns. Opaque device-state results cannot grant
    shaped materialized storage merely by being present in this inventory.
    """
    from merlin.common.ir_lock import IR_LOCK
    from merlin.targetgen.contract import interface_emit as I
    from merlin.targetgen.contract.linalg_iface import make_linalg_context
    from xdsl.dialects.builtin import TensorType
    from xdsl.parser import Parser

    text = raw.decode("utf-8")
    I.parse_interface_mlir(text)  # The owning grammar must accept every opcode.
    with IR_LOCK:
        module = Parser(make_linalg_context(), I.to_generic_form(text)).parse_module()
    from .plan import _attribute_text, _tensor_type

    all_ops = list(module.body.block.ops)
    names = [_attribute_text(op, "op_name__") or op.name for op in all_ops]
    if not all_ops or any(not name.startswith("merlin_iface.") for name in names):
        raise ValueError("interface inventory requires only owner-defined module-level operations")
    declarations = [op for op, name in zip(all_ops, names, strict=True) if name == "merlin_iface.tensor"]
    if any(len(op.results) != 1 or not isinstance(op.results[0].type, TensorType) for op in declarations):
        raise ValueError("interface entry declarations require one typed tensor result")
    operations = [op for op in all_ops if op not in declarations]
    owners = {op.results[0]: {"arg_index": i} for i, op in enumerate(declarations)}
    owners.update({value: {"op_index": i, "result_index": j}
                   for i, op in enumerate(operations) for j, value in enumerate(op.results)})
    seen, rows, returns = {op.results[0] for op in declarations}, [], []
    for i, op in enumerate(operations):
        name = _attribute_text(op, "op_name__") or op.name
        if any(value not in seen for value in op.operands):
            raise ValueError("interface source operand is not defined before use")
        rows.append({"source_op_index": i, "operation": name,
                     "region_id": _attribute_text(op, "prov.region_id"),
                     "operands": [{**_tensor_type(v), "source": owners[v]} for v in op.operands],
                     "results": [_tensor_type(v) for v in op.results]})
        seen.update(op.results)
        if name not in {"merlin_iface.resident_pack", "merlin_iface.matmul", "merlin_iface.evict"}:
            if len(op.results) != 1 or not isinstance(op.results[0].type, TensorType):
                raise ValueError("interface publication requires one typed tensor result")
            returns.append({**_tensor_type(op.results[0]), "source": owners[op.results[0]]})
    if not returns:
        raise ValueError("interface source publishes no tensor outputs")
    return _inventory(raw, [_tensor_type(op.results[0]) for op in declarations], rows, returns, "merlin_iface.module")
