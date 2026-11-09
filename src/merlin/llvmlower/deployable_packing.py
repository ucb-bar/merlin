"""Source-proved constant producers and byte-only layouts for outlined chains."""

from __future__ import annotations

import math


def _provenance_only(op):
    return "prov.quant_inner" not in op.attributes and all(key.startswith("prov.") for key in op.attributes)


def canonicalize_constant_splats(module) -> int:
    """Preserve literal attributes while exposing existing cloneable fill producers.

    The caller owns this module. No literal is converted through Python numeric
    types. Dynamic/effectful/unknown splats are left intact for normal refusal.
    Local reduction/elementwise constants are compiled with their arithmetic;
    this is not a controller scalar-arithmetic interpreter.
    """
    from xdsl.dialects import arith, tensor
    from xdsl.dialects.builtin import (
        Float32Type,
        Float64Type,
        FloatAttr,
        IntegerAttr,
        IntegerType,
        NoneAttr,
        TensorType,
    )
    from xdsl.dialects.linalg.ops import FillOp
    from xdsl.ir import OpResult
    from xdsl.rewriter import Rewriter

    count = 0
    for op in list(module.walk()):
        if (
            not isinstance(op, tensor.SplatOp)
            or len(op.operands) != 1
            or op.properties
            or not _provenance_only(op)
            or "prov.quant_inner" in op.attributes
        ):
            continue
        result = op.results[0].type
        value = op.operands[0]
        if (
            not isinstance(result, TensorType)
            or not isinstance(result.encoding, NoneAttr)
            or not isinstance(value, OpResult)
        ):
            continue
        owner = value.owner
        if (
            not isinstance(owner, arith.ConstantOp)
            or not _provenance_only(owner)
            or set(owner.properties) != {"value"}
            or "prov.quant_inner" in owner.attributes
        ):
            continue
        elem = result.element_type
        if value.type != elem or not result.get_shape() or any(d <= 0 for d in result.get_shape()):
            continue
        literal = owner.properties["value"]
        admitted = (
            isinstance(elem, IntegerType) and elem.width.data in (8, 16, 32, 64) and isinstance(literal, IntegerAttr)
        ) or (isinstance(elem, (Float32Type, Float64Type)) and isinstance(literal, FloatAttr))
        if not admitted:
            continue
        empty = tensor.EmptyOp([], result)
        fill = FillOp(inputs=[value], outputs=[empty.tensor], res=[result], attributes=dict(op.attributes))
        Rewriter.replace_op(op, [empty, fill], new_results=fill.results)
        count += 1
    return count


def static_permutation(function, node, specs):
    """Prove a complete fresh-output transpose kernel is only a byte permutation."""
    from xdsl.dialects.builtin import DenseArrayBase, NoneAttr, TensorType

    if len(node.inputs) != 1 or len(node.outputs) != 1 or len(function.body.blocks) != 1:
        return None
    if function.attributes or set(function.properties) - {"sym_name", "sym_visibility", "function_type"}:
        return None
    block = function.body.block
    ops = list(block.ops)
    if len(block.args) != 1 or len(ops) != 3:
        return None
    empty, transpose, returned = ops
    if (
        empty.name != "tensor.empty"
        or empty.operands
        or empty.regions
        or empty.properties
        or "prov.quant_inner" in empty.attributes
        or transpose.name != "linalg.transpose"
        or returned.name != "func.return"
        or not all(_provenance_only(op) for op in ops)
        or set(transpose.properties) != {"permutation"}
        or returned.properties
        or len(empty.results) != 1
        or len(transpose.results) != 1
        or tuple(transpose.operands) != (block.args[0], empty.results[0])
        or tuple(returned.operands) != (transpose.results[0],)
        or len(transpose.regions) != 1
        or len(transpose.regions[0].blocks) != 1
    ):
        return None
    body = transpose.regions[0].block
    inner = list(body.ops)
    if (
        len(body.args) != 2
        or len(inner) != 1
        or inner[0].name != "linalg.yield"
        or tuple(inner[0].operands) != (body.args[0],)
        or inner[0].properties
        or not _provenance_only(inner[0])
    ):
        return None
    source_type, output_type = block.args[0].type, transpose.results[0].type
    if (
        not isinstance(source_type, TensorType)
        or not isinstance(output_type, TensorType)
        or not isinstance(source_type.encoding, NoneAttr)
        or not isinstance(output_type.encoding, NoneAttr)
    ):
        return None
    if not isinstance(transpose.properties["permutation"], DenseArrayBase):
        return None
    if (
        empty.results[0].type != output_type
        or source_type.element_type != output_type.element_type
        or any(arg.type != source_type.element_type for arg in body.args)
        or tuple(function.function_type.inputs) != (source_type,)
        or tuple(function.function_type.outputs) != (output_type,)
    ):
        return None
    permutation = tuple(int(x) for x in transpose.properties["permutation"].get_values())
    source = specs[node.inputs[0]]
    output = specs[node.outputs[0]]
    if (
        not 1 <= len(source.shape) <= 8
        or len(permutation) != len(source.shape)
        or sorted(permutation) != list(range(len(source.shape)))
        or output.shape != tuple(source.shape[i] for i in permutation)
        or source.dtype != output.dtype
        or source.bytes != output.bytes
    ):
        return None
    strides, dense = [], 1
    for size in reversed(source.shape):
        strides.append(dense)
        dense *= size
    strides.reverse()
    return {
        "kind": "pack_permutation",
        "input": source.id,
        "output": output.id,
        "source_kernel": node.op,
        "source_op": transpose.name,
        "permutation": permutation,
        "source_shape": source.shape,
        "output_shape": output.shape,
        "source_strides": tuple(strides[i] for i in permutation),
        "element_bytes": source.bytes // math.prod(source.shape),
        "effect": "readonly source and complete fresh dense destination; byte copies only",
    }
