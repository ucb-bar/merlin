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
        if value.type != elem or any(d <= 0 for d in result.get_shape()):
            continue
        literal = owner.properties["value"]
        admitted = (
            isinstance(elem, IntegerType) and elem.width.data in (1, 8, 16, 32, 64) and isinstance(literal, IntegerAttr)
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
    # Raw byte permutations do not establish the logical i1 storage contract.
    # Keep this operation on the compiled compute worker instead.
    if str(source_type.element_type) == "i1":
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


def static_concat(op, node, specs):
    """Prove ordered static dense byte concatenation into a fresh tensor.

    Inputs may alias each other; the ordinary unique whole-chain allocation
    must keep the destination disjoint. This grants no in-place cache update,
    arithmetic interpretation, implicit bool storage or shortened lifetime.
    """
    from xdsl.dialects import tensor
    from xdsl.dialects.builtin import IntegerAttr

    from .deployable_outline import static_tensor_contract

    if (
        not isinstance(op, tensor.ConcatOp)
        or not _provenance_only(op)
        or op.regions
        or set(op.properties) != {"dim"}
        or not isinstance(op.properties["dim"], IntegerAttr)
        or not op.operands
        or len(node.inputs) != len(op.operands)
        or len(node.outputs) != 1
        or len(op.results) != 1
    ):
        raise ValueError("unsupported static concat tensor/effect contract")
    destination = specs[node.outputs[0]]
    out_shape, dtype, out_bytes = static_tensor_contract(op.results[0].type)
    axis = op.properties["dim"].value.data
    if (
        destination.arg_index is not None
        or node.outputs[0] in node.inputs
        or not 1 <= len(out_shape) <= 8
        or type(axis) is not int
        or not 0 <= axis < len(out_shape)
        or dtype == "i1"
        or (destination.shape, destination.dtype, destination.bytes) != (out_shape, dtype, out_bytes)
    ):
        raise ValueError("concat requires ordinary ranked dense nonboolean storage and a static in-bounds axis")
    shapes = []
    for bid, operand in zip(node.inputs, op.operands, strict=True):
        shape, element, size = static_tensor_contract(operand.type)
        source = specs[bid]
        if (
            (source.shape, source.dtype, source.bytes) != (shape, element, size)
            or element != dtype
            or len(shape) != len(out_shape)
            or any(s != d for index, (s, d) in enumerate(zip(shape, out_shape, strict=True)) if index != axis)
        ):
            raise ValueError("concat input types/shapes do not match the complete output contract")
        shapes.append(shape)
    if sum(shape[axis] for shape in shapes) != out_shape[axis]:
        raise ValueError("concat axis extents do not cover the exact output")
    element_bytes = out_bytes // math.prod(out_shape)
    inner = math.prod(out_shape[axis + 1 :])
    outer = math.prod(out_shape[:axis])
    blocks = [shape[axis] * inner * element_bytes for shape in shapes]
    if sum(blocks) * outer != out_bytes or any(
        block * outer != specs[bid].bytes for bid, block in zip(node.inputs, blocks, strict=True)
    ):
        raise ValueError("concat static byte spans are inconsistent")
    return {
        "kind": "pack_concat",
        "inputs": list(node.inputs),
        "output": node.outputs[0],
        "source_op": op.name,
        "source_attributes": {key: str(value) for key, value in op.attributes.items()},
        "axis": axis,
        "source_shapes": shapes,
        "output_shape": out_shape,
        "dtype": dtype,
        "outer_blocks": outer,
        "input_block_bytes": blocks,
        "output_block_bytes": sum(blocks),
        "element_bytes": element_bytes,
        "logical_bytes_read": out_bytes,
        "logical_bytes_written": out_bytes,
        "effect": "readonly inputs may alias each other; fresh disjoint complete output; no lifetime reuse",
    }
