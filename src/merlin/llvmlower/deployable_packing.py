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


def tile_policy(tile_shape, tile_budget_bytes):
    """Validate explicit logical-footprint knobs; infer no cache/target facts."""
    if tile_shape is None and tile_budget_bytes is None:
        return None
    if (
        not isinstance(tile_shape, tuple)
        or len(tile_shape) != 2
        or any(type(size) is not int or not 0 < size < 1 << 63 for size in tile_shape)
        or type(tile_budget_bytes) is not int
        or not 0 < tile_budget_bytes < 1 << 63
    ):
        raise ValueError("positive explicit rank2 tile_shape and logical tile budget required")
    return {"tile_shape": tile_shape, "logical_tile_budget_bytes": tile_budget_bytes}


def select_packing_policy(view, policy):
    """Select only on an already-proved fresh dense byte permutation."""
    if policy is None:
        return view
    result = dict(view)
    common = {
        **policy,
        "extra_scratch_bytes": 0,
        "logical_read_bytes": math.prod(view["source_shape"]) * view["element_bytes"],
        "logical_write_bytes": math.prod(view["output_shape"]) * view["element_bytes"],
        "physical_traffic_bytes": None,
        "physical_cache_geometry_known": False,
        "traffic_basis": "one logical element read and write; not physical transactions or cache traffic",
    }
    if len(view["output_shape"]) != 2:
        result["packing_policy"] = {
            **common,
            "strategy": "flat",
            "reason": "blocked selection supports rank2 only; proved flat permutation retained",
        }
        return result
    actual = tuple(
        min(requested, size) for requested, size in zip(policy["tile_shape"], view["output_shape"], strict=True)
    )
    footprint = 2 * math.prod(actual) * view["element_bytes"]
    if footprint > policy["logical_tile_budget_bytes"]:
        raise ValueError(f"source+destination logical tile footprint {footprint} exceeds declared budget")
    result["packing_policy"] = {
        **common,
        "strategy": "blocked_rank2",
        "effective_tile_shape": actual,
        "logical_tile_footprint_bytes": footprint,
        "budget_basis": "source plus destination logical tile bytes; no scratch allocation or assumed cache capacity",
    }
    return result


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


def static_insert_slice(op, node, specs):
    """Prove a complete fresh-output static insertion, never in-place mutation.

    Copying the entire initial tensor precedes bounded ordered source spans.
    Both inputs remain immutable and may alias each other; unique whole-chain
    storage keeps the result disjoint. Rank-reduced/dynamic/bool cases require
    separate proofs and do not gain permission here.
    """
    from xdsl.dialects import tensor
    from xdsl.dialects.builtin import DenseArrayBase

    from .deployable_outline import static_tensor_contract

    fields = {"static_offsets", "static_sizes", "static_strides", "operandSegmentSizes"}
    if (
        not isinstance(op, tensor.InsertSliceOp)
        or not _provenance_only(op)
        or op.regions
        or set(op.properties) != fields
        or any(not isinstance(op.properties[key], DenseArrayBase) for key in fields)
        or tuple(op.properties["operandSegmentSizes"].get_values()) != (1, 1, 0, 0, 0)
        or len(op.operands) != 2
        or len(op.results) != 1
        or len(node.inputs) != 2
        or len(node.outputs) != 1
    ):
        raise ValueError("unsupported fresh static insert tensor/effect contract")
    source, initial = (specs[bid] for bid in node.inputs)
    output = specs[node.outputs[0]]
    source_shape, dtype, source_bytes = static_tensor_contract(op.operands[0].type)
    initial_shape, initial_dtype, initial_bytes = static_tensor_contract(op.operands[1].type)
    out_shape, out_dtype, out_bytes = static_tensor_contract(op.results[0].type)
    if (
        not 1 <= len(initial_shape) <= 8
        or len(source_shape) != len(initial_shape)
        or dtype == "i1"
        or dtype != initial_dtype
        or (out_shape, out_dtype, out_bytes) != (initial_shape, dtype, initial_bytes)
        or (source.shape, source.dtype, source.bytes) != (source_shape, dtype, source_bytes)
        or (initial.shape, initial.dtype, initial.bytes) != (initial_shape, dtype, initial_bytes)
        or (output.shape, output.dtype, output.bytes) != (out_shape, dtype, out_bytes)
        or output.arg_index is not None
        or node.outputs[0] in node.inputs
    ):
        raise ValueError("insert requires exact same-rank nonboolean types and a distinct fresh output")
    offsets, sizes, strides = (
        tuple(op.properties[key].get_values()) for key in ("static_offsets", "static_sizes", "static_strides")
    )
    if (
        any(len(v) != len(initial_shape) for v in (offsets, sizes, strides))
        or sizes != source_shape
        or any(type(v) is not int for values in (offsets, sizes, strides) for v in values)
        or any(
            o < 0 or n <= 0 or s <= 0 or o + (n - 1) * s >= extent
            for o, n, s, extent in zip(offsets, sizes, strides, initial_shape, strict=True)
        )
    ):
        raise ValueError("insert static slice coordinates do not match source or initial bounds")
    dense, pitch = [], 1
    for extent in reversed(initial_shape):
        dense.append(pitch)
        pitch *= extent
    dense.reverse()
    width = source_bytes // math.prod(source_shape)
    contiguous_last = strides[-1] == 1
    span = source_shape[-1] if contiguous_last else 1
    iteration_shape = source_shape[:-1] if contiguous_last else source_shape
    selected_pitches = tuple(s * p for s, p in zip(strides, dense, strict=True))
    iteration_pitches = selected_pitches[:-1] if contiguous_last else selected_pitches
    if math.prod(iteration_shape) * span * width != source_bytes:
        raise ValueError("insert static spans do not cover the exact source bytes")
    return {
        "kind": "pack_insert_slice",
        "inputs": list(node.inputs),
        "output": node.outputs[0],
        "source_op": op.name,
        "source_attributes": {key: str(value) for key, value in op.attributes.items()},
        "source_shape": source_shape,
        "initial_shape": initial_shape,
        "output_shape": out_shape,
        "dtype": dtype,
        "offsets": offsets,
        "sizes": sizes,
        "strides": strides,
        "initial_bytes": initial_bytes,
        "element_bytes": width,
        "copy_count": math.prod(iteration_shape),
        "copy_bytes": span * width,
        "copy_iteration_shape": iteration_shape,
        "copy_destination_pitches": iteration_pitches,
        "copy_base_elements": sum(o * p for o, p in zip(offsets, dense, strict=True)),
        "logical_bytes_read": initial_bytes + source_bytes,
        "logical_bytes_written": initial_bytes + source_bytes,
        "effect": (
            "readonly inputs may alias; full initial copy and bounded source overwrite into fresh disjoint result; "
            "no lifetime reuse"
        ),
    }
