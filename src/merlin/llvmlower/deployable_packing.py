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
