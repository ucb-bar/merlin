"""Target-neutral formation of exact second-tensor residual epilogues.

The recognizer records source operations and indexing maps; it does not choose an
accelerator instruction.  In particular, ordered f32 multiplications, per-channel
bias, residual broadcasting, round-to-nearest-even, and clamping remain distinct
semantic stages.  A target may select a fused host traversal only when every
structural proof below succeeds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from xdsl.dialects.builtin import IntegerType, TensorType
from xdsl.ir import Operation, SSAValue
from xdsl.ir.affine import AffineConstantExpr, AffineDimExpr


@dataclass(frozen=True)
class OrderedBranch:
    """One i32 tensor converted through the frozen ordered affine+bias chain."""

    producer: Operation
    affine: Operation
    bias: Operation
    support_operations: tuple[Operation, ...]
    output: SSAValue
    shape: tuple[int, ...]
    channel_axis: int
    accumulator_scale: SSAValue
    per_channel_scale: SSAValue
    per_channel_bias: SSAValue

    @property
    def operations(self) -> tuple[Operation, ...]:
        return (*self.support_operations, self.affine, self.bias)


@dataclass(frozen=True)
class ResidualEpilogue:
    """Exact residual add with an optional ReLU and exact i8 quantization sink."""

    residual_add: Operation
    relu: Operation | None
    quantize: Operation
    output: SSAValue
    exposed_float: SSAValue
    shape: tuple[int, ...]
    branches: tuple[OrderedBranch, ...]
    residual_input_maps: tuple[tuple[int | str, ...], tuple[int | str, ...]]
    output_scale_reciprocal: SSAValue
    zero_point: SSAValue

    @property
    def float_sink(self) -> Operation:
        return self.relu or self.residual_add

    def receipt(self, source_indices: dict[Operation, int], deferred: tuple[int, ...] = ()) -> dict[str, Any]:
        return {
            "schema": "target_neutral_residual_epilogue_v1",
            "residual_add_source_op_index": source_indices[self.residual_add],
            "relu_source_op_index": (
                source_indices[self.relu] if self.relu is not None else None),
            "quantize_source_op_index": source_indices[self.quantize],
            "shape": list(self.shape),
            "residual_input_maps": [list(row) for row in self.residual_input_maps],
            "ordered_i32_branches": [
                {
                    "producer_source_op_index": source_indices[branch.producer],
                    "source_op_indices": sorted(
                        source_indices[op] for op in branch.operations),
                    "channel_axis": branch.channel_axis,
                }
                for branch in self.branches
            ],
            "deferred_branch_producers": list(deferred),
            "stages": [
                "ordered_branch_f32_affine_and_bias", "second_tensor_add",
                *(["relu"] if self.relu is not None else []),
                "output_scale_reciprocal", "roundeven", "zero_point", "clamp_i8",
            ],
            "numeric_contract": {
                "source_f32_operation_order_preserved": True,
                "broadcast_indexing_maps_preserved": True,
                "rounding": "round_to_nearest_even",
                "clamp": [-128, 127],
                "reassociation": False,
            },
            "formed_semantics": "exact_second_tensor_residual_epilogue",
        }


@dataclass(frozen=True)
class Refusal:
    operation: Operation
    reason: str


def _shape(value: SSAValue) -> tuple[int, ...] | None:
    ty = value.type
    if not isinstance(ty, TensorType):
        return None
    return tuple(int(dim) for dim in ty.get_shape())


def _width(value: SSAValue) -> int | None:
    ty = value.type
    if not isinstance(ty, TensorType) or not isinstance(ty.element_type, IntegerType):
        return None
    return int(ty.element_type.width.data)


def _maps(op: Operation) -> tuple[tuple[int | str, ...], ...] | None:
    attr = op.properties.get("indexing_maps")
    if attr is None:
        attr = op.attributes.get("indexing_maps")
    data = getattr(attr, "data", None)
    if data is None:
        return None
    result: list[tuple[int | str, ...]] = []
    for item in data:
        amap = getattr(item, "data", None)
        if amap is None or amap.num_symbols:
            return None
        row: list[int | str] = []
        for expr in amap.results:
            if isinstance(expr, AffineDimExpr):
                row.append(int(expr.position))
            elif isinstance(expr, AffineConstantExpr):
                row.append(f"c{int(expr.value)}")
            else:
                return None
        result.append(tuple(row))
    return tuple(result)


def _parallel_pointwise(op: Operation, shape: tuple[int, ...]) -> bool:
    if op.name != "linalg.generic" or len(op.results) != 1 or _shape(op.results[0]) != shape:
        return False
    attrs = op.properties.get("iterator_types")
    if attrs is None:
        attrs = op.attributes.get("iterator_types")
    entries = getattr(attrs, "data", ())
    return len(entries) == len(shape) and all("parallel" in str(item) for item in entries)


def _body(op: Operation, names: tuple[str, ...], argc: int):
    if len(op.regions) != 1 or len(op.regions[0].blocks) != 1:
        return None
    block = op.regions[0].blocks[0]
    operations = tuple(block.ops)
    if len(block.args) != argc or tuple(item.name for item in operations) != names:
        return None
    return block, operations


def _literal(value: SSAValue) -> float | int | None:
    owner = value.owner if isinstance(value.owner, Operation) else None
    if owner is None or owner.name != "arith.constant":
        return None
    attr = owner.properties.get("value")
    if attr is None:
        attr = owner.attributes.get("value")
    raw = getattr(getattr(attr, "value", None), "data", None)
    if raw is None:
        raw = getattr(attr, "data", None)
    return raw if isinstance(raw, (int, float)) else None


def _splat_literal(value: SSAValue) -> float | int | None:
    owner = value.owner if isinstance(value.owner, Operation) else None
    if owner is None or owner.name != "tensor.splat" or not owner.operands:
        return None
    return _literal(owner.operands[0])


def _match_reciprocal(value: SSAValue) -> bool:
    op = value.owner if isinstance(value.owner, Operation) else None
    if op is None or op.name != "linalg.generic" or _shape(value) != () or len(op.inputs) != 1:
        return False
    if _maps(op) != ((), ()):
        return False
    found = _body(op, ("arith.constant", "arith.divf", "linalg.yield"), 2)
    if found is None:
        return False
    block, (one, divide, yielded) = found
    return (_literal(one.results[0]) == 1.0
            and tuple(divide.operands) == (one.results[0], block.args[0])
            and tuple(yielded.operands) == (divide.results[0],))


def _match_quantize(op: Operation, source: SSAValue, shape: tuple[int, ...]):
    identity = tuple(range(len(shape)))
    if (not _parallel_pointwise(op, shape) or _width(op.results[0]) != 8
            or len(op.inputs) != 3 or op.inputs[0] is not source
            or _maps(op) != (identity, (), (), identity)):
        return None
    reciprocal, zero_point = op.inputs[1], op.inputs[2]
    if not _match_reciprocal(reciprocal) or _splat_literal(zero_point) != 0:
        return None
    found = _body(op, (
        "arith.mulf", "math.roundeven", "arith.sitofp", "arith.addf",
        "arith.maximumf", "arith.minimumf", "arith.fptosi", "linalg.yield"), 4)
    if found is None:
        return None
    block, (scale, rounded, zp_float, added, lower, upper, convert, yielded) = found
    if (tuple(scale.operands) != (block.args[0], block.args[1])
            or tuple(rounded.operands) != (scale.results[0],)
            or tuple(zp_float.operands) != (block.args[2],)
            or tuple(added.operands) != (rounded.results[0], zp_float.results[0])
            or lower.operands[0] is not added.results[0]
            or upper.operands[0] is not lower.results[0]
            or tuple(convert.operands) != (upper.results[0],)
            or tuple(yielded.operands) != (convert.results[0],)
            or _literal(lower.operands[1]) != -128.0
            or _literal(upper.operands[1]) != 127.0):
        return None
    return reciprocal, zero_point


def _match_relu(op: Operation, source: SSAValue, shape: tuple[int, ...]) -> bool:
    identity = tuple(range(len(shape)))
    if (not _parallel_pointwise(op, shape) or len(op.inputs) != 1
            or op.inputs[0] is not source or _maps(op) != (identity, identity)):
        return False
    found = _body(op, ("arith.constant", "arith.maximumf", "linalg.yield"), 2)
    if found is None:
        return False
    block, (zero, maximum, yielded) = found
    return (_literal(zero.results[0]) == 0.0
            and tuple(maximum.operands) == (block.args[0], zero.results[0])
            and tuple(yielded.operands) == (maximum.results[0],))


def _support_for(op: Operation) -> tuple[Operation, ...] | None:
    """Exclusive empty initializers required by one exact pointwise op."""
    support: list[Operation] = []
    for value in op.outputs:
        owner = value.owner if isinstance(value.owner, Operation) else None
        if owner is None or owner.name != "tensor.empty":
            return None
        if any(use.operation is not op for use in value.uses):
            return None
        support.append(owner)
    return tuple(support)


def match_ordered_branch(value: SSAValue) -> OrderedBranch | None:
    """Walk backward from f32 branch output to an exact single-use i32 producer."""
    shape = _shape(value)
    bias = value.owner if isinstance(value.owner, Operation) else None
    if shape is None or bias is None or not _parallel_pointwise(bias, shape):
        return None
    identity = tuple(range(len(shape)))
    maps = _maps(bias)
    if len(bias.inputs) != 2 or maps is None or len(maps) != 3 or maps[0] != identity or maps[2] != identity:
        return None
    channel_map = maps[1]
    if len(channel_map) != 1 or not isinstance(channel_map[0], int):
        return None
    found = _body(bias, ("arith.addf", "linalg.yield"), 3)
    if found is None:
        return None
    bias_block, (addition, yielded) = found
    if (tuple(addition.operands) != (bias_block.args[0], bias_block.args[1])
            or tuple(yielded.operands) != (addition.results[0],)):
        return None

    affine_value = bias.inputs[0]
    affine = affine_value.owner if isinstance(affine_value.owner, Operation) else None
    if affine is None or not _parallel_pointwise(affine, shape) or len(affine.inputs) != 3:
        return None
    affine_maps = _maps(affine)
    if (affine_maps is None or len(affine_maps) != 4
            or affine_maps[0] != identity or affine_maps[1] != ()
            or affine_maps[2] != channel_map or affine_maps[3] != identity):
        return None
    found = _body(affine, ("arith.sitofp", "arith.mulf", "arith.mulf", "linalg.yield"), 4)
    if found is None:
        return None
    affine_block, (cast, first, second, affine_yield) = found
    if (tuple(cast.operands) != (affine_block.args[0],)
            or tuple(first.operands) != (cast.results[0], affine_block.args[1])
            or tuple(second.operands) != (first.results[0], affine_block.args[2])
            or tuple(affine_yield.operands) != (second.results[0],)):
        return None
    producer = affine.inputs[0].owner if isinstance(affine.inputs[0].owner, Operation) else None
    producer_uses = list(affine.inputs[0].uses)
    affine_uses = list(affine.results[0].uses)
    if (producer is None or _width(affine.inputs[0]) != 32
            or len(producer_uses) != 1 or producer_uses[0].operation is not affine
            or len(affine_uses) != 1 or affine_uses[0].operation is not bias):
        return None
    affine_support, bias_support = _support_for(affine), _support_for(bias)
    if affine_support is None or bias_support is None:
        return None
    return OrderedBranch(
        producer, affine, bias, (*affine_support, *bias_support), value, shape,
        int(channel_map[0]), affine.inputs[1], affine.inputs[2], bias.inputs[1])


def _match_residual_add(op: Operation):
    shape = _shape(op.results[0]) if len(op.results) == 1 else None
    if shape is None or not _parallel_pointwise(op, shape) or len(op.inputs) != 2:
        return None
    maps = _maps(op)
    identity = tuple(range(len(shape)))
    if maps is None or len(maps) != 3 or maps[2] != identity:
        return None
    # Each projected input index must be statically in range.  Dim projections implement
    # ordinary broadcasting by omission; constant coordinates implement unit-axis broadcasts.
    for value, row in zip(op.inputs, maps[:2]):
        input_shape = _shape(value)
        if input_shape is None or len(input_shape) != len(row):
            return None
        for token, extent in zip(row, input_shape):
            if isinstance(token, int):
                if not 0 <= token < len(shape) or shape[token] > extent:
                    return None
            else:
                constant = int(token[1:])
                if not 0 <= constant < extent:
                    return None
    found = _body(op, ("arith.addf", "linalg.yield"), 3)
    if found is None:
        return None
    block, (addition, yielded) = found
    if (tuple(addition.operands) != (block.args[0], block.args[1])
            or tuple(yielded.operands) != (addition.results[0],)):
        return None
    return shape, maps[:2]


def recognize(op: Operation) -> ResidualEpilogue | Refusal | None:
    """Recognize from the second-tensor add through its exact i8 sink."""
    residual = _match_residual_add(op)
    if residual is None:
        return None
    shape, input_maps = residual
    branches = tuple(branch for value in op.inputs
                     if (branch := match_ordered_branch(value)) is not None)
    # Ordinary bias additions and other binary maps also have an addf body. They are not
    # residual candidates unless one input proves the complete i32 branch contract.
    if not branches:
        return None
    add_uses = list(op.results[0].uses)
    if len(add_uses) != 1:
        return Refusal(op, "residual add result is not single-use")
    current = op.results[0]
    consumer = add_uses[0].operation
    relu = consumer if _match_relu(consumer, current, shape) else None
    if relu is not None:
        current = relu.results[0]
        consumers = [use.operation for use in current.uses]
    else:
        consumers = [consumer]
    quantizers = [candidate for candidate in dict.fromkeys(consumers)
                  if _match_quantize(candidate, current, shape) is not None]
    if len(quantizers) != 1:
        return Refusal(op, "residual float result has no unique exact roundeven/clamp i8 sink")
    quantize = quantizers[0]
    # Other users are legal: they make the f32 value an exposed block output.  The selected
    # lowering writes it once while producing the i8 consumer in that same traversal.
    if any(use.operation is not quantize for use in op.results[0].uses) and relu is None:
        return Refusal(op, "unactivated residual add has a non-quantize side use")
    quant = _match_quantize(quantize, current, shape)
    assert quant is not None
    for branch in branches:
        uses = list(branch.output.uses)
        if len(uses) != 1 or uses[0].operation is not op:
            return Refusal(op, "ordered branch output is not exclusive to the residual add")
    return ResidualEpilogue(
        op, relu, quantize, quantize.results[0], current, shape, branches,
        input_maps, quant[0], quant[1])


def recognize_all(operations: list[Operation]) -> tuple[list[ResidualEpilogue], list[Refusal]]:
    formed: list[ResidualEpilogue] = []
    refused: list[Refusal] = []
    for operation in operations:
        result = recognize(operation)
        if isinstance(result, ResidualEpilogue):
            formed.append(result)
        elif isinstance(result, Refusal):
            refused.append(result)
    return formed, refused
