"""Structural formation of exact contraction-to-i8 epilogues.

This is a source-semantic recognizer, not a target selector.  It deliberately records the
floating-point operation order rather than replacing it with an algebraically equivalent
formula: two rounded ``mulf`` operations are not one multiply by a precomputed product.  Target
lowering may therefore choose an exact fused host loop, or a native readout only when a separate
capability proof says the target implements every recorded stage in that order.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from xdsl.dialects.builtin import IntegerType, TensorType
from xdsl.ir import Operation, SSAValue
from xdsl.ir.affine import AffineDimExpr


@dataclass(frozen=True)
class QuantizedEpilogue:
    """One exact, single-use i32 contraction epilogue ending at an i8 tensor."""

    producer: Operation
    operations: tuple[Operation, ...]
    output: SSAValue
    shape: tuple[int, ...]
    channel_axis: int
    per_channel_scale: SSAValue
    per_channel_bias: SSAValue
    accumulator_scale: SSAValue
    output_scale_reciprocal: SSAValue
    zero_point: SSAValue
    relu: bool

    @property
    def stages(self) -> tuple[str, ...]:
        stages = ["ordered_f32_acc_scale", "per_channel_scale", "per_channel_bias"]
        if self.relu:
            stages.append("relu")
        stages.extend(("output_scale_reciprocal", "roundeven", "zero_point", "clamp_i8"))
        return tuple(stages)

    def receipt(self, source_indices: dict[Operation, int]) -> dict[str, Any]:
        return {
            "schema": "target_neutral_quantized_epilogue_v1",
            "producer_source_op_index": source_indices[self.producer],
            "source_op_indices": [source_indices[op] for op in self.operations],
            "sink_source_op_index": source_indices[self.operations[-1]],
            "shape": list(self.shape),
            "channel_axis": self.channel_axis,
            "stages": list(self.stages),
            "output_dtype": "i8",
            "numeric_contract": {
                "float_order_preserved": True,
                "rounding": "round_to_nearest_even",
                "clamp": [-128, 127],
                "reassociation": False,
            },
            # Target selection is deliberately absent here.  This record describes source
            # semantics; a backend capability selector decides whether it can erase the
            # full-width boundary or must retain the original host chain.
            "target_selection": "deferred",
        }


@dataclass(frozen=True)
class Refusal:
    producer: Operation
    reason: str
    blocker: Operation | None = None

    def receipt(self, source_indices: dict[Operation, int]) -> dict[str, Any]:
        row: dict[str, Any] = {
            "producer_source_op_index": source_indices[self.producer],
            "reason": self.reason,
        }
        if self.blocker is not None and self.blocker in source_indices:
            row["blocker_source_op_index"] = source_indices[self.blocker]
            row["blocker_op"] = self.blocker.name
        return row


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


def _maps(op: Operation) -> tuple[tuple[int, ...], ...] | None:
    attr = op.properties.get("indexing_maps") or op.attributes.get("indexing_maps")
    data = getattr(attr, "data", None)
    if data is None:
        return None
    result = []
    for item in data:
        amap = getattr(item, "data", None)
        if amap is None or amap.num_symbols:
            return None
        positions = []
        for expr in amap.results:
            if not isinstance(expr, AffineDimExpr):
                return None
            positions.append(int(expr.position))
        result.append(tuple(positions))
    return tuple(result)


def _parallel_pointwise(op: Operation, shape: tuple[int, ...]) -> bool:
    if op.name != "linalg.generic" or len(op.results) != 1 or _shape(op.results[0]) != shape:
        return False
    attrs = op.properties.get("iterator_types") or op.attributes.get("iterator_types")
    entries = getattr(attrs, "data", ())
    return len(entries) == len(shape) and all("parallel" in str(item) for item in entries)


def _body(op: Operation, names: tuple[str, ...], argc: int):
    if len(op.regions) != 1 or len(op.regions[0].blocks) != 1:
        return None
    block = op.regions[0].blocks[0]
    if len(block.args) != argc or tuple(item.name for item in block.ops) != names:
        return None
    return block, tuple(block.ops)


def _sole_consumer(value: SSAValue) -> Operation | None:
    uses = list(value.uses)
    return uses[0].operation if len(uses) == 1 else None


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


def exact_identity_narrowing(value: QuantizedEpilogue) -> tuple[bool, str]:
    """Prove the strict subset equal to a native identity-scale narrow readout.

    This is intentionally a semantic proof, not an approximate scale combiner.  For the
    admitted subset the source evaluates ``float(acc) * 1 * 1 + 0``, optionally applies ReLU,
    multiplies by the reciprocal of an output scale equal to one, round-even converts, and
    clamps to i8.  Native identity narrowing has the same result for every i32 accumulator:
    values in the i8 range are exactly representable in f32 and values outside it saturate to
    the same endpoint.  Runtime/per-channel values and non-identity arithmetic fail closed.

    The proof is target-neutral.  It says the rich source operation can be reduced to the
    abstract stages ``[optional relu, narrow_i8]``; target capability selection remains in the
    lowering layer.
    """
    if _splat_literal(value.accumulator_scale) != 1.0:
        return False, "accumulator_scale_is_not_proven_identity"
    if _splat_literal(value.per_channel_scale) != 1.0:
        return False, "per_channel_scale_is_not_proven_uniform_identity"
    if _splat_literal(value.per_channel_bias) != 0.0:
        return False, "per_channel_bias_is_not_proven_uniform_zero"
    reciprocal = value.output_scale_reciprocal.owner
    if (not isinstance(reciprocal, Operation) or reciprocal.name != "linalg.generic"
            or len(reciprocal.operands) < 1
            or _splat_literal(reciprocal.operands[0]) != 1.0):
        return False, "output_scale_is_not_proven_identity"
    if _splat_literal(value.zero_point) != 0:
        return False, "output_zero_point_is_not_proven_zero"
    return True, "exact_identity_narrowing"


def _match_affine(op: Operation, source: SSAValue, shape: tuple[int, ...]):
    if not _parallel_pointwise(op, shape) or len(op.inputs) != 3:
        return None
    maps = _maps(op)
    identity = tuple(range(len(shape)))
    if maps is None or len(maps) != 4 or maps[0] != identity or maps[1] != () or maps[3] != identity:
        return None
    channel = maps[2]
    if len(channel) != 1 or op.inputs[0] is not source:
        return None
    found = _body(op, ("arith.sitofp", "arith.mulf", "arith.mulf", "linalg.yield"), 4)
    if found is None:
        return None
    block, (cast, first, second, yielded) = found
    if (tuple(cast.operands) != (block.args[0],)
            or tuple(first.operands) != (cast.results[0], block.args[1])
            or tuple(second.operands) != (first.results[0], block.args[2])
            or tuple(yielded.operands) != (second.results[0],)):
        return None
    return int(channel[0]), op.inputs[1], op.inputs[2]


def _looks_like_affine(op: Operation, source: SSAValue, shape: tuple[int, ...]) -> bool:
    """Whether the outer maps claim the affine stage whose body must be checked exactly."""
    if not _parallel_pointwise(op, shape) or len(op.inputs) != 3 or op.inputs[0] is not source:
        return False
    maps = _maps(op)
    identity = tuple(range(len(shape)))
    return (maps is not None and len(maps) == 4 and maps[0] == identity
            and maps[1] == () and len(maps[2]) == 1 and maps[3] == identity)


def _match_bias(op: Operation, source: SSAValue, shape: tuple[int, ...], channel_axis: int):
    if not _parallel_pointwise(op, shape) or len(op.inputs) != 2 or op.inputs[0] is not source:
        return None
    identity = tuple(range(len(shape)))
    if _maps(op) != (identity, (channel_axis,), identity):
        return None
    found = _body(op, ("arith.addf", "linalg.yield"), 3)
    if found is None:
        return None
    block, (add, yielded) = found
    if (tuple(add.operands) != (block.args[0], block.args[1])
            or tuple(yielded.operands) != (add.results[0],)):
        return None
    return op.inputs[1]


def _match_relu(op: Operation, source: SSAValue, shape: tuple[int, ...]) -> bool:
    if not _parallel_pointwise(op, shape) or len(op.inputs) != 1 or op.inputs[0] is not source:
        return False
    identity = tuple(range(len(shape)))
    if _maps(op) != (identity, identity):
        return False
    found = _body(op, ("arith.constant", "arith.maximumf", "linalg.yield"), 2)
    if found is None:
        return False
    block, (zero, maximum, yielded) = found
    return (_literal(zero.results[0]) == 0.0
            and tuple(maximum.operands) == (block.args[0], zero.results[0])
            and tuple(yielded.operands) == (maximum.results[0],))


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
    if (not _parallel_pointwise(op, shape) or _width(op.results[0]) != 8
            or len(op.inputs) != 3 or op.inputs[0] is not source):
        return None
    identity = tuple(range(len(shape)))
    if _maps(op) != (identity, (), (), identity):
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


def _is_dynamic_tensor_add(op: Operation, source: SSAValue, shape: tuple[int, ...]) -> bool:
    if not _parallel_pointwise(op, shape) or len(op.inputs) != 2 or source not in op.inputs:
        return False
    identity = tuple(range(len(shape)))
    if _maps(op) != (identity, identity, identity):
        return False
    other = op.inputs[1] if op.inputs[0] is source else op.inputs[0]
    if _shape(other) != shape:
        return False
    found = _body(op, ("arith.addf", "linalg.yield"), 3)
    return found is not None


def recognize(producer: Operation) -> QuantizedEpilogue | Refusal | None:
    """Recognize the exact common chain starting at an i32 contraction result.

    ``None`` means the producer does not begin this spelling.  A :class:`Refusal` means the
    chain began exactly but reached a semantically important unsupported branch, which is useful
    evidence rather than a silent non-match.
    """
    if len(producer.results) != 1 or _width(producer.results[0]) != 32:
        return None
    shape = _shape(producer.results[0])
    if shape is None or len(shape) < 2:
        return None
    affine = _sole_consumer(producer.results[0])
    if affine is None:
        return None
    affine_match = _match_affine(affine, producer.results[0], shape)
    if affine_match is None:
        if _looks_like_affine(affine, producer.results[0], shape):
            return Refusal(
                producer,
                "ordered affine body is not exact accumulator-scale then per-channel-scale",
                affine)
        return None
    channel_axis, accumulator_scale, channel_scale = affine_match
    bias = _sole_consumer(affine.results[0])
    if bias is None:
        return Refusal(producer, "affine result is not single-use")
    bias_value = _match_bias(bias, affine.results[0], shape, channel_axis)
    if bias_value is None:
        if _is_dynamic_tensor_add(bias, affine.results[0], shape):
            return Refusal(producer, "dynamic second tensor/residual terminates epilogue", bias)
        return Refusal(producer, "ordered affine is not followed by exact per-channel bias", bias)

    current = bias.results[0]
    operations = [affine, bias]
    next_op = _sole_consumer(current)
    if next_op is None:
        return Refusal(producer, "bias result is not single-use")
    relu = _match_relu(next_op, current, shape)
    if relu:
        operations.append(next_op)
        current = next_op.results[0]
        next_op = _sole_consumer(current)
        if next_op is None:
            return Refusal(producer, "relu result is not single-use")
    if _is_dynamic_tensor_add(next_op, current, shape):
        return Refusal(producer, "dynamic second tensor/residual terminates epilogue", next_op)
    quant = _match_quantize(next_op, current, shape)
    if quant is None:
        return Refusal(producer, "chain does not end in exact roundeven/clamp i8 quantize", next_op)
    reciprocal, zero_point = quant
    operations.append(next_op)
    return QuantizedEpilogue(
        producer=producer,
        operations=tuple(operations),
        output=next_op.results[0],
        shape=shape,
        channel_axis=channel_axis,
        per_channel_scale=channel_scale,
        per_channel_bias=bias_value,
        accumulator_scale=accumulator_scale,
        output_scale_reciprocal=reciprocal,
        zero_point=zero_point,
        relu=relu,
    )
