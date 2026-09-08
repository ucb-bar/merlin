"""Canonicalise and recognize Gemmini's native-aligned integer epilogue.

The designated capture contract is intentionally different from ordinary PT2E floating-point
QDQ spelling.  For a contraction whose two quantization scales and output scale are compile-time
scalars, and whose bias is an i32 tensor in accumulator units, the declared operation is::

  acc_i32 + bias_i32 -> one f32 multiplier -> round-even -> saturate i8 -> optional ReLU

This module has two deliberately separate halves. ``fold`` converts only a structurally proven
instance of the designated capture into that canonical source form. ``recognize`` later proves
that the canonical form is still intact before target selection.  Every mismatch is a refusal;
provenance strings and model names are never admission criteria.
"""
from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import Any

from xdsl.dialects.builtin import IntegerType, TensorType
from xdsl.ir import Operation, SSAValue
from xdsl.ir.affine import AffineDimExpr


CONTRACT = "i32_bias_then_one_f32_multiplier_roundeven_saturate_i8"
_LAYOUT = ("tensor.collapse_shape", "tensor.expand_shape")


def _f32(value: float) -> float:
    return struct.unpack("!f", struct.pack("!f", float(value)))[0]


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


def _name(op: Operation | None) -> str:
    if op is None:
        return ""
    if op.name != "builtin.unregistered":
        return op.name
    return str(getattr(getattr(op, "op_name", None), "data", ""))


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


def _int_property(op: Operation, name: str) -> int | None:
    attr = op.properties.get(name)
    if attr is None:
        attr = op.attributes.get(name)
    raw = getattr(getattr(attr, "value", None), "data", None)
    return int(raw) if isinstance(raw, int) else None


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


def _body(op: Operation, names: tuple[str, ...], argc: int):
    if len(op.regions) != 1 or len(op.regions[0].blocks) != 1:
        return None
    block = op.regions[0].blocks[0]
    operations = tuple(block.ops)
    if len(block.args) != argc or tuple(item.name for item in operations) != names:
        return None
    return block, operations


def _sole_consumer(value: SSAValue) -> Operation | None:
    uses = [use.operation for use in value.uses if use.operation.parent_block() is not None]
    return uses[0] if len(uses) == 1 else None


def _is_parallel_pointwise(op: Operation, shape: tuple[int, ...]) -> bool:
    if op.name != "linalg.generic" or len(op.results) != 1 or _shape(op.results[0]) != shape:
        return False
    attrs = op.properties.get("iterator_types") or op.attributes.get("iterator_types")
    entries = getattr(attrs, "data", ())
    return len(entries) == len(shape) and all("parallel" in str(item) for item in entries)


def _match_f32_bias(op: Operation, source: SSAValue):
    shape = _shape(source)
    if shape is None or not _is_parallel_pointwise(op, shape) or len(op.inputs) != 2:
        return None
    if op.inputs[0] is not source:
        return None
    maps = _maps(op)
    identity = tuple(range(len(shape)))
    if maps is None or len(maps) != 3 or maps[0] != identity or maps[2] != identity:
        return None
    if len(maps[1]) != 1:
        return None
    found = _body(op, ("arith.addf", "linalg.yield"), 3)
    if found is None:
        return None
    block, (add, yielded) = found
    if (tuple(add.operands) != (block.args[0], block.args[1])
            or tuple(yielded.operands) != (add.results[0],)):
        return None
    return int(maps[1][0]), op.inputs[1]


def _match_relu(op: Operation, source: SSAValue) -> bool:
    shape = _shape(source)
    if shape is None or not _is_parallel_pointwise(op, shape) or len(op.inputs) != 1:
        return False
    identity = tuple(range(len(shape)))
    if _maps(op) != (identity, identity) or op.inputs[0] is not source:
        return False
    found = _body(op, ("arith.constant", "arith.maximumf", "linalg.yield"), 2)
    if found is None:
        return False
    block, (zero, maximum, yielded) = found
    return (_literal(zero.results[0]) == 0.0
            and tuple(maximum.operands) == (block.args[0], zero.results[0])
            and tuple(yielded.operands) == (maximum.results[0],))


def _dequant_i32_bias(value: SSAValue):
    op = value.owner if isinstance(value.owner, Operation) else None
    if (_name(op) != "quant_ext.dequantize_per_tensor" or op is None
            or len(op.operands) < 3 or _width(op.operands[0]) != 32
            or _splat_literal(op.operands[2]) != 0):
        return None
    if _shape(op.operands[1]) != () or _shape(op.operands[2]) != ():
        return None
    return op, op.operands[0], op.operands[1]


def _erase(operation: Operation | None) -> None:
    if operation is None or operation.parent_block() is None:
        return
    if any(use.operation.parent_block() is not None for value in operation.results for use in value.uses):
        return
    operation.detach()
    operation.erase(safe_erase=False)


def _erase_dead_tree(seeds: list[Operation | None]) -> None:
    """Erase only known pure producers made dead by an admitted rewrite."""
    pure = {"arith.constant", "tensor.splat", "tensor.empty", "linalg.fill",
            "quant_ext.dequantize_per_tensor", "builtin.unregistered"}
    work = [item for item in seeds if item is not None]
    while work:
        op = work.pop()
        if op.parent_block() is None or _name(op) not in pure:
            continue
        operands = [v.owner for v in op.operands if isinstance(v.owner, Operation)]
        if any(use.operation.parent_block() is not None for value in op.results for use in value.uses):
            continue
        _erase(op)
        work.extend(operands)


def _old_requant_scales(op: Operation) -> tuple[float, float] | None:
    if op.name != "linalg.generic" or len(op.inputs) != 3 or _width(op.inputs[0]) != 32:
        return None
    if _shape(op.inputs[1]) != () or _shape(op.inputs[2]) != ():
        return None
    found = _body(op, ("arith.sitofp", "arith.mulf", "arith.mulf", "linalg.yield"), 4)
    if found is None:
        return None
    first, second = _splat_literal(op.inputs[1]), _splat_literal(op.inputs[2])
    if not isinstance(first, float) or not isinstance(second, float):
        return None
    return _f32(first), _f32(second)


def fold(module, report: dict[str, Any] | None = None) -> int:
    """Rewrite proven designated-QDQ epilogues into canonical accumulator form.

    Gemmini's repeating D preload broadcasts along M, so only a bias on the final ``N`` axis of
    a rank-2 contraction is admitted.  NCHW im2col convolutions therefore report the row-axis
    refusal; a future physical-layout/source-conv pass may reorient them without weakening this
    local proof.
    """
    from xdsl.dialects import arith, tensor
    from xdsl.dialects import math as mathd
    from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr, FloatAttr, f32, i8, i32
    from xdsl.dialects.linalg import ops as L
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineMap

    refusals: dict[str, int] = {}
    admitted = 0

    def refuse(reason: str) -> None:
        refusals[reason] = refusals.get(reason, 0) + 1

    producers = [op for op in module.walk()
                 if getattr(op.attributes.get("prov.role"), "data", "") == "contraction"
                 and len(op.results) == 1 and _width(op.results[0]) == 32]
    for producer in producers:
        requant = _sole_consumer(producer.results[0])
        scales = _old_requant_scales(requant) if requant is not None else None
        if scales is None:
            continue
        current = requant.results[0]
        layouts: list[Operation] = []
        while len(layouts) < 8:
            use = _sole_consumer(current)
            if use is None or use.name not in _LAYOUT or len(use.results) != 1:
                break
            layouts.append(use)
            current = use.results[0]
        bias_op = _sole_consumer(current)
        bias_match = _match_f32_bias(bias_op, current) if bias_op is not None else None
        if bias_match is None:
            continue
        bias_axis, bias_float = bias_match
        bias_info = _dequant_i32_bias(bias_float)
        if bias_info is None:
            refuse("bias_is_not_symmetric_i32_accumulator_units")
            continue
        bias_dequant, bias_i32, bias_scale = bias_info
        bias_shape = _shape(bias_i32)
        out_shape = _shape(producer.results[0])
        # Any reshape boundary makes the bias axis a logical consumer axis rather than the
        # contraction's hardware N axis. Refuse until global physical layout proves otherwise.
        if (layouts or out_shape is None or len(out_shape) != 2 or bias_axis != 1
                or bias_shape != (out_shape[1],)):
            refuse("bias_axis_not_gemmini_column")
            continue

        current = bias_op.results[0]
        relu_op = _sole_consumer(current)
        has_relu = relu_op is not None and _match_relu(relu_op, current)
        if has_relu:
            current = relu_op.results[0]
        quant = _sole_consumer(current)
        if (_name(quant) != "quant_ext.quantize_per_tensor" or quant is None
                or len(quant.operands) < 3 or quant.operands[0] is not current
                or _width(quant.results[0]) != 8
                or _splat_literal(quant.operands[2]) != 0
                or _int_property(quant, "quant_min") != -128
                or _int_property(quant, "quant_max") != 127):
            refuse("terminal_quantizer_is_not_symmetric_saturating_i8")
            continue
        output_scale = _splat_literal(quant.operands[1])
        source_bias_scale = _splat_literal(bias_scale)
        if not isinstance(output_scale, float) or output_scale <= 0.0:
            refuse("output_scale_is_not_positive_static_scalar")
            continue
        accumulator_scale = _f32(_f32(scales[0]) * _f32(scales[1]))
        if not isinstance(source_bias_scale, float) or _f32(source_bias_scale) != accumulator_scale:
            refuse("bias_scale_is_not_exact_accumulator_scale")
            continue
        multiplier = _f32(accumulator_scale / _f32(output_scale))
        if not multiplier > 0.0:
            refuse("combined_multiplier_is_not_positive_f32")
            continue

        block = producer.parent_block()
        shape = list(out_shape)
        identity = AffineMapAttr(AffineMap.identity(2))
        scalar = AffineMapAttr(AffineMap(2, 0, ()))
        column = AffineMapAttr(AffineMap(2, 0, (AffineDimExpr(1),)))
        parallel = ArrayAttr([L.IteratorTypeAttr(L.IteratorType.PARALLEL)] * 2)

        bias_type = TensorType(i32, shape)
        bias_empty = tensor.EmptyOp((), bias_type)
        bb = Block(arg_types=[i32, i32, i32])
        added = arith.AddiOp(bb.args[0], bb.args[1])
        bb.add_ops([added, L.YieldOp(added.result)])
        canonical_bias = L.GenericOp(
            inputs=(producer.results[0], bias_i32), outputs=(bias_empty.results[0],),
            body=Region(bb), indexing_maps=ArrayAttr([identity, column, identity]),
            iterator_types=parallel, result_types=(bias_type,))

        multiplier_c = arith.ConstantOp(FloatAttr(multiplier, f32))
        multiplier_s = tensor.SplatOp(multiplier_c.results[0], (), TensorType(f32, []))
        output_type = quant.results[0].type
        output_empty = tensor.EmptyOp((), output_type)
        nb = Block(arg_types=[i32, f32, i8])
        cast = arith.SIToFPOp(nb.args[0], f32)
        scaled = arith.MulfOp(cast.result, nb.args[1])
        rounded = mathd.RoundEvenOp(scaled.result)
        lower_c = arith.ConstantOp(FloatAttr(0.0 if has_relu else -128.0, f32))
        upper_c = arith.ConstantOp(FloatAttr(127.0, f32))
        lower = arith.MaximumfOp(rounded.result, lower_c.results[0])
        upper = arith.MinimumfOp(lower.result, upper_c.results[0])
        narrow = arith.FPToSIOp(upper.result, i8)
        nb.add_ops([cast, scaled, rounded, lower_c, upper_c, lower, upper, narrow,
                    L.YieldOp(narrow.result)])
        canonical_narrow = L.GenericOp(
            inputs=(canonical_bias.results[0], multiplier_s.results[0]),
            outputs=(output_empty.results[0],), body=Region(nb),
            indexing_maps=ArrayAttr([identity, scalar, identity]),
            iterator_types=parallel, result_types=(output_type,))
        from xdsl.dialects.builtin import StringAttr
        provenance = {k: v for k, v in producer.attributes.items() if k.startswith("prov.")}
        canonical_bias.attributes.update(provenance)
        canonical_bias.attributes["prov.role"] = StringAttr("accumulator_bias")
        canonical_narrow.attributes.update(provenance)
        canonical_narrow.attributes["prov.role"] = StringAttr("native_narrow")
        canonical_narrow.attributes["prov.native_aligned_contract"] = StringAttr(CONTRACT)
        canonical_narrow.attributes["prov.native_relu"] = StringAttr(
            "true" if has_relu else "false")
        for new in (bias_empty, canonical_bias, multiplier_c, multiplier_s,
                    output_empty, canonical_narrow):
            block.insert_op_before(new, quant)
        quant.results[0].replace_all_uses_with(canonical_narrow.results[0])

        dead_seeds = [
            quant, relu_op if has_relu else None, bias_op, requant,
            quant.operands[1].owner if isinstance(quant.operands[1].owner, Operation) else None,
            quant.operands[2].owner if isinstance(quant.operands[2].owner, Operation) else None,
            bias_dequant,
        ]
        # Consumer-first removal. Layouts cannot occur in an admitted formation.
        for old in dead_seeds[:4]:
            _erase(old)
        _erase_dead_tree(dead_seeds[4:])
        admitted += 1

    if report is not None:
        report.update({"admitted": admitted, "refused": dict(sorted(refusals.items())),
                       "source_contract": CONTRACT})
    return admitted


@dataclass(frozen=True)
class NativeAlignedEpilogue:
    producer: Operation
    operations: tuple[Operation, ...]
    output: SSAValue
    shape: tuple[int, ...]
    bias: SSAValue
    multiplier: float
    relu: bool

    def receipt(self, source_indices: dict[Operation, int]) -> dict[str, Any]:
        return {
            "schema": "native_aligned_i32_epilogue_v1",
            "producer_source_op_index": source_indices[self.producer],
            "source_op_indices": [source_indices[op] for op in self.operations],
            "sink_source_op_index": source_indices[self.operations[-1]],
            "shape": list(self.shape),
            "stages": ["bias", "acc_scale", "round_to_nearest_even", "saturate_i8",
                       *(["relu"] if self.relu else [])],
            "output_dtype": "i8",
            "numeric_contract": {
                "bias_domain": "accumulator_i32",
                "multiplier": self.multiplier,
                "rounding": "round_to_nearest_even",
                "saturation": [-128, 127],
                "relu_after_saturation": self.relu,
                "reassociation": False,
            },
            "target_selection": "exact_native",
        }


def recognize(producer: Operation) -> NativeAlignedEpilogue | None:
    """Prove the canonical form emitted by :func:`fold` without trusting its marker."""
    if len(producer.results) != 1 or _width(producer.results[0]) != 32:
        return None
    shape = _shape(producer.results[0])
    if shape is None or len(shape) != 2:
        return None
    bias_op = _sole_consumer(producer.results[0])
    if bias_op is None or not _is_parallel_pointwise(bias_op, shape) or len(bias_op.inputs) != 2:
        return None
    identity = tuple(range(2))
    if (_maps(bias_op) != (identity, (1,), identity)
            or bias_op.inputs[0] is not producer.results[0]
            or _width(bias_op.inputs[1]) != 32
            or _shape(bias_op.inputs[1]) != (shape[1],)):
        return None
    found = _body(bias_op, ("arith.addi", "linalg.yield"), 3)
    if found is None:
        return None
    block, (added, yielded) = found
    if (tuple(added.operands) != (block.args[0], block.args[1])
            or tuple(yielded.operands) != (added.results[0],)):
        return None

    narrow = _sole_consumer(bias_op.results[0])
    if (narrow is None or not _is_parallel_pointwise(narrow, shape) or len(narrow.inputs) != 2
            or narrow.inputs[0] is not bias_op.results[0] or _width(narrow.results[0]) != 8
            or _maps(narrow) != (identity, (), identity)):
        return None
    multiplier = _splat_literal(narrow.inputs[1])
    if not isinstance(multiplier, float) or multiplier <= 0.0:
        return None
    found = _body(narrow, (
        "arith.sitofp", "arith.mulf", "math.roundeven", "arith.constant",
        "arith.constant", "arith.maximumf", "arith.minimumf", "arith.fptosi",
        "linalg.yield"), 3)
    if found is None:
        return None
    nb, (cast, scaled, rounded, lower_c, upper_c, lower, upper, converted, yielded) = found
    lo, hi = _literal(lower_c.results[0]), _literal(upper_c.results[0])
    if lo not in (-128.0, 0.0) or hi != 127.0:
        return None
    if (tuple(cast.operands) != (nb.args[0],)
            or tuple(scaled.operands) != (cast.results[0], nb.args[1])
            or tuple(rounded.operands) != (scaled.results[0],)
            or tuple(lower.operands) != (rounded.results[0], lower_c.results[0])
            or tuple(upper.operands) != (lower.results[0], upper_c.results[0])
            or tuple(converted.operands) != (upper.results[0],)
            or tuple(yielded.operands) != (converted.results[0],)):
        return None
    return NativeAlignedEpilogue(
        producer, (bias_op, narrow), narrow.results[0], shape, bias_op.inputs[1],
        _f32(multiplier), lo == 0.0)
