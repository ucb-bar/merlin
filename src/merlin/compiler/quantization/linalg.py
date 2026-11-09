"""Lower captured quantization expressions to exact target-neutral tensor arithmetic.

Preserve source ordering, numerical/effect obligations and explicit selection.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from merlin.compiler.ir.provenance import carry_provenance

_DEQUANT_KINDS = {
    "quant_ext.dequantize_per_tensor": "per_tensor",
    "quant_ext.dequantize_per_channel": "per_channel",
    "quant_ext.dequantize_per_group": "per_group",
}

_QUANT_KINDS = {
    "quant_ext.quantize_per_tensor": "per_tensor",
}

def lower_quant_ext(module, *, on_rewrite: Callable[[Any, tuple[Any, ...], Any], None] | None = None) -> int:
    """Rewrite supported quant_ext quantize/dequantize ops; returns the count.

    Generic dequant → f32 (or bf16) via a linalg.generic; the scale/zp indexing map is derived
    from the granularity: scalar-broadcast (per_tensor), axis-projection (per_channel), or
    axis-floordiv by group_size (per_group). Per-tensor activation quantization becomes the exact
    target-neutral PT2E expression
    ``clamp(roundeven(x * float32(1 / scale)) + zp, qmin, qmax)``.  The reciprocal-first order is
    load-bearing: it is TorchAO's specified operation order and differs from ``x / scale`` at some
    rounding boundaries.  The reciprocal is formed once as a rank-0 tensor rather than dividing
    every activation element. No target datapath is named here, so scalar, RVV, and generated
    accelerator backends see the same IR.
    """
    from xdsl.dialects import arith, tensor
    from xdsl.dialects import math as mathd
    from xdsl.dialects.builtin import (
        AffineMapAttr,
        ArrayAttr,
        Float64Type,
        FloatAttr,
        IntegerType,
        TensorType,
        f32,
        f64,
        i8,
        i32,
    )
    from xdsl.dialects.linalg import ops as linalg_ops
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineMap

    dequant_rewrites = []
    quant_rewrites = []
    for op in module.walk():
        # quant_ext is not a registered dialect; ops parse as builtin.unregistered.
        name = getattr(op, "op_name", None)
        if op.name == "builtin.unregistered" and name is not None and name.data in _DEQUANT_KINDS:
            dequant_rewrites.append((op, _DEQUANT_KINDS[name.data]))
        elif op.name == "builtin.unregistered" and name is not None and name.data in _QUANT_KINDS:
            quant_rewrites.append((op, _QUANT_KINDS[name.data]))

    for op, kind in dequant_rewrites:
        w, scale, zp = op.operands
        out_t = op.results[0].type
        rank = len(out_t.get_shape())
        block = op.parent_block()

        empty = tensor.EmptyOp((), out_t)
        identity = AffineMap.identity(rank)
        if kind == "per_tensor":
            # scalar scale/zp broadcast to every element: affine_map<(d...) -> ()>
            sz_map = AffineMap(rank, 0, ())
        elif kind == "per_channel":
            axis = int(op.properties["axis"].value.data)
            sz_map = AffineMap(rank, 0, (identity.results[axis],))
        else:  # per_group: scale/zp indexed by the group of the quantized axis (dim floordiv group_size)
            axis = int(op.properties["axis"].value.data) if "axis" in op.properties else rank - 1
            gsz = int(op.properties["group_size"].value.data)
            results = list(identity.results)
            results[axis] = identity.results[axis] // gsz  # xDSL AffineExpr floordiv
            sz_map = AffineMap(rank, 0, tuple(results))
        maps = ArrayAttr(
            [AffineMapAttr(identity), AffineMapAttr(sz_map), AffineMapAttr(sz_map), AffineMapAttr(identity)]
        )
        iters = ArrayAttr([linalg_ops.IteratorTypeAttr(linalg_ops.IteratorType.PARALLEL) for _ in range(rank)])

        elem = out_t.element_type
        scale_elem = scale.type.element_type
        # PT2E may request a BF16/FP16 result while retaining an f32 scale. MLIR arithmetic
        # requires same-type operands. Keep same-type captures unchanged; for mixed precision,
        # compute at the wider precision and round exactly once to the result element type.
        # In particular, truncating the scale *before* multiplication would add a rounding
        # absent from the captured quantization expression.
        if elem == scale_elem:
            compute_elem = elem
        elif isinstance(elem, Float64Type) or isinstance(scale_elem, Float64Type):
            compute_elem = f64
        else:
            compute_elem = f32
        body = Block(arg_types=[w.type.element_type, scale.type.element_type, zp.type.element_type, elem])
        wv, sv, zv, _ = body.args
        if (
            kind == "per_channel"
            and w.type.element_type == i8
            and zp.type.element_type == i32
            and elem == f32
            and scale_elem == f32
        ):
            # PyTorch's signed-i8/i32 per-channel contract subtracts after integer promotion.
            # Converting both operands first changes results for large zero points.
            promoted = arith.ExtSIOp(wv, i32)
            difference = arith.SubiOp(promoted.result, zv)
            converted = arith.SIToFPOp(difference.result, compute_elem)
            arithmetic = [promoted, difference, converted]
            unscaled = converted.result
        else:
            wf = arith.SIToFPOp(wv, compute_elem)
            zf = arith.SIToFPOp(zv, compute_elem)
            sub = arith.SubfOp(wf.result, zf.result)
            arithmetic = [wf, zf, sub]
            unscaled = sub.result
        scale_cast = arith.ExtFOp(sv, compute_elem) if scale_elem != compute_elem else None
        mul = arith.MulfOp(unscaled, scale_cast.result if scale_cast is not None else sv)
        result_cast = arith.TruncFOp(mul.result, elem) if elem != compute_elem else None
        body.add_ops(arithmetic)
        if scale_cast is not None:
            body.add_op(scale_cast)
        body.add_op(mul)
        if result_cast is not None:
            body.add_op(result_cast)
        body.add_op(linalg_ops.YieldOp(result_cast.result if result_cast is not None else mul.result))

        generic = linalg_ops.GenericOp(
            inputs=(w, scale, zp),
            outputs=(empty.tensor,),
            body=Region(body),
            indexing_maps=maps,
            iterator_types=iters,
            result_types=(out_t,),
        )
        carry_provenance(generic, op, f"dequant_{kind}")

        block.insert_op_before(empty, op)
        block.insert_op_before(generic, op)
        if on_rewrite is not None:
            on_rewrite(op, (empty, generic), generic)
        op.results[0].replace_all_uses_with(generic.results[0])
        block.detach_op(op)

    for op, kind in quant_rewrites:
        if kind != "per_tensor":  # pragma: no cover - table is exhaustive
            continue
        value, scale, zp = op.operands
        out_t = op.results[0].type
        if not isinstance(out_t, TensorType) or not isinstance(out_t.element_type, IntegerType):
            raise ValueError("quant_ext.quantize_per_tensor must produce an integer tensor")
        rank = len(out_t.get_shape())
        identity = AffineMap.identity(rank)
        scalar_map = AffineMap(rank, 0, ())
        maps = ArrayAttr(
            [AffineMapAttr(identity), AffineMapAttr(scalar_map), AffineMapAttr(scalar_map), AffineMapAttr(identity)]
        )
        iters = ArrayAttr([linalg_ops.IteratorTypeAttr(linalg_ops.IteratorType.PARALLEL) for _ in range(rank)])
        elem = value.type.element_type
        scale_elem = scale.type.element_type
        # quantized_decomposed.quantize_per_tensor promotes half-precision activations to
        # f32 before reciprocal-scale multiplication, round-even, zero point, and clamp.
        # A captured BF16 activation with an f32 scale otherwise produces mixed-type
        # arith.mulf (and rounds the quotient at BF16 precision).
        compute_elem = f64 if isinstance(elem, Float64Type) or isinstance(scale_elem, Float64Type) else f32
        qelem = out_t.element_type
        qmin_attr = op.properties.get("quant_min") or op.attributes.get("quant_min")
        qmax_attr = op.properties.get("quant_max") or op.attributes.get("quant_max")
        if qmin_attr is None or qmax_attr is None:
            raise ValueError("quant_ext.quantize_per_tensor requires quant_min and quant_max")
        qmin = int(qmin_attr.value.data)
        qmax = int(qmax_attr.value.data)

        # TorchAO quantizes with x * (1 / scale), not x / scale. Compute the f32 reciprocal once;
        # besides preserving the exact rounding boundary, this avoids a scalar divide per element
        # on CPU lanes and on targets which leave quantization outside their native dialect.
        inv_t = scale.type
        inv_empty = tensor.EmptyOp((), inv_t)
        inv_body = Block(arg_types=[scale.type.element_type, scale.type.element_type])
        _sv, _unused = inv_body.args
        one = arith.ConstantOp(FloatAttr(1.0, scale.type.element_type))
        reciprocal = arith.DivfOp(one.result, _sv)
        inv_body.add_ops([one, reciprocal, linalg_ops.YieldOp(reciprocal.result)])
        scalar_identity = AffineMap(0, 0, ())
        inv_generic = linalg_ops.GenericOp(
            inputs=(scale,),
            outputs=(inv_empty.tensor,),
            body=Region(inv_body),
            indexing_maps=ArrayAttr([AffineMapAttr(scalar_identity), AffineMapAttr(scalar_identity)]),
            iterator_types=ArrayAttr([]),
            result_types=(inv_t,),
        )
        carry_provenance(inv_generic, op, "quant_reciprocal")

        empty = tensor.EmptyOp((), out_t)
        qmin_c = arith.ConstantOp(FloatAttr(float(qmin), compute_elem))
        qmax_c = arith.ConstantOp(FloatAttr(float(qmax), compute_elem))
        body = Block(arg_types=[elem, scale.type.element_type, zp.type.element_type, qelem])
        xv, inv_sv, zv, _ = body.args
        value_cast = arith.ExtFOp(xv, compute_elem) if elem != compute_elem else None
        scale_cast = arith.ExtFOp(inv_sv, compute_elem) if scale_elem != compute_elem else None
        scaled = arith.MulfOp(
            value_cast.result if value_cast is not None else xv,
            scale_cast.result if scale_cast is not None else inv_sv,
        )
        rounded = mathd.RoundEvenOp(scaled.result)
        zpf = arith.SIToFPOp(zv, compute_elem)
        shifted = arith.AddfOp(rounded.result, zpf.result)
        low = arith.MaximumfOp(shifted.result, qmin_c.results[0])
        high = arith.MinimumfOp(low.result, qmax_c.results[0])
        # Torch's final integer tensor cast maps NaN to zero. LLVM fptosi on
        # NaN is poison: make the conversion operand defined without changing
        # finite rounding, zero-point order or +/-infinity saturation.
        ordered = arith.CmpfOp(high.result, high.result, "ord")
        cast_zero = arith.ConstantOp(FloatAttr(0.0, compute_elem))
        safe = arith.SelectOp(ordered.result, high.result, cast_zero.result)
        converted = arith.FPToSIOp(safe.result, qelem)
        if value_cast is not None:
            body.add_op(value_cast)
        if scale_cast is not None:
            body.add_op(scale_cast)
        body.add_ops([scaled, rounded, zpf, shifted, low, high, ordered, cast_zero, safe,
                      converted, linalg_ops.YieldOp(converted.result)])
        generic = linalg_ops.GenericOp(
            inputs=(value, inv_generic.results[0], zp),
            outputs=(empty.tensor,),
            body=Region(body),
            indexing_maps=maps,
            iterator_types=iters,
            result_types=(out_t,),
        )
        carry_provenance(generic, op, "quant_per_tensor")
        block = op.parent_block()
        for new_op in (inv_empty, inv_generic, empty, qmin_c, qmax_c, generic):
            block.insert_op_before(new_op, op)
        if on_rewrite is not None:
            on_rewrite(op, (inv_empty, inv_generic, empty, qmin_c, qmax_c, generic), generic)
        op.results[0].replace_all_uses_with(generic.results[0])
        block.detach_op(op)
    return len(dequant_rewrites) + len(quant_rewrites)
