"""Explicit per-tensor W8A8 preparation using existing affine/quant_ext lowerings.

Dequant order: cast i32 to f32, widen that value and both f32 scales to f64,
combine the scales and multiply in f64, round once to f32, restore storage.
Finite overflow refuses at runtime. Nonfinite promoted inputs are sanitized
before integer conversion and produce canonical NaN over the contraction.
Subnormal inputs quantize to zero under the unchanged epsilon scale floor.
"""

from __future__ import annotations

from dataclasses import replace

from merlin.llvmlower.torchao_affine import ACTIVATION_QUANT

# Numeric policy, independent of hardware: use the established symmetric reduced range,
# epsilon and reciprocal-first ties-even arithmetic, with one block for the tensor.
SPEC = replace(ACTIVATION_QUANT["int8_dyn_act_int8_weight"], granularity="per_tensor")
POLICIES = ("fp32", "int8_w8a8", "spec_int8")


def closed_float_mac(op) -> bool:
    """Recognize only unflagged multiply/add/yield, never arbitrary reductions."""
    from xdsl.dialects import arith
    from xdsl.dialects.linalg.ops import YieldOp

    if len(op.operands) != 3 or len(op.results) != 1 or not hasattr(op, "body"):
        return False
    if len(op.body.blocks) != 1:
        return False
    block = op.body.block
    operations = list(block.ops)
    if len(block.args) != 3 or len(operations) < 3:
        return False
    mul, add, yielded = operations[-3:]
    if not isinstance(mul, arith.MulfOp) or not isinstance(add, arith.AddfOp) or not isinstance(yielded, YieldOp):
        return False
    if mul.fastmath.data or add.fastmath.data:
        return False
    a, b, c = block.args
    promoted = {a: a, b: b}
    for cast in operations[:-3]:
        if not isinstance(cast, arith.ExtFOp) or cast.operands[0] not in promoted:
            return False
        source = cast.operands[0]
        if promoted[source] is not source or cast.result.type != c.type:
            return False
        promoted[source] = cast.result
    a, b = promoted[a], promoted[b]
    return (
        tuple(mul.operands) in ((a, b), (b, a))
        and tuple(add.operands) in ((c, mul.result), (mul.result, c))
        and tuple(yielded.operands) == (add.result,)
    )


def selected_contractions(module, triples, ranks):
    from merlin.kernels.shapes import observe_contractions, zero_initialised
    from merlin.system.offload import why_not

    for op, shape in observe_contractions(module):
        if any(dtype not in {"f16", "f32", "f64"} for dtype in shape.dtypes):
            continue
        if not closed_float_mac(op) or not zero_initialised(op):
            continue
        candidate = replace(shape, dtypes=("i8", "i8", "i32"))
        if why_not(candidate, triples=triples, ranks=ranks) is None:
            # Avoid overflow even under worst-case quantized operands.
            if shape.reduction[0] * max(abs(SPEC.quant_min), abs(SPEC.quant_max)) ** 2 <= (1 << 31) - 1:
                yield op


def quantize_module(module, selected, *, accumulators=None) -> int:
    """Retain public float ABI and host epilogues; replace selected MACs with integer MACs."""
    from xdsl.dialects import arith, cf, math, tensor
    from xdsl.dialects.builtin import (
        AffineMapAttr,
        ArrayAttr,
        FloatAttr,
        FunctionType,
        IntegerAttr,
        StringAttr,
        TensorType,
        UnregisteredOp,
        f32,
        f64,
        i1,
        i8,
        i32,
    )
    from xdsl.dialects.linalg import ops as L
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineMap
    from xdsl.rewriter import Rewriter

    from merlin.llvmlower.passes_xdsl import lower_quant_ext
    from merlin.llvmlower.torchao_affine import emit_symmetric_qparams

    selected = list(selected)
    source_ordinals = {op: i for i, op in enumerate(module.walk())}
    observed = []
    for op in selected:
        emitted, quantized, scales, invalids = [], [], [], []
        for src in op.operands[:2]:
            if src.type.element_type != f32:
                promoted_type = TensorType(f32, src.type.get_shape())
                promoted_empty = tensor.EmptyOp([], promoted_type)
                promotion = Block(arg_types=[src.type.element_type, f32])
                cast_class = arith.TruncFOp if str(src.type.element_type) == "f64" else arith.ExtFOp
                promoted = cast_class(promotion.args[0], f32)
                promotion.add_ops([promoted, L.YieldOp(promoted.result)])
                rank = len(src.type.get_shape())
                identity = AffineMapAttr(AffineMap.identity(rank))
                promote = L.GenericOp(
                    inputs=[src],
                    outputs=[promoted_empty.tensor],
                    body=Region(promotion),
                    indexing_maps=ArrayAttr([identity, identity]),
                    iterator_types=ArrayAttr([L.IteratorTypeAttr(L.IteratorType.PARALLEL)] * rank),
                    result_types=[promoted_type],
                )
                emitted.extend([promoted_empty, promote])
                src = promote.results[0]
            ops, scale = emit_symmetric_qparams(src, TensorType(f32, []), SPEC)
            emitted.extend(ops)
            # The original amax is NaN/Inf iff any promoted operand is nonfinite.
            # Sanitize both scale and values before FPToSI: never cast NaN/Inf.
            scalar_type = TensorType(f32, [])
            flag_type = TensorType(i1, [])
            safe_empty = tensor.EmptyOp([], scalar_type)
            flag_empty = tensor.EmptyOp([], flag_type)
            guard = Block(arg_types=[f32, f32, i1])
            inf = arith.ConstantOp(FloatAttr(float("inf"), f32))
            epsilon = arith.ConstantOp(FloatAttr(SPEC.eps, f32))
            finite = arith.CmpfOp(guard.args[0], inf.result, "olt")
            safe_scale = arith.SelectOp(finite.result, guard.args[0], epsilon.result)
            guard.add_ops([inf, epsilon, finite, safe_scale, L.YieldOp(safe_scale.result, finite.result)])
            smap = AffineMapAttr(AffineMap(0, 0, ()))
            guarded = L.GenericOp(
                inputs=[scale],
                outputs=[safe_empty.tensor, flag_empty.tensor],
                body=Region(guard),
                indexing_maps=ArrayAttr([smap] * 3),
                iterator_types=ArrayAttr([]),
                result_types=[scalar_type, flag_type],
            )
            scale = guarded.results[0]
            invalids.append(guarded.results[1])  # true means finite
            safe_src_empty = tensor.EmptyOp([], src.type)
            sanitization = Block(arg_types=[f32, f32])
            magnitude = math.AbsFOp(sanitization.args[0])
            inf_x = arith.ConstantOp(FloatAttr(float("inf"), f32))
            zero_x = arith.ConstantOp(FloatAttr(0.0, f32))
            finite_x = arith.CmpfOp(magnitude.result, inf_x.result, "olt")
            safe_x = arith.SelectOp(finite_x.result, sanitization.args[0], zero_x.result)
            sanitization.add_ops([magnitude, inf_x, zero_x, finite_x, safe_x, L.YieldOp(safe_x.result)])
            rank = len(src.type.get_shape())
            identity = AffineMapAttr(AffineMap.identity(rank))
            sanitized = L.GenericOp(
                inputs=[src],
                outputs=[safe_src_empty.tensor],
                body=Region(sanitization),
                indexing_maps=ArrayAttr([identity, identity]),
                iterator_types=ArrayAttr([L.IteratorTypeAttr(L.IteratorType.PARALLEL)] * rank),
                result_types=[src.type],
            )
            emitted.extend([safe_empty, flag_empty, guarded, safe_src_empty, sanitized])
            src = sanitized.results[0]
            zero = arith.ConstantOp(IntegerAttr(0, i32))
            zp = tensor.SplatOp(zero.result, [], TensorType(i32, []))
            quant = UnregisteredOp.with_name("quant_ext.quantize_per_tensor").create(
                operands=[src, scale, zp.result],
                result_types=[TensorType(i8, src.type.get_shape())],
                properties={
                    "quant_min": IntegerAttr(SPEC.quant_min, i32),
                    "quant_max": IntegerAttr(SPEC.quant_max, i32),
                },
            )
            emitted.extend([zero, zp, quant])
            quantized.append(quant.results[0])
            scales.append(scale)
        result_type = op.results[0].type
        acc_type = TensorType(i32, result_type.get_shape())
        zero = arith.ConstantOp(IntegerAttr(0, i32))
        init = tensor.SplatOp(zero.result, [], acc_type)
        body = Block(arg_types=[i8, i8, i32])
        a, b = arith.ExtSIOp(body.args[0], i32), arith.ExtSIOp(body.args[1], i32)
        mul = arith.MuliOp(a.result, b.result)
        add = arith.AddiOp(body.args[2], mul.result)
        body.add_ops([a, b, mul, add, L.YieldOp(add.result)])
        contraction = L.GenericOp(
            inputs=quantized,
            outputs=[init.result],
            body=Region(body),
            indexing_maps=op.get_indexing_maps(),
            iterator_types=op.get_iterator_types(),
            result_types=[acc_type],
        )
        contraction.attributes.update(op.attributes)
        # A captured source group can own multiple MACs (FFT, resize). Catalog binding
        # needs a unique identity for each new contraction, derived from source order.
        region = op.attributes.get("prov.region_id")
        if region is not None:
            contraction.attributes["prov.source_region_id"] = region
        contraction.attributes["prov.region_id"] = StringAttr(f"w8a8_contraction_{source_ordinals[op]}")
        contraction.attributes["merlin.numeric_policy"] = StringAttr("int8_w8a8")
        observed.append(contraction.results[0])
        empty = tensor.EmptyOp([], result_type)
        element = result_type.element_type
        readout = Block(arg_types=[i32, f32, f32, i1, i1, element])
        cast = arith.SIToFPOp(readout.args[0], f32)
        acc64 = arith.ExtFOp(cast.result, f64)
        lhs64 = arith.ExtFOp(readout.args[1], f64)
        rhs64 = arith.ExtFOp(readout.args[2], f64)
        combined = arith.MulfOp(lhs64.result, rhs64.result)
        product = arith.MulfOp(acc64.result, combined.result)
        rounded = arith.TruncFOp(product.result, f32)
        finite_inputs = arith.AndIOp(readout.args[3], readout.args[4])
        nan = arith.ConstantOp(FloatAttr(float("nan"), f32))
        selected_value = arith.SelectOp(finite_inputs.result, rounded.result, nan.result)
        readout.add_ops([cast, acc64, lhs64, rhs64, combined, product, rounded, finite_inputs, nan, selected_value])
        result_value = selected_value.result
        if element != f32:
            cast_class = arith.ExtFOp if str(element) == "f64" else arith.TruncFOp
            restored = cast_class(result_value, element)
            readout.add_op(restored)
            result_value = restored.result
        # Nonfinite inputs follow the declared NaN rule. Finite inputs must have
        # representable f32 dequantization AND representable restored storage.
        checks = []
        for value in (rounded.result, result_value):
            magnitude = math.AbsFOp(value)
            infinity = arith.ConstantOp(FloatAttr(float("inf"), value.type))
            finite_output = arith.CmpfOp(magnitude.result, infinity.result, "olt")
            readout.add_ops([magnitude, infinity, finite_output])
            checks.append(finite_output.result)
        representable = arith.AndIOp(*checks)
        true = arith.ConstantOp(IntegerAttr(1, i1))
        invalid = arith.XOrIOp(finite_inputs.result, true.result)
        allowed = arith.OrIOp(representable.result, invalid.result)
        assertion = cf.AssertOp(allowed.result, "int8_w8a8 dequantized result is not representable")
        readout.add_ops([representable, true, invalid, allowed, assertion, L.YieldOp(result_value)])
        rank = len(result_type.get_shape())
        identity = AffineMapAttr(AffineMap.identity(rank))
        scalar = AffineMapAttr(AffineMap(rank, 0, ()))
        result = L.GenericOp(
            inputs=[contraction.results[0], *scales, *invalids],
            outputs=[empty.tensor],
            body=Region(readout),
            indexing_maps=ArrayAttr([identity, scalar, scalar, scalar, scalar, identity]),
            iterator_types=ArrayAttr([L.IteratorTypeAttr(L.IteratorType.PARALLEL)] * rank),
            result_types=[result_type],
        )
        result.attributes.update({k: v for k, v in op.attributes.items() if k.startswith("prov.")})
        result.attributes["prov.role"] = StringAttr("w8a8_host_dequantize")
        Rewriter.replace_op(op, [*emitted, zero, init, contraction, empty, result], [result.results[0]])
    if accumulators is not None and observed:
        # Diagnostic results are opt-in at the harness boundary; public lowering
        # retains its float ABI unless the caller requests raw readout evidence.
        function = next(op for op in module.walk() if op.name == "func.func" and op.sym_name.data == "forward")
        returned = next(op for op in function.body.block.ops if op.name == "func.return")
        first = len(returned.operands)
        returned.operands = (*returned.operands, *observed)
        function.function_type = FunctionType.from_lists(
            list(function.function_type.inputs), [v.type for v in returned.operands]
        )
        accumulators.extend(dict(index=first + i, shape=list(v.type.get_shape())) for i, v in enumerate(observed))
        from merlin.llvmlower.result_layout import ATTRIBUTE

        if ATTRIBUTE in module.attributes:
            import json

            layout = json.loads(module.attributes[ATTRIBUTE].data)
            layout["results"].extend([None] * len(observed))
            module.attributes[ATTRIBUTE] = StringAttr(json.dumps(layout))
    lower_quant_ext(module)
    if selected:
        module.verify()
    return len(selected)
