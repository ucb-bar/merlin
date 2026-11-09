"""Normalize tensor contraction ranks and explicitly selected accumulator precision.

Preserve source ordering, numerical/effect obligations and explicit selection.
"""

from __future__ import annotations

from merlin.compiler.ir.provenance import carry_provenance


def collapse_overrank_matmul(module) -> int:
    """Fix `linalg.matmul` ops with a >2-D LHS + 2-D RHS (model2MLIR emits `aten.linear` on a
    3-D activation as a 2-D-map matmul over 3-D operands, which is invalid linalg). Rebuild
    each as a `linalg.generic` batched matmul with a broadcast RHS, using proper-rank maps:

        out[b.., n] = sum_k lhs[b.., k] * rhs[k, n]

    Iteration dims = (R-1 leading parallel) + N(parallel) + K(reduction). Returns count.
    """
    from xdsl.dialects import arith, tensor
    from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr, FloatAttr, IntegerAttr, IntegerType, TensorType
    from xdsl.dialects.linalg import ops as L
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineMap

    par, red = L.IteratorType.PARALLEL, L.IteratorType.REDUCTION
    targets = [
        op
        for op in module.walk()
        if op.name == "linalg.matmul"
        and isinstance(op.results[0].type, TensorType)
        and len(op.results[0].type.get_shape()) > 2
        and len(op.inputs[1].type.get_shape()) == 2
    ]
    for op in targets:
        lhs, rhs = op.inputs[0], op.inputs[1]
        out_t = op.results[0].type
        et = out_t.element_type
        R = len(out_t.get_shape())  # out rank (== lhs rank)
        ndim = R + 1  # leading(R-1) + N + K
        idm = AffineMap.identity(ndim)
        dims = idm.results
        lead = dims[: R - 1]
        N, K = dims[R - 1], dims[R]
        lhs_map = AffineMap(ndim, 0, (*lead, K))
        rhs_map = AffineMap(ndim, 0, (K, N))
        out_map = AffineMap(ndim, 0, (*lead, N))
        iters = ArrayAttr([L.IteratorTypeAttr(par)] * (R - 1) + [L.IteratorTypeAttr(par), L.IteratorTypeAttr(red)])
        block = op.parent_block()

        empty = tensor.EmptyOp((), out_t)
        zero = arith.ConstantOp(IntegerAttr(0, et) if isinstance(et, IntegerType) else FloatAttr(0.0, et))
        fill = L.FillOp(inputs=[zero.results[0]], outputs=[empty.results[0]], res=[out_t])
        body = Block(arg_types=[et, et, et])
        a, b, acc = body.args
        prod = arith.MulfOp(a, b)
        summ = arith.AddfOp(acc, prod.result)
        body.add_ops([prod, summ, L.YieldOp(summ.result)])
        gen = L.GenericOp(
            inputs=(lhs, rhs),
            outputs=(fill.results[0],),
            body=Region(body),
            indexing_maps=ArrayAttr([AffineMapAttr(lhs_map), AffineMapAttr(rhs_map), AffineMapAttr(out_map)]),
            iterator_types=iters,
            result_types=(out_t,),
        )
        carry_provenance(gen, op, "collapse_overrank_matmul")
        for new_op in (empty, zero, fill, gen):
            block.insert_op_before(new_op, op)
        op.results[0].replace_all_uses_with(gen.results[0])
        block.detach_op(op)
    return len(targets)


def lower_bf16_matmul_f32acc(module) -> int:
    """Rewrite every 16-bit-float (bf16 OR fp16) contraction to accumulate in f32; returns count.

    A half-precision (bf16/fp16) ``linalg.matmul`` accumulates in its 16-bit output type, rounding
    every partial sum to the small mantissa — lossy over long contractions, and for fp16 (5-bit
    exponent, max ~65504) the accumulator also OVERFLOWS to inf on real-LLM-sized reductions.
    Hardware/torch instead accumulate in f32 and round only the final result. This rewrites

        C_bf16 = matmul(A_bf16, B_bf16, init=C0)     # bf16 accumulate

    into a ``linalg.generic`` matmul whose body extends each operand to f32, multiplies and
    accumulates in f32 starting from ``extf(C0)``, followed by a ``truncf`` back to bf16 —
    matching the reference numerics. (``extf``/``truncf`` here are scalar ops inside the
    generic body, so the earlier "extf is tensor-rank-polymorphic" limitation does not apply.)

    Also handles bf16 ``linalg.generic`` contractions (e.g. ``batch_matmul``, ``aten.bmm``);
    these otherwise accumulate in bf16 too (attention scores/context), the dominant precision
    loss on VLAs. Either kind qualifies only when its body is exactly ``yield(acc + a * b)`` over
    its own block arguments; any other body computes something else and is left untouched. The
    op's own indexing maps and iterator types are reused (a ``linalg.matmul`` may carry
    transposing maps); only the accumulation dtype is promoted.
    """
    import math

    from xdsl.dialects import arith, tensor
    from xdsl.dialects.arith import FastMathFlag, FastMathFlagsAttr
    from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr, BFloat16Type, Float16Type, FloatAttr, TensorType, f32
    from xdsl.dialects.linalg import ops as L
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineMap

    # FP-CONTRACTION LICENSE on the accumulate. The mul/add below are a multiply-accumulate; without
    # `contract` LLVM must keep them as a separate fmul + fadd and CANNOT form an FMA, so the RVV
    # backend emits vfwmul.vf + vfadd.vv instead of the single fused vfwmacc.vf -- measured: zero
    # vfwmacc in the emitted fp16 kernel until this flag was set. The f32 path never needed it
    # because linalg.matmul -> vector.contract -> vector.fma already lowers to llvm.fmuladd; a
    # MIXED-precision contraction (16-bit operands, f32 accumulator) cannot use vector.contract
    # (which is same-type only) and so arrives here as explicit mulf/addf.
    #
    # This is a strict-accuracy WIN, not a fast-math relaxation: contraction removes the
    # intermediate rounding of the product, which is exactly the single-rounding semantics the
    # hardware FMA and the f32 reference path already have. No other fast-math flag is set (no
    # reassoc/nnan/ninf), so the reduction order is unchanged and NaN/Inf behavior is preserved.
    _CONTRACT = FastMathFlagsAttr([FastMathFlag.ALLOW_CONTRACT])

    par, red = L.IteratorType.PARALLEL, L.IteratorType.REDUCTION

    def is_half_tensor(t):
        return isinstance(t, TensorType) and isinstance(t.element_type, (BFloat16Type, Float16Type))

    def is_mul_add(block):
        """``yield(acc + a * b)`` over the block's two inputs and accumulator, in either operand order."""
        if len(block.args) != 3 or [o.name for o in block.ops] != ["arith.mulf", "arith.addf", "linalg.yield"]:
            return False
        x, y, acc = block.args
        mul, add, out = block.ops
        return (
            {*mul.operands} == {x, y}
            and {*add.operands} == {acc, mul.results[0]}
            and tuple(out.operands) == (add.results[0],)
        )

    def is_contraction(op):
        """A two-input, one-result 16-bit structured reduction whose body is ``acc + a * b``."""
        return (
            isinstance(op, (L.MatmulOp, L.GenericOp))
            and len(op.inputs) == 2
            and len(op.results) == 1
            and all(is_half_tensor(v.type) for v in (*op.inputs, *op.results))
            and any(it.data == red for it in op.get_iterator_types().data)
            and is_mul_add(op.regions[0].blocks[0])
        )

    def is_zero(value):
        """A splat or fill of a ``+0.0`` constant, so the f32 accumulator can start at ``+0.0``."""
        if not isinstance(value.owner, (tensor.SplatOp, L.FillOp)):
            return False
        constant = value.owner.operands[0].owner
        return (
            isinstance(constant, arith.ConstantOp)
            and isinstance(constant.value, FloatAttr)
            and math.copysign(1.0, constant.value.value.data) == 1.0
            and constant.value.value.data == 0.0
        )

    def convert(value, element, cast):
        """Elementwise ``cast`` of a static tensor into ``element``: ``[tensor.empty, linalg.generic]``."""
        shape = value.type.get_shape()
        result_t = TensorType(element, shape)
        empty = tensor.EmptyOp((), result_t)
        body = Block(arg_types=[value.type.element_type, element])
        converted = cast(body.args[0], element)
        body.add_ops([converted, L.YieldOp(converted.result)])
        identity = AffineMapAttr(AffineMap.identity(len(shape)))
        generic = L.GenericOp(
            inputs=(value,),
            outputs=(empty.results[0],),
            body=Region(body),
            indexing_maps=ArrayAttr([identity, identity]),
            iterator_types=ArrayAttr([L.IteratorTypeAttr(par)] * len(shape)),
            result_types=(result_t,),
        )
        return [empty, generic]

    targets = [op for op in module.walk() if is_contraction(op)]
    for op in targets:
        a, b = op.inputs
        out_t = op.results[0].type
        f32_t = TensorType(f32, out_t.get_shape())
        block = op.parent_block()

        # The f32 accumulator starts from the op's own destination (a bias, a running sum, ...).
        # A +0.0 destination needs no conversion pass: start from an f32 +0.0 fill directly.
        if is_zero(op.outputs[0]):
            empty_f = tensor.EmptyOp((), f32_t)
            zero = arith.ConstantOp(FloatAttr(0.0, f32))
            start = [empty_f, zero, L.FillOp(inputs=[zero.results[0]], outputs=[empty_f.results[0]], res=[f32_t])]
        else:
            start = convert(op.outputs[0], f32, arith.ExtFOp)

        # Preserve each operand's own 16-bit type (bf16 or f16); truncate back to the result's.
        a_e, b_e = a.type.element_type, b.type.element_type
        body = Block(arg_types=[a_e, b_e, f32])
        x, y, acc = body.args
        xf, yf = arith.ExtFOp(x, f32), arith.ExtFOp(y, f32)
        prod = arith.MulfOp(xf.result, yf.result, _CONTRACT)
        summ = arith.AddfOp(acc, prod.result, _CONTRACT)
        body.add_ops([xf, yf, prod, summ, L.YieldOp(summ.result)])
        acc_f32 = L.GenericOp(
            inputs=(a, b),
            outputs=(start[-1].results[0],),
            body=Region(body),
            indexing_maps=op.get_indexing_maps(),
            iterator_types=op.get_iterator_types(),
            result_types=(f32_t,),
        )
        finish = convert(acc_f32.results[0], out_t.element_type, arith.TruncFOp)

        # Tag by the ACTUAL operand width -- this pass covers bf16 and fp16, and a breadcrumb
        # that says "bf16" on an fp16 kernel sends the next reader to the wrong datapath.
        carry_provenance(acc_f32, op, "fp16_f32acc" if isinstance(a_e, Float16Type) else "bf16_f32acc")
        for new_op in (*start, acc_f32, *finish):
            block.insert_op_before(new_op, op)
        op.results[0].replace_all_uses_with(finish[-1].results[0])
        block.detach_op(op)
    return len(targets)
