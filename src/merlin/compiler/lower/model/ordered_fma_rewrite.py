"""Explicit scheduling of structurally proved, zero-seeded FMA contractions.

Only canonical equally batched matrix indexing and an increasing innermost K
reduction are accepted. Source identity attributes do not select operations.
The caller supplies finite-intermediate and RNE contracts for matched operations.
"""

from __future__ import annotations

import math as scalar_math

from merlin.compiler.lower.model.ordered_fma_matmul import build_ordered_fma_matmul


def _match(op):
    from xdsl.dialects import arith, math, tensor
    from xdsl.dialects.builtin import FloatAttr, TensorType, bf16, f32
    from xdsl.dialects.linalg import ops as linalg
    from xdsl.ir.affine import AffineDimExpr, AffineMap

    if not isinstance(op, linalg.GenericOp) or len(op.inputs) != 2 or len(op.outputs) != 1 or len(op.results) != 1:
        return None
    a, b = op.inputs
    out = op.outputs[0]
    if any(not isinstance(v.type, TensorType) for v in (a, b, out)):
        return None
    dtype = a.type.element_type
    if dtype not in (bf16, f32) or b.type.element_type != dtype or out.type.element_type != f32:
        return None
    ash, bsh, osh = (v.type.get_shape() for v in (a, b, out))
    rank = len(osh)
    if rank < 2 or len(ash) != rank or len(bsh) != rank or any(n <= 0 for n in (*ash, *bsh, *osh)):
        return None
    if op.results[0].type != out.type or ash[:-2] != bsh[:-2] or osh[:-2] != ash[:-2] or osh[-2] != ash[-2]:
        return None
    if tuple(x.data for x in op.iterator_types) != (
        *([linalg.IteratorType.PARALLEL] * rank),
        linalg.IteratorType.REDUCTION,
    ):
        return None
    prefix = tuple(AffineDimExpr(i) for i in range(rank - 2))
    m, n, k = (AffineDimExpr(i) for i in (rank - 2, rank - 1, rank))
    maps = tuple(x.data for x in op.indexing_maps)
    if (
        len(maps) != 3
        or maps[0] != AffineMap(rank + 1, 0, (*prefix, m, k))
        or maps[2] != AffineMap(rank + 1, 0, (*prefix, m, n))
    ):
        return None
    if maps[1] == AffineMap(rank + 1, 0, (*prefix, k, n)):
        transposed = False
        expected_rhs = (*ash[:-2], ash[-1], osh[-1])
    elif maps[1] == AffineMap(rank + 1, 0, (*prefix, n, k)):
        transposed = True
        expected_rhs = (*ash[:-2], osh[-1], ash[-1])
    else:
        return None
    if bsh != expected_rhs:
        return None
    seed = out.owner
    if not isinstance(seed, (tensor.SplatOp, linalg.FillOp)):
        return None
    constant = seed.operands[0].owner
    if (
        not isinstance(constant, arith.ConstantOp)
        or not isinstance(constant.value, FloatAttr)
        or constant.result.type != f32
    ):
        return None
    zero = constant.value.value.data
    if zero != 0.0 or scalar_math.copysign(1.0, zero) != 1.0:
        return None
    if len(op.body.blocks) != 1:
        return None
    body = op.body.block
    ops = list(body.ops)
    if len(body.args) != 3 or not isinstance(ops[-1], linalg.YieldOp):
        return None
    if dtype == bf16:
        if len(ops) != 4 or not all(isinstance(x, arith.ExtFOp) for x in ops[:2]):
            return None
        if (
            ops[0].input is not body.args[0]
            or ops[1].input is not body.args[1]
            or any(x.result.type != f32 for x in ops[:2])
        ):
            return None
        lhs, rhs = ops[0].result, ops[1].result
    else:
        if len(ops) != 2:
            return None
        lhs, rhs = body.args[:2]
    fused = ops[-2]
    if not isinstance(fused, math.FmaOp) or tuple(fused.operands) != (lhs, rhs, body.args[2]):
        return None
    if tuple(ops[-1].arguments) != (fused.result,):
        return None
    return transposed


def rewrite_ordered_fma_contractions(
    module,
    *,
    output_tile: int,
    pre_widen_operands: str = "none",
    assume_rne: bool = False,
    assume_finite_intermediates: bool = False,
):
    """Rewrite proved source FMA generics; no default pipeline invokes this pass.

    No broadcasts, nonzero/negative-zero seeds, changed reduction order or
    additional arithmetic are accepted. BF16 widening selection is explicit;
    f32 operands need no widening. Allocation and conversion remain in emitted
    IR and must be included in measured costs. Source attributes are preserved
    on the result-producing operation without controlling applicability.
    """
    from xdsl.dialects.builtin import bf16
    from xdsl.rewriter import Rewriter

    if type(output_tile) is not int or output_tile <= 0:
        raise ValueError("positive integer output tile required")
    if pre_widen_operands not in ("none", "lhs", "rhs", "both"):
        raise ValueError("pre_widen_operands must be none, lhs, rhs or both")
    if not assume_rne or not assume_finite_intermediates:
        raise ValueError("explicit finite-intermediate and RNE contracts required")
    module.verify()
    matched = [(op, result) for op in module.walk() if (result := _match(op)) is not None]
    counts = {"rewritten": 0, "bf16": 0, "f32": 0, "source_macs": 0}
    for op, transposed in matched:
        dtype = op.inputs[0].type.element_type
        packing = pre_widen_operands if dtype == bf16 else "none"
        ops, result = build_ordered_fma_matmul(
            *op.inputs,
            output_tile=output_tile,
            rhs_transposed=transposed,
            pre_widen_operands=packing,
            accumulation_order="zero_seeded_increasing_k_fma",
            assume_rne=assume_rne,
            assume_finite_intermediates=assume_finite_intermediates,
        )
        result.owner.attributes.update(op.attributes)
        counts["rewritten"] += 1
        counts[str(dtype)] += 1
        counts["source_macs"] += scalar_math.prod(op.results[0].type.get_shape()) * op.inputs[0].type.get_shape()[-1]
        Rewriter.replace_op(op, ops, [result])
    module.verify()
    return counts
