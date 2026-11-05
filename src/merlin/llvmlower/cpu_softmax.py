"""Explicit source-compatible float32 CPU softmax; never enabled by default.

PyTorch2.10 AVX2 uses eight-lane ordered sums, SLEEF exp_u10 and reciprocal
multiplication. This scalar IR preserves that selected numerical policy without
requiring the destination to implement the source vector ISA.
The exponential schedule adapts SLEEF xexpf, revision5a1d179; its Boost
Software License and copyright notice are preserved in NOTICE.
"""

from __future__ import annotations

CPU_SOFTMAX_F32_AVX2_POLICY = "torch_2_10_cpu_softmax_f32_avx2"


def cpu_softmax_sum_order(backend_policy):
    """The pinned source lane count and ordered binary denominator tree."""
    if backend_policy != CPU_SOFTMAX_F32_AVX2_POLICY:
        raise ValueError("explicit pinned f32 softmax policy required")
    return 8, (((0, 4), (2, 6)), ((1, 5), (3, 7)))


def cpu_softmax_row_numerics(*, backend_policy, assume_finite_inputs=False):
    """Expose the existing source algorithm to shared counted-row scheduling.

    This is source numerical data, not target support or qualification. The
    selected exp implementation must realize build_cpu_softmax_exp_nonpositive.
    """
    from merlin.compile.scheduling.row_softmax import RowSoftmaxNumerics

    lanes, tree = cpu_softmax_sum_order(backend_policy)
    return RowSoftmaxNumerics(
        backend_policy,
        "sleef_xexpf_5a1d179_nonpositive_f32",
        "f32",
        lanes,
        tree,
        "reciprocal_multiply",
        assume_finite_inputs,
        True,
    )


def build_cpu_softmax_exp_nonpositive(value, *, backend_policy: str, domain: str):
    """Scalar SLEEF exp_u10 for nonpositive f32, including negative infinity.

    NaN/positive arguments are outside the caller-proven domain. Clamp before
    integer conversion so underflow inputs cannot introduce conversion poison.
    Float status flags are outside this unconstrained floating-point policy.
    """
    from xdsl.dialects import arith, math
    from xdsl.dialects.builtin import FloatAttr, f32, i32

    if backend_policy != CPU_SOFTMAX_F32_AVX2_POLICY:
        raise ValueError("explicit pinned f32 softmax policy required")
    if domain != "nonpositive":
        raise ValueError("nonpositive domain required")
    if value.type != f32:
        raise TypeError("scalar f32 required")
    ops = []

    def emit(op):
        ops.append(op)
        return op.results[0]

    def c(v):
        return emit(arith.ConstantOp(FloatAttr(v, f32)))

    def ci(v):
        return emit(arith.ConstantOp.from_int_and_width(v, 32))

    def add(a, b):
        return emit(arith.AddfOp(a, b))

    def mul(a, b):
        return emit(arith.MulfOp(a, b))

    def fma(a, b, d):
        return emit(math.FmaOp(a, b, d))

    threshold = c(-104.0)
    under = emit(arith.CmpfOp(value, threshold, "olt"))
    safe = emit(arith.SelectOp(under, threshold, value))
    scaled = mul(safe, c(1.44269504088896340736))
    magic = c(12582912.0)
    rounded = emit(arith.SubfOp(add(scaled, magic), magic))
    q = emit(arith.FPToSIOp(rounded, i32))
    reduced = fma(rounded, c(-0.693145751953125), safe)
    reduced = fma(rounded, c(-1.428606765330187045e-6), reduced)
    u = c(0.000198527617612853646278381)
    for coefficient in [
        0.00139304355252534151077271,
        0.00833336077630519866943359,
        0.0416664853692054748535156,
        0.166666671633720397949219,
        0.5,
    ]:
        u = fma(u, reduced, c(coefficient))
    u = add(c(1.0), fma(mul(reduced, reduced), u, reduced))
    half = emit(arith.ShRSIOp(q, ci(1)))
    other = emit(arith.SubiOp(q, half))

    def power(exponent):
        biased = emit(arith.AddiOp(exponent, ci(127)))
        bits = emit(arith.ShLIOp(biased, ci(23)))
        return emit(arith.BitcastOp(bits, f32))

    result = mul(mul(u, power(half)), power(other))
    result = emit(arith.SelectOp(under, c(0.0), result))
    return ops, result


def build_cpu_softmax_lastdim(value, *, backend_policy: str, assume_finite_inputs: bool = False):
    """Return detached ops and source-compatible softmax along the last axis."""
    from xdsl.dialects import arith, scf, tensor
    from xdsl.dialects.builtin import (
        AffineMapAttr,
        ArrayAttr,
        DenseIntOrFPElementsAttr,
        FloatAttr,
        IndexType,
        TensorType,
        f32,
    )
    from xdsl.dialects.linalg import ops as L
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineDimExpr, AffineMap

    if backend_policy != CPU_SOFTMAX_F32_AVX2_POLICY:
        raise ValueError("explicit pinned f32 softmax policy required")
    if not assume_finite_inputs:
        raise ValueError("finite-input contract required")
    if not isinstance(value.type, TensorType) or value.type.element_type != f32:
        raise TypeError("f32 tensor required")
    shape = list(value.type.get_shape())
    rank = len(shape)
    if not shape or any(n <= 0 for n in shape) or shape[-1] < 8:
        raise ValueError("static positive shape and last axis at least eight required")
    n = shape[-1]
    prefix = shape[:-1]
    ops = []

    def maps(*values):
        return ArrayAttr([AffineMapAttr(v) for v in values])

    def generic(inputs, output, body, indexing, reduction=False):
        its = (
            [L.IteratorType.PARALLEL] * (len(shape) - 1) + [L.IteratorType.REDUCTION]
            if reduction
            else [L.IteratorType.PARALLEL] * len(output.type.get_shape())
        )
        return L.GenericOp(
            inputs=inputs,
            outputs=(output,),
            body=Region(body),
            indexing_maps=indexing,
            iterator_types=ArrayAttr([L.IteratorTypeAttr(x) for x in its]),
            result_types=(output.type,),
        )

    projection = AffineMap(rank, 0, tuple(AffineDimExpr(i) for i in range(rank - 1)))
    max_type = TensorType(f32, prefix)
    init = arith.ConstantOp(DenseIntOrFPElementsAttr.from_list(max_type, [float("-inf")]))
    bb = Block(arg_types=[f32, f32])
    mx = arith.MaximumfOp(*bb.args)
    bb.add_ops([mx, L.YieldOp(mx.result)])
    maximum = generic((value,), init.result, bb, maps(AffineMap.identity(rank), projection), True)
    ops.extend([init, maximum])
    empty = tensor.EmptyOp((), value.type)
    bb = Block(arg_types=[f32, f32, f32])
    shift = arith.SubfOp(bb.args[0], bb.args[1])
    exp_ops, ex = build_cpu_softmax_exp_nonpositive(shift.result, backend_policy=backend_policy, domain="nonpositive")
    bb.add_ops([shift, *exp_ops, L.YieldOp(ex)])
    exps = generic(
        (value, maximum.results[0]),
        empty.tensor,
        bb,
        maps(AffineMap.identity(rank), projection, AffineMap.identity(rank)),
    )
    ops.extend([empty, exps])
    lane_count, sum_tree = cpu_softmax_sum_order(backend_policy)
    lane_type = TensorType(f32, [*prefix, lane_count])
    empty = tensor.EmptyOp((), lane_type)
    bb = Block(arg_types=[f32])
    indices = [L.IndexOp(i) for i in range(rank)]
    eight = arith.ConstantOp.from_int_and_width(lane_count, IndexType())
    end = arith.ConstantOp.from_int_and_width(n, IndexType())
    start = arith.AddiOp(indices[-1].result, eight.result)
    first = tensor.ExtractOp(exps.results[0], [*[x.result for x in indices[:-1]], indices[-1].result], f32)
    loop_body = Block(arg_types=[IndexType(), f32])
    item = tensor.ExtractOp(exps.results[0], [*[x.result for x in indices[:-1]], loop_body.args[0]], f32)
    total = arith.AddfOp(loop_body.args[1], item.result)
    loop_body.add_ops([item, total, scf.YieldOp(total.result)])
    loop = scf.ForOp(start.result, end.result, eight.result, [first.result], Region(loop_body))
    bb.add_ops([*indices, eight, end, start, first, loop, L.YieldOp(loop.results[0])])
    lanes = generic((), empty.tensor, bb, maps(AffineMap.identity(rank)))
    ops.extend([empty, lanes])
    empty = tensor.EmptyOp((), max_type)
    bb = Block(arg_types=[f32])
    indices = [L.IndexOp(i) for i in range(rank - 1)]
    bb.add_ops(indices)
    xs = []
    for j in range(lane_count):
        index = arith.ConstantOp.from_int_and_width(j, IndexType())
        item = tensor.ExtractOp(lanes.results[0], [*[x.result for x in indices], index.result], f32)
        bb.add_ops([index, item])
        xs.append(item.result)

    def plus(a, b):
        op = arith.AddfOp(a, b)
        bb.add_op(op)
        return op.result

    def combine(tree):
        return xs[tree] if type(tree) is int else plus(combine(tree[0]), combine(tree[1]))

    total = combine(sum_tree)
    one = arith.ConstantOp(FloatAttr(1.0, f32))
    reciprocal = arith.DivfOp(one.result, total)
    bb.add_ops([one, reciprocal, L.YieldOp(reciprocal.result)])
    denom = generic((), empty.tensor, bb, maps(AffineMap.identity(rank - 1)))
    ops.extend([empty, denom])
    empty = tensor.EmptyOp((), value.type)
    bb = Block(arg_types=[f32, f32, f32])
    normalized = arith.MulfOp(bb.args[0], bb.args[1])
    bb.add_ops([normalized, L.YieldOp(normalized.result)])
    result = generic(
        (exps.results[0], denom.results[0]),
        empty.tensor,
        bb,
        maps(AffineMap.identity(rank), projection, AffineMap.identity(rank)),
    )
    ops.extend([empty, result])
    return ops, result.results[0]
