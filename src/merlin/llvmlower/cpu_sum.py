"""Explicit Torch AVX2 source order for contiguous last-axis f32 sums.

The source SumKernel.cpp cascade has four groups of eight lanes, chunks of
sixteen groups, and sequential terminal additions. This builder covers the
first two cascade levels (8 <= width < 8192); larger axes are refused.
Caller proof of source dispatch, contiguous logical order, finite intermediate
values and RNE is required. This is never an automatic reassociation policy.
The pinned TorchAO 2.16.0.dev20261104 CPU wheel uses the same K=2048 order.
"""

from __future__ import annotations

CPU_SUM_F32_AVX2_POLICY = "torch_2_10_cpu_sum_f32_avx2"
TORCHAO_20261104_AVX2_POLICY = "torchao_20261104_cpu_sum_f32_avx2"


def build_cpu_sum_lastdim(
    value, *, backend_policy: str, allow_reassociation: bool = False, assume_finite_intermediates: bool = False
):
    """Return detached ops and a tensor with the final axis removed."""
    from xdsl.dialects import arith, scf, tensor
    from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr, FloatAttr, IndexType, TensorType, f32
    from xdsl.dialects.linalg import ops as L
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineMap

    if backend_policy not in (CPU_SUM_F32_AVX2_POLICY, TORCHAO_20261104_AVX2_POLICY):
        raise ValueError("explicit pinned AVX2 source sum policy required")
    if not allow_reassociation or not assume_finite_intermediates:
        raise ValueError("finite intermediates and source reassociation contract required")
    if not isinstance(value.type, TensorType) or value.type.element_type != f32:
        raise TypeError("f32 tensor required")
    shape = value.type.get_shape()
    if not shape or any(n <= 0 for n in shape) or not 8 <= shape[-1] < 8192:
        raise ValueError("positive static dimensions and 8 <= width < 8192 required")
    prefix, n = shape[:-1], shape[-1]
    groups = n // 32
    ops = []

    def generic(output_shape, body):
        typ = TensorType(f32, output_shape)
        empty = tensor.EmptyOp((), typ)
        result = L.GenericOp(
            inputs=(),
            outputs=(empty.tensor,),
            body=Region(body),
            indexing_maps=ArrayAttr([AffineMapAttr(AffineMap.identity(len(output_shape)))]),
            iterator_types=ArrayAttr([L.IteratorTypeAttr(L.IteratorType.PARALLEL)] * len(output_shape)),
            result_types=(typ,),
        )
        ops.extend([empty, result])
        return result.results[0]

    def index(block, number):
        op = arith.ConstantOp.from_int_and_width(number, IndexType())
        block.add_op(op)
        return op.result

    def add(block, a, b):
        op = arith.AddfOp(a, b)
        block.add_op(op)
        return op.result

    body = Block(arg_types=[f32])
    positions = [L.IndexOp(i) for i in range(len(prefix) + 2)]
    zero = arith.ConstantOp(FloatAttr(0.0, f32))
    body.add_ops([*positions, zero])
    prefix_indices = [op.result for op in positions[:-2]]
    eight, thirty_two = index(body, 8), index(body, 32)
    part_offset = arith.MuliOp(positions[-2].result, eight)
    lane_offset = arith.AddiOp(part_offset.result, positions[-1].result)
    body.add_ops([part_offset, lane_offset])

    def chunk_sum(first, last):
        start, end, step = index(body, first), index(body, last), index(body, 1)
        inner = Block(arg_types=[IndexType(), f32])
        offset = arith.MuliOp(inner.args[0], thirty_two)
        at = arith.AddiOp(offset.result, lane_offset.result)
        item = tensor.ExtractOp(value, [*prefix_indices, at.result], f32)
        total = arith.AddfOp(inner.args[1], item.result)
        inner.add_ops([offset, at, item, total, scf.YieldOp(total.result)])
        loop = scf.ForOp(start, end, step, [zero.result], Region(inner))
        body.add_op(loop)
        return loop.results[0]

    carry = zero.result
    for first in range(0, groups - groups % 16, 16):
        carry = add(body, carry, chunk_sum(first, first + 16))
    tail = chunk_sum(groups - groups % 16, groups)
    partial = add(body, tail, carry)
    body.add_op(L.YieldOp(partial))
    partials = generic((*prefix, 4, 8), body)

    body = Block(arg_types=[f32])
    positions = [L.IndexOp(i) for i in range(len(prefix) + 1)]
    body.add_ops(positions)
    prefix_indices = [op.result for op in positions[:-1]]
    values = []
    for part in range(4):
        item = tensor.ExtractOp(partials, [*prefix_indices, index(body, part), positions[-1].result], f32)
        body.add_op(item)
        values.append(item.result)
    for vec in range(groups * 4, n // 8):
        at = arith.AddiOp(index(body, vec * 8), positions[-1].result)
        item = tensor.ExtractOp(value, [*prefix_indices, at.result], f32)
        body.add_ops([at, item])
        values[0] = add(body, values[0], item.result)
    total = values[0]
    for item in values[1:]:
        total = add(body, total, item)
    body.add_op(L.YieldOp(total))
    lanes = generic((*prefix, 8), body)

    body = Block(arg_types=[f32])
    positions = [L.IndexOp(i) for i in range(len(prefix))]
    zero = arith.ConstantOp(FloatAttr(0.0, f32))
    body.add_ops([*positions, zero])
    prefix_indices = [op.result for op in positions]
    total = zero.result
    for tail in range(n // 8 * 8, n):
        item = tensor.ExtractOp(value, [*prefix_indices, index(body, tail)], f32)
        body.add_op(item)
        total = add(body, total, item.result)
    for lane in range(8):
        item = tensor.ExtractOp(lanes, [*prefix_indices, index(body, lane)], f32)
        body.add_op(item)
        total = add(body, total, item.result)
    body.add_op(L.YieldOp(total))
    result = generic(prefix, body)
    return ops, result
