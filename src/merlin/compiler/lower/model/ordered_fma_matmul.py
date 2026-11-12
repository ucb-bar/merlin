"""Independent-output scheduling for explicitly ordered floating contractions.

This builder changes only the interleaving of independent output elements. Each
output starts at positive zero and uses increasing-K f32 FMA. BF16 operands are
widened exactly before multiplication. It grants no reassociation permission.
"""

from __future__ import annotations


def build_ordered_fma_matmul(
    lhs,
    rhs,
    *,
    output_tile: int,
    accumulation_order: str,
    rhs_transposed: bool = False,
    pre_widen_operands: str = "none",
    assume_rne: bool = False,
    assume_finite_intermediates: bool = False,
):
    """Return detached ops and f32 output for static, equally batched tensors.

    Callers must prove source zero initialization, increasing-K fused f32
    accumulation, RNE and finite intermediates. No broadcasting or transpose is
    inferred; ``rhs_transposed`` explicitly selects [..., N, K] storage. Both
    inputs have the same f32 or BF16 element type. Full output
    tiles and the statically known tail use distinct loops, so no out-of-bounds
    load is masked or speculated. Explicit BF16 pre-widening selects ``none``,
    ``lhs``, ``rhs`` or ``both`` and materializes exact f32 copies once before
    the contraction; its allocation and conversion costs
    belong to the caller's performance accounting. This is opt-in; no default
    pipeline calls it.
    """
    from xdsl.dialects import arith, math, scf, tensor
    from xdsl.dialects.builtin import FloatAttr, IndexType, TensorType, bf16, f32
    from xdsl.ir import Block, Region

    if accumulation_order != "zero_seeded_increasing_k_fma" or not assume_rne or not assume_finite_intermediates:
        raise ValueError("explicit zero-seeded ordered-FMA, finite-intermediate and RNE contracts required")
    if isinstance(output_tile, bool) or not isinstance(output_tile, int) or output_tile < 1:
        raise ValueError("positive integer output tile required")
    if any(not isinstance(v.type, TensorType) for v in (lhs, rhs)):
        raise TypeError("tensor operands required")
    dtype = lhs.type.element_type
    if dtype not in (f32, bf16) or rhs.type.element_type != dtype:
        raise TypeError("matching f32 or BF16 operand types required")
    if pre_widen_operands not in ("none", "lhs", "rhs", "both"):
        raise ValueError("pre_widen_operands must be none, lhs, rhs or both")
    if pre_widen_operands != "none" and dtype != bf16:
        raise ValueError("pre-widening requires BF16 operands")
    ls, rs = lhs.type.get_shape(), rhs.type.get_shape()
    if len(ls) < 2 or len(ls) != len(rs) or any(n <= 0 for n in (*ls, *rs)):
        raise ValueError("equal static positive operand ranks at least two required")
    if not isinstance(rhs_transposed, bool):
        raise TypeError("rhs_transposed must be a bool")
    rhs_k, rhs_n = (rs[-1], rs[-2]) if rhs_transposed else (rs[-2], rs[-1])
    if ls[:-2] != rs[:-2] or ls[-1] != rhs_k:
        raise ValueError("matching batch and contraction dimensions required")
    shape = (*ls[:-2], ls[-2], rhs_n)
    output_type = TensorType(f32, shape)
    empty = tensor.EmptyOp((), output_type)
    zero = arith.ConstantOp(FloatAttr(0.0, f32))
    constants = {}
    for n in (
        0,
        1,
        ls[-1],
        output_tile,
        *shape,
        (shape[-1] // output_tile) * output_tile,
        *range(min(output_tile, shape[-1])),
    ):
        if n not in constants:
            constants[n] = arith.ConstantOp.from_int_and_width(n, IndexType())

    def c(n):
        return constants[n].result

    packed_ops = []
    if pre_widen_operands != "none":

        def widen_operand(source):
            dimensions = source.type.get_shape()
            packed_type = TensorType(f32, dimensions)
            packed = tensor.EmptyOp((), packed_type)

            def copy(parent, depth, indices, destination):
                if depth == len(dimensions):
                    element = tensor.ExtractOp(source, indices, bf16)
                    wide = arith.ExtFOp(element.result, f32)
                    insert = tensor.InsertOp(wide.result, destination, indices)
                    parent.add_ops([element, wide, insert])
                    return insert.result
                body = Block(arg_types=[IndexType(), packed_type])
                value = copy(body, depth + 1, [*indices, body.args[0]], body.args[1])
                body.add_op(scf.YieldOp(value))
                loop = scf.ForOp(c(0), c(dimensions[depth]), c(1), [destination], Region(body))
                parent.add_op(loop)
                return loop.results[0]

            root = Block()
            value = copy(root, 0, [], packed.tensor)
            return [packed, *[root.detach_op(op) for op in list(root.ops)]], value

        if pre_widen_operands in ("lhs", "both"):
            lhs_ops, lhs = widen_operand(lhs)
            packed_ops.extend(lhs_ops)
        if pre_widen_operands in ("rhs", "both"):
            rhs_ops, rhs = widen_operand(rhs)
            packed_ops.extend(rhs_ops)

    def tile(parent, indices, column, width, destination):
        inner = Block(arg_types=[IndexType(), *([f32] * width)])
        a = tensor.ExtractOp(lhs, [*indices, inner.args[0]], lhs.type.element_type)
        inner.add_op(a)
        av = a.result
        if lhs.type.element_type == bf16:
            wide = arith.ExtFOp(av, f32)
            inner.add_op(wide)
            av = wide.result
        columns = []
        for lane in range(width):
            if lane:
                offset = arith.AddiOp(column, c(lane))
                parent.add_op(offset)
                columns.append(offset.result)
            else:
                columns.append(column)
        totals = []
        for lane, col in enumerate(columns):
            rhs_indices = [col, inner.args[0]] if rhs_transposed else [inner.args[0], col]
            b = tensor.ExtractOp(rhs, [*indices[:-1], *rhs_indices], rhs.type.element_type)
            inner.add_op(b)
            bv = b.result
            if rhs.type.element_type == bf16:
                wide = arith.ExtFOp(bv, f32)
                inner.add_op(wide)
                bv = wide.result
            fused = math.FmaOp(av, bv, inner.args[lane + 1])
            inner.add_op(fused)
            totals.append(fused.result)
        inner.add_op(scf.YieldOp(*totals))
        reduction = scf.ForOp(c(0), c(ls[-1]), c(1), [zero.result] * width, Region(inner))
        parent.add_op(reduction)
        for col, value in zip(columns, reduction.results):
            insert = tensor.InsertOp(value, destination, [*indices, col])
            parent.add_op(insert)
            destination = insert.result
        return destination

    def nest(parent, depth, indices, destination):
        if depth == len(shape) - 1:
            end = (shape[-1] // output_tile) * output_tile
            if end:
                body = Block(arg_types=[IndexType(), output_type])
                value = tile(body, indices, body.args[0], output_tile, body.args[1])
                body.add_op(scf.YieldOp(value))
                loop = scf.ForOp(c(0), c(end), c(output_tile), [destination], Region(body))
                parent.add_op(loop)
                destination = loop.results[0]
            remainder = shape[-1] - end
            if remainder:
                destination = tile(parent, indices, c(end), remainder, destination)
            return destination
        body = Block(arg_types=[IndexType(), output_type])
        value = nest(body, depth + 1, [*indices, body.args[0]], body.args[1])
        body.add_op(scf.YieldOp(value))
        loop = scf.ForOp(c(0), c(shape[depth]), c(1), [destination], Region(body))
        parent.add_op(loop)
        return loop.results[0]

    root = Block()
    result = nest(root, 0, [], empty.tensor)
    ops = [root.detach_op(op) for op in list(root.ops)]
    return [empty, zero, *constants.values(), *packed_ops, *ops], result
