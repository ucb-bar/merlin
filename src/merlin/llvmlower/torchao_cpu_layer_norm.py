"""Opt-in Torch 2.16 AVX2 f32 width-256 native LayerNorm arithmetic.

The pinned CPU source uses two 16-vector Welford chunks of eight f32 lanes,
then merges the chunks and lanes in a fixed order. The builder emits each
arithmetic step explicitly, including the source FMA sites. No default pass
selects it; callers must bind the exact source CPU commit and AVX2 dispatch.
"""

from __future__ import annotations

import math as pymath

TORCH_SOURCE_COMMIT = "7f24a19c0967db04f2d5ba44c1e56eca489932ec"
POLICY = "torch_2_16_20261104_cpu_layer_norm_f32_avx2_width256"
MARKER = "merlin.torchao_cpu_layer_norm"


def build_torch216_avx2_layer_norm(x, gamma, beta, *, epsilon: float):
    """Return ``(ops, result)`` for a contiguous f32 width-256 LayerNorm."""
    from xdsl.dialects import arith, math, tensor
    from xdsl.dialects.builtin import (
        AffineMapAttr,
        ArrayAttr,
        DenseIntOrFPElementsAttr,
        FloatAttr,
        IntegerAttr,
        StringAttr,
        TensorType,
        f32,
        i64,
    )
    from xdsl.dialects.linalg import ops as linalg
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineConstantExpr, AffineDimExpr, AffineMap

    if not pymath.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("finite positive LayerNorm epsilon required")
    if not isinstance(x.type, TensorType) or x.type.element_type != f32:
        raise TypeError("f32 LayerNorm input required")
    shape = list(x.type.get_shape())
    if len(shape) < 2 or any(d <= 0 for d in shape) or shape[-1] != 256:
        raise ValueError("static contiguous last-axis width 256 required")
    if any(not isinstance(v.type, TensorType) or v.type != TensorType(f32, [256]) for v in (gamma, beta)):
        raise TypeError("f32 width-256 gamma and beta required")
    rows = pymath.prod(shape[:-1])
    ops = []

    def emit(op):
        ops.append(op)
        return op.results[0]

    def amap(rank, axes):
        return AffineMapAttr(AffineMap(rank, 0, tuple(AffineDimExpr(a) for a in axes)))

    def zero(s):
        constant = emit(arith.ConstantOp(FloatAttr(0.0, f32)))
        return emit(tensor.SplatOp(constant, [], TensorType(f32, s)))

    reassociation = ArrayAttr(
        [ArrayAttr([IntegerAttr(i, i64) for i in range(len(shape) - 1)]), ArrayAttr([IntegerAttr(len(shape) - 1, i64)])]
    )
    flat = (
        x
        if len(shape) == 2
        else emit(
            tensor.CollapseShapeOp.build(
                operands=[x],
                properties={"reassociation": reassociation},
                result_types=[TensorType(f32, [rows, 256])],
            )
        )
    )

    def point(inputs, s, body_fn, *, maps=None, outputs=2):
        empties = [emit(tensor.EmptyOp((), TensorType(f32, s))) for _ in range(outputs)]
        block = Block(arg_types=[f32] * (len(inputs) + outputs))
        body_ops, values = body_fn(list(block.args[: len(inputs)]))
        block.add_ops([*body_ops, linalg.YieldOp(*values)])
        maps = maps or [amap(len(s), range(len(s)))] * len(inputs)
        output_maps = [amap(len(s), range(len(s)))] * outputs
        op = linalg.GenericOp(
            inputs=inputs,
            outputs=empties,
            body=Region(block),
            indexing_maps=ArrayAttr([*maps, *output_maps]),
            iterator_types=ArrayAttr([linalg.IteratorTypeAttr(linalg.IteratorType.PARALLEL)] * len(s)),
            result_types=[a.type for a in empties],
        )
        ops.append(op)
        return list(op.results)

    coeff = emit(
        arith.ConstantOp(DenseIntOrFPElementsAttr.from_list(TensorType(f32, [16]), [1.0 / (j + 1) for j in range(16)]))
    )

    def partial(chunk):
        tile = emit(tensor.ExtractSliceOp.from_static_parameters(flat, [0, chunk * 128], [rows, 128]))
        groups = ArrayAttr([ArrayAttr([IntegerAttr(0, i64)]), ArrayAttr([IntegerAttr(1, i64), IntegerAttr(2, i64)])])
        tile = emit(tensor.ExpandShapeOp(tile, (), groups, [rows, 16, 8], TensorType(f32, [rows, 16, 8])))
        mean, m2 = zero([rows, 8]), zero([rows, 8])
        block = Block(arg_types=[f32, f32, f32, f32])
        value, c, old_mean, old_m2 = block.args[:4]
        delta = arith.SubfOp(value, old_mean)
        new_mean = math.FmaOp(c, delta.result, old_mean)
        delta2 = arith.SubfOp(value, new_mean.result)
        new_m2 = math.FmaOp(delta.result, delta2.result, old_m2)
        block.add_ops([delta, new_mean, delta2, new_m2, linalg.YieldOp(new_mean.result, new_m2.result)])
        op = linalg.GenericOp(
            inputs=(tile, coeff),
            outputs=(mean, m2),
            body=Region(block),
            indexing_maps=ArrayAttr([amap(3, [0, 2, 1]), amap(3, [2]), amap(3, [0, 1]), amap(3, [0, 1])]),
            iterator_types=ArrayAttr(
                [linalg.IteratorTypeAttr(linalg.IteratorType.PARALLEL)] * 2
                + [linalg.IteratorTypeAttr(linalg.IteratorType.REDUCTION)]
            ),
            result_types=(mean.type, m2.type),
        )
        ops.append(op)
        return list(op.results)

    lane_mean, lane_m2 = zero([rows, 8]), zero([rows, 8])
    for chunk in range(2):
        new_mean, new_m2 = partial(chunk)

        def merge(values, count=chunk * 16):
            old_mean, old_m2, added_mean, added_m2 = values
            c = arith.ConstantOp(FloatAttr(16.0 / (count + 16), f32))
            n = arith.ConstantOp(FloatAttr(float(count), f32))
            delta = arith.SubfOp(added_mean, old_mean)
            m2_tmp = arith.AddfOp(old_m2, added_m2)
            c_delta = arith.MulfOp(c.result, delta.result)
            m0_delta = arith.MulfOp(delta.result, n.result)
            mean = arith.AddfOp(old_mean, c_delta.result)
            m2 = math.FmaOp(m0_delta.result, c_delta.result, m2_tmp.result)
            return [c, n, delta, m2_tmp, c_delta, m0_delta, mean, m2], [mean.result, m2.result]

        lane_mean, lane_m2 = point([lane_mean, lane_m2, new_mean, new_m2], [rows, 8], merge)

    mean, m2 = zero([rows]), zero([rows])
    for lane in range(8):

        def merge_lane(values, count=lane * 32):
            old_mean, old_m2, added_mean, added_m2 = values
            c = arith.ConstantOp(FloatAttr(32.0 / (count + 32), f32))
            n = arith.ConstantOp(FloatAttr(float(count), f32))
            delta = arith.SubfOp(added_mean, old_mean)
            new_mean = math.FmaOp(c.result, delta.result, old_mean)
            sq = arith.MulfOp(delta.result, delta.result)
            scaled = arith.MulfOp(sq.result, c.result)
            inner = math.FmaOp(scaled.result, n.result, added_m2)
            new_m2 = arith.AddfOp(old_m2, inner.result)
            return [c, n, delta, new_mean, sq, scaled, inner, new_m2], [new_mean.result, new_m2.result]

        select_lane = AffineMapAttr(AffineMap(1, 0, (AffineDimExpr(0), AffineConstantExpr(lane))))
        mean, m2 = point(
            [mean, m2, lane_mean, lane_m2],
            [rows],
            merge_lane,
            maps=[amap(1, [0]), amap(1, [0]), select_lane, select_lane],
        )

    def statistics(values):
        row_mean, row_m2 = values
        divisor = arith.ConstantOp(FloatAttr(256.0, f32))
        eps = arith.ConstantOp(FloatAttr(epsilon, f32))
        one = arith.ConstantOp(FloatAttr(1.0, f32))
        variance = arith.DivfOp(row_m2, divisor.result)
        shifted = arith.AddfOp(variance.result, eps.result)
        root = math.SqrtOp(shifted.result)
        rstd = arith.DivfOp(one.result, root.result)
        return [divisor, eps, one, variance, shifted, root, rstd], [rstd.result]

    rstd = point([mean, m2], [rows], statistics, outputs=1)[0]
    output = emit(tensor.EmptyOp((), TensorType(f32, [rows, 256])))
    block = Block(arg_types=[f32, f32, f32, f32, f32, f32])
    value, g, b, row_mean, row_rstd = block.args[:5]
    centered = arith.SubfOp(value, row_mean)
    normalized = arith.MulfOp(centered.result, row_rstd)
    affine = math.FmaOp(normalized.result, g, b)
    block.add_ops([centered, normalized, affine, linalg.YieldOp(affine.result)])
    result = emit(
        linalg.GenericOp(
            inputs=(flat, gamma, beta, mean, rstd),
            outputs=(output,),
            body=Region(block),
            indexing_maps=ArrayAttr(
                [amap(2, [0, 1]), amap(2, [1]), amap(2, [1]), amap(2, [0]), amap(2, [0]), amap(2, [0, 1])]
            ),
            iterator_types=ArrayAttr([linalg.IteratorTypeAttr(linalg.IteratorType.PARALLEL)] * 2),
            result_types=(output.type,),
        )
    )
    if len(shape) != 2:
        result = emit(tensor.ExpandShapeOp(result, (), reassociation, shape, x.type))
    for op in ops:
        op.attributes[MARKER] = StringAttr(POLICY)
    return ops, result


def rewrite_torch216_avx2_layer_norms(module, *, source_commit: str, cpu_capability: str, on_rewrite=None) -> int:
    """Replace complete captured native LayerNorm regions under a pinned CPU policy.

    Region IDs only delimit each captured operation's decomposition. Eligibility
    follows the operation, dtype, shape, and source numeric contract. A selected
    policy refuses malformed or unsupported native LayerNorm instead of silently
    applying a partial numerical rewrite.
    """
    from xdsl.dialects import arith, math, tensor
    from xdsl.dialects.builtin import FloatAttr, TensorType, f32
    from xdsl.dialects.linalg import ops as linalg
    from xdsl.rewriter import InsertPoint, Rewriter

    from .normalization_reassociation import _positive_zero_splat
    from .passes_xdsl import carry_provenance, prune_dead_pure_tensor_ops

    if source_commit != TORCH_SOURCE_COMMIT or cpu_capability != "AVX2":
        raise ValueError("exact captured Torch CPU commit and AVX2 dispatch required")
    if getattr(module.attributes.get("prov.quantization"), "data", None) != "int8_dyn_act_int8_weight":
        raise ValueError("TorchAO source W8A8 capture scheme required")
    for key, expected in (
        ("prov.torch_source_commit", TORCH_SOURCE_COMMIT),
        ("prov.torch_cpu_capability", "AVX2"),
    ):
        value = getattr(module.attributes.get(key), "data", None)
        if value is not None and value != expected:
            raise ValueError(f"conflicting captured {key} metadata")

    def attr(op, key):
        return getattr(getattr(op, "attributes", {}).get(key), "data", None)

    def generic(value, kind, inputs):
        op = value.owner if hasattr(value, "owner") else value
        if not isinstance(op, linalg.GenericOp) or len(op.inputs) != inputs:
            return None
        body = list(op.regions[0].block.ops)
        if len(body) != 2 or not isinstance(body[0], kind) or not isinstance(body[1], linalg.YieldOp):
            return None
        if list(body[0].operands) != list(op.regions[0].block.args[:inputs]):
            return None
        if body[1].operands[0] != body[0].result:
            return None
        return op

    def through_shapes(value):
        while isinstance(value.owner, (tensor.CollapseShapeOp, tensor.ExpandShapeOp)):
            value = value.owner.operands[0]
        return value

    def splat_constant(value):
        owner = value.owner
        if not isinstance(owner, tensor.SplatOp):
            return None
        constant = owner.operands[0].owner
        if not isinstance(constant, arith.ConstantOp) or not isinstance(constant.value, FloatAttr):
            return None
        return float(constant.value.value.data)

    def reduce_add(op):
        if not isinstance(op, linalg.ReduceOp) or len(op.operands) != 2:
            return False
        body = list(op.regions[0].block.ops)
        return (
            len(body) == 2
            and isinstance(body[0], arith.AddfOp)
            and isinstance(body[1], linalg.YieldOp)
            and list(body[0].operands) == list(op.regions[0].block.args)
            and list(body[1].operands) == [body[0].result]
            and _positive_zero_splat(op.operands[1])
        )

    regions: dict[tuple, list] = {}
    for op in module.walk():
        if attr(op, "prov.aten") == "aten.native_layer_norm.default" and MARKER not in op.attributes:
            if attr(op, "prov.op") != "layer_norm":
                raise ValueError("captured native LayerNorm operation provenance required")
            region_id = attr(op, "prov.region_id")
            if not region_id:
                raise ValueError("captured native LayerNorm region ID required")
            regions.setdefault((op.parent_block(), region_id), []).append(op)
    # model2MLIR's custom linalg.reduce assembly omits the operation attributes
    # when it prints a captured region. Its +0 init splat retains provenance,
    # so recover the reduction's region from that exact operand.
    for op in module.walk():
        if isinstance(op, linalg.ReduceOp) and len(op.operands) == 2:
            init = op.operands[1].owner
            region_id = attr(init, "prov.region_id")
            key = (op.parent_block(), region_id)
            if key in regions and attr(init, "prov.aten") == "aten.native_layer_norm.default":
                if op not in regions[key]:
                    regions[key].append(op)

    rewritten = 0
    for (_, region_id), ops in regions.items():
        reductions = sorted(
            (op for op in ops if isinstance(op, linalg.ReduceOp)),
            key=lambda op: list(op.parent_block().ops).index(op),
        )
        if len(reductions) != 2 or not all(reduce_add(op) for op in reductions):
            raise ValueError(f"{region_id}: complete two-reduction native LayerNorm required")
        first, second = reductions
        x = first.operands[0]
        if not isinstance(x.type, TensorType) or x.type.element_type != f32:
            raise ValueError(f"{region_id}: f32 native LayerNorm input required")
        shape = list(x.type.get_shape())
        if len(shape) < 2 or any(n <= 0 for n in shape) or shape[-1] != 256:
            raise ValueError(f"{region_id}: static width-256 native LayerNorm required")
        if tuple(first.dimensions.iter_values()) != (len(shape) - 1,) or tuple(second.dimensions.iter_values()) != (
            len(shape) - 1,
        ):
            raise ValueError(f"{region_id}: native LayerNorm last-axis reductions required")

        output_candidates = [
            op
            for op in ops
            if isinstance(op, linalg.GenericOp)
            and op.results[0].type == x.type
            and generic(op.results[0], arith.AddfOp, 2) is not None
            and isinstance(op.inputs[1].type, TensorType)
            and len(op.inputs[1].type.get_shape()) == 1
        ]
        if len(output_candidates) != 1:
            raise ValueError(f"{region_id}: unique complete native LayerNorm affine output required")
        output = output_candidates[0]
        scaled = generic(output.inputs[0], arith.MulfOp, 2)
        normalized = generic(scaled.inputs[0], arith.MulfOp, 2) if scaled is not None else None
        centered = generic(normalized.inputs[0], arith.SubfOp, 2) if normalized is not None else None
        root = generic(normalized.inputs[1], math.RsqrtOp, 1) if normalized is not None else None
        square = generic(second.operands[0], arith.MulfOp, 2)
        if (
            scaled is None
            or normalized is None
            or centered is None
            or root is None
            or square is None
            or centered.inputs[0] != x
            or square.inputs[0] != centered.results[0]
            or square.inputs[1] != centered.results[0]
            or scaled.inputs[1].type != TensorType(f32, [256])
            or output.inputs[1].type != TensorType(f32, [256])
        ):
            raise ValueError(f"{region_id}: malformed native LayerNorm dataflow")

        mean_div = generic(through_shapes(centered.inputs[1]), arith.DivfOp, 2)
        # The centered input also feeds the variance square. Trace its second
        # reduction forward through division and shape expansion to epsilon.
        epsilon_add = generic(root.inputs[0], arith.AddfOp, 2)
        if epsilon_add is None:
            raise ValueError(f"{region_id}: native LayerNorm epsilon addition required")
        variance_div = generic(through_shapes(epsilon_add.inputs[0]), arith.DivfOp, 2)
        if (
            mean_div is None
            or variance_div is None
            or mean_div.inputs[0] != first.results[0]
            or variance_div.inputs[0] != second.results[0]
            or splat_constant(mean_div.inputs[1]) != 256.0
            or splat_constant(variance_div.inputs[1]) != 256.0
        ):
            raise ValueError(f"{region_id}: native LayerNorm mean/variance division required")
        epsilon = splat_constant(epsilon_add.inputs[1])
        if epsilon is None or not pymath.isfinite(epsilon) or epsilon <= 0:
            raise ValueError(f"{region_id}: finite positive native LayerNorm epsilon required")

        generated, result = build_torch216_avx2_layer_norm(x, scaled.inputs[1], output.inputs[1], epsilon=epsilon)
        for new in generated:
            carry_provenance(new, output, "torchao_cpu_avx2_layer_norm")
        Rewriter.insert_op(generated, InsertPoint.before(output))
        if on_rewrite is not None:
            on_rewrite(output, tuple(generated), generated[-1])
        output.results[0].replace_all_uses_with(result)
        rewritten += 1
    if rewritten:
        prune_dead_pure_tensor_ops(module)
    return rewritten
