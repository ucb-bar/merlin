"""Explicit PyTorch 2.10 CPUBlas BF16 convolution four-partial schedule.

The caller must independently establish the CPUBlas fallback backend (source
metadata alone cannot prove dispatch). This schedule changes f32 addition order,
requires finite intermediates and RNE, and preserves neither exception order nor
unrestricted IEEE semantics. It is never installed in the default pipeline.
PyTorch v2.10.0 aten/src/ATen/native/cpu/BlasKernel.cpp sum/gemm_notrans:
four independent f32 partials, then ((s0+s1)+s2)+s3, then terminal bias and BF16.
Only K divisible by four is supported; the source's tail path is refused.
The scalar IR uses separate strict Mulf/Addf (no reassociation/contract flags).
This is a measured backend schedule, not a promise about all CPU/BLAS dispatches.
Audited BlasKernel.cpp SHA256:
5f55b3a7c17c724e1b5e963f37f248fff1857867faff36d58d3b24da9cdc8363.
The optional recorded test takes MERLIN_BF16_CONV_FIXTURE containing float32
representations of original BF16 weight/columns/bias/output .npy arrays.
"""

from __future__ import annotations

from merlin.targetgen import legacy_labels as LL

CPU_BF16_BLAS_ILP4_POLICY = "torch_2_10_cpu_blas_bf16_ilp4"


def rewrite_cpu_bf16_conv_ilp4(
    op,
    *,
    backend_policy: str,
    source_node_ids: tuple[str, ...],
    allow_reassociation: bool = False,
    assume_finite_intermediates: bool = False,
):
    """Validate and replace one source-bound contraction; return its replacement.

    Raises ValueError without mutation on unsupported input. The caller selects
    the exact operation and supplies its recorded source node IDs. Retained BF16
    ExtF provenance, pure packing, zero init and sole-use terminal bias/TruncF are
    checked structurally. No blanket rewrite of arbitrary f32 contractions occurs.
    """
    import math

    from xdsl.dialects import arith, tensor
    from xdsl.dialects.builtin import (
        AffineMapAttr,
        ArrayAttr,
        FloatAttr,
        IntegerAttr,
        StringAttr,
        TensorType,
        bf16,
        f32,
        i64,
    )
    from xdsl.dialects.linalg import ops as linalg
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineConstantExpr, AffineDimExpr, AffineMap
    from xdsl.rewriter import Rewriter

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    require(backend_policy == CPU_BF16_BLAS_ILP4_POLICY, "explicit pinned CPUBlas BF16 policy required")
    require(allow_reassociation and assume_finite_intermediates, "finite and reassociation obligations required")
    require(bool(source_node_ids), "source node binding required")
    require(op.parent is not None, "attached contraction required")
    require(isinstance(op, (linalg.MatmulOp, linalg.GenericOp)), "matmul contraction required")
    require("merlin.cpu_bf16_conv" not in op.attributes, "already scheduled")
    require(len(op.operands) == 3 and len(op.results) == 1, "two-input tensor contraction required")
    attrs = op.attributes
    require(
        getattr(attrs.get("prov.aten"), "data", None) == "aten.convolution.default", "convolution provenance required"
    )
    require(getattr(attrs.get("prov.orig_dtype"), "data", None) == "bfloat16", "BF16 source dtype required")
    require(
        LL.is_gathered_conv_path(getattr(attrs.get("prov.conv_path"), "data", None)),
        "gathered-window convolution provenance required",
    )
    require(
        tuple(getattr(x, "data", None) for x in attrs.get("prov.source_node_ids", ())) == source_node_ids,
        "source node binding mismatch",
    )

    def same_source(owner):
        require(hasattr(owner, "attributes"), "retained BF16 producer required")
        require(
            all(
                owner.attributes.get(k) == attrs.get(k)
                for k in (
                    "prov.source_node_ids",
                    "prov.origin_node_ids",
                    "prov.orig_dtype",
                    "prov.aten",
                    "prov.conv_path",
                    "prov.fqn",
                )
            ),
            "source provenance changed along contraction chain",
        )

    def tensor_shape(value, element=f32):
        require(
            isinstance(value.type, TensorType) and value.type.element_type == element, "tensor element type mismatch"
        )
        shape = tuple(value.type.get_shape())
        require(all(n > 0 for n in shape), "static positive tensor required")
        return shape

    def body_ops(owner):
        require(isinstance(owner, linalg.GenericOp), "pure generic required")
        require(all(x.data == linalg.IteratorType.PARALLEL for x in owner.iterator_types), "parallel packing required")
        require(len(owner.results) == 1 and len(owner.outputs) == 1, "single tensor result required")
        shape = tuple(owner.results[0].type.get_shape())
        require(owner.indexing_maps.data[-1].data == AffineMap.identity(len(shape)), "identity output map required")
        operations = list(owner.body.block.ops)
        require(bool(operations) and isinstance(operations[-1], linalg.YieldOp), "generic yield required")
        return operations, owner.body.block.args

    def widened(value):
        owner = value.owner
        same_source(owner)
        tensor_shape(value)
        if isinstance(owner, (tensor.CollapseShapeOp, tensor.ExpandShapeOp)):
            widened(owner.operands[0])
            return
        operations, args = body_ops(owner)
        require(len(owner.inputs) == 1, "unary BF16 widening/packing required")
        require(isinstance(operations[-1], linalg.YieldOp), "packing yield required")
        if len(operations) == 1:
            require(tuple(operations[0].operands) == (args[0],), "packing must yield unmodified input")
            widened(owner.inputs[0])
            return
        require(len(operations) == 2 and isinstance(operations[0], arith.ExtFOp), "retained BF16 ExtF required")
        require(
            tuple(operations[0].operands) == (args[0],) and operations[0].result.type == f32,
            "exact BF16 widening required",
        )
        require(tuple(operations[1].operands) == (operations[0].result,), "widening yield mismatch")
        source_shape = tensor_shape(owner.inputs[0], bf16)
        require(
            source_shape == tensor_shape(value)
            and owner.indexing_maps.data[0].data == AffineMap.identity(len(source_shape)),
            "pointwise BF16 widening required",
        )

    lhs, rhs, init = op.operands
    ls, rs, os = map(tensor_shape, (lhs, rhs, init))
    require(len(ls) == len(rs) == len(os) == 2, "rank-two contraction required")
    m, k = ls
    require(rs[0] == k and os == (m, rs[1]) and op.results[0].type == init.type, "matmul shape mismatch")
    n = rs[1]
    require(k % 4 == 0, "K must be divisible by four")
    if isinstance(op, linalg.GenericOp):
        d = [AffineDimExpr(i) for i in range(3)]
        expected = [AffineMap(3, 0, x) for x in ((d[0], d[2]), (d[2], d[1]), (d[0], d[1]))]
        require([x.data for x in op.indexing_maps] == expected, "canonical matmul maps required")
        require(
            [x.data for x in op.iterator_types] == [linalg.IteratorType.PARALLEL] * 2 + [linalg.IteratorType.REDUCTION],
            "canonical matmul iterators required",
        )
    b = op.body.block
    operations = list(b.ops)
    require(
        len(operations) == 3
        and isinstance(operations[0], arith.MulfOp)
        and isinstance(operations[1], arith.AddfOp)
        and isinstance(operations[2], linalg.YieldOp),
        "canonical multiply/add body required",
    )
    require(
        tuple(operations[0].operands) == tuple(b.args[:2])
        and tuple(operations[1].operands) == (operations[0].result, b.args[2])
        and tuple(operations[2].operands) == (operations[1].result,),
        "canonical multiply/add operands required",
    )
    require(not operations[0].fastmath.data and not operations[1].fastmath.data, "strict arithmetic required")
    widened(lhs)
    widened(rhs)
    require(isinstance(init.owner, tensor.SplatOp), "zero splat init required")
    constant = init.owner.operands[0].owner
    require(
        isinstance(constant, arith.ConstantOp) and isinstance(constant.value, FloatAttr), "constant zero init required"
    )
    z = constant.value.value.data
    require(z == 0 and math.copysign(1, z) > 0, "positive zero init required")

    def next_user(value):
        uses = tuple(value.uses)
        require(len(uses) == 1, "sole-use bias/cast epilogue required")
        user = uses[0].operation
        same_source(user)
        return user

    current = op.results[0]
    user = next_user(current)
    while isinstance(user, (tensor.CollapseShapeOp, tensor.ExpandShapeOp)):
        current = user.results[0]
        user = next_user(current)
    operations, args = body_ops(user)
    require(
        len(user.inputs) == 2
        and current in user.inputs
        and len(operations) == 2
        and isinstance(operations[0], arith.AddfOp),
        "terminal bias add required",
    )
    require(
        set(operations[0].operands) == set(args[:2]) and tuple(operations[1].operands) == (operations[0].result,),
        "bias body mismatch",
    )
    bias = user.inputs[1 if user.inputs[0] == current else 0]
    require(tensor_shape(bias) == (m,), "channel bias required")
    widened(bias)
    shape = tensor_shape(current)
    rank = len(shape)
    channel_axes = [
        i for i in range(rank) if shape[i] == m and math.prod(shape[:i]) == 1 and math.prod(shape[i + 1 :]) == n
    ]
    require(bool(channel_axes), "channel-preserving reshape required")
    input_index = 0 if user.inputs[0] == current else 1
    require(
        user.indexing_maps.data[input_index].data == AffineMap.identity(rank)
        and user.indexing_maps.data[1 - input_index].data == AffineMap(rank, 0, (AffineDimExpr(channel_axes[0]),))
        and tensor_shape(user.results[0]) == shape,
        "channel bias maps required",
    )
    require(not operations[0].fastmath.data, "strict bias arithmetic required")
    terminal = next_user(user.results[0])
    operations, args = body_ops(terminal)
    require(
        len(terminal.inputs) == 1
        and len(operations) == 2
        and isinstance(operations[0], arith.TruncFOp)
        and tuple(operations[0].operands) == (args[0],)
        and operations[0].result.type == bf16
        and tuple(operations[1].operands) == (operations[0].result,),
        "terminal BF16 cast required",
    )
    require(
        tensor_shape(terminal.results[0], bf16) == shape
        and terminal.indexing_maps.data[0].data == AffineMap.identity(rank),
        "pointwise terminal cast required",
    )

    # Validation is complete before creating any SSA uses or mutating the block.
    new_ops = []

    def emit(new):
        new.attributes.update(attrs)
        new.attributes["merlin.cpu_bf16_conv"] = StringAttr(CPU_BF16_BLAS_ILP4_POLICY)
        new_ops.append(new)
        return new.results[0]

    def amap(rank, values):
        return AffineMapAttr(AffineMap(rank, 0, tuple(values)))

    def expand(value, shape, groups):
        reassociation = ArrayAttr([ArrayAttr([IntegerAttr(i, i64) for i in g]) for g in groups])
        return emit(tensor.ExpandShapeOp(value, (), reassociation, shape, TensorType(f32, shape)))

    a = expand(lhs, [m, k // 4, 4], [[0], [1, 2]])
    b = expand(rhs, [k // 4, 4, n], [[0, 1], [2]])
    zero = emit(arith.ConstantOp(FloatAttr(0.0, f32)))
    partial_type = TensorType(f32, [m, n, 4])
    fill = emit(tensor.SplatOp(zero, [], partial_type))
    body = Block(arg_types=[f32] * 3)
    mul = arith.MulfOp(body.args[0], body.args[1])
    add = arith.AddfOp(mul.result, body.args[2])
    body.add_ops([mul, add, linalg.YieldOp(add.result)])
    d = [AffineDimExpr(i) for i in range(4)]
    partial = emit(
        linalg.GenericOp(
            inputs=[a, b],
            outputs=[fill],
            body=Region(body),
            indexing_maps=ArrayAttr([amap(4, [d[0], d[3], d[2]]), amap(4, [d[3], d[2], d[1]]), amap(4, d[:3])]),
            iterator_types=ArrayAttr(
                [linalg.IteratorTypeAttr(linalg.IteratorType.PARALLEL)] * 3
                + [linalg.IteratorTypeAttr(linalg.IteratorType.REDUCTION)]
            ),
            result_types=[partial_type],
        )
    )
    body = Block(arg_types=[f32] * 5)
    s = body.args[0]
    for i in range(1, 4):
        add = arith.AddfOp(s, body.args[i])
        body.add_op(add)
        s = add.result
    body.add_op(linalg.YieldOp(s))
    folded = emit(
        linalg.GenericOp(
            inputs=[partial] * 4,
            outputs=[init],
            body=Region(body),
            indexing_maps=ArrayAttr(
                [amap(2, [d[0], d[1], AffineConstantExpr(i)]) for i in range(4)] + [amap(2, d[:2])]
            ),
            iterator_types=ArrayAttr([linalg.IteratorTypeAttr(linalg.IteratorType.PARALLEL)] * 2),
            result_types=[init.type],
        )
    )
    Rewriter.replace_op(op, new_ops, [folded])
    return folded
