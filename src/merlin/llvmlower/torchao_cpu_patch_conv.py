"""Opt-in source-order replay of the pinned SpectFormer patch convolution.

The captured im2col matmul reduces in C,H,W order and adds bias afterward.
Torch's pinned oneDNN AVX2 direct convolution seeds the accumulator with bias
and performs fused float32 operations in H,W,C order. The rewrite is limited
to the verified 3x16x16 patch, 256 output channels, and 14x14 output grid.
"""

from __future__ import annotations

import math as pymath

TORCH_SOURCE_COMMIT = "7f24a19c0967db04f2d5ba44c1e56eca489932ec"
MARKER = "merlin.torchao_cpu_patch_conv"
MODULE_MARKER = "merlin.torchao_cpu_patch_conv_applied"


def rewrite_torch216_avx2_patch_conv(module, *, source_commit: str, cpu_capability: str, on_rewrite=None) -> int:
    """Fuse the source bias into one strictly ordered patch contraction.

    This pass is default off. The caller must bind the exact Torch build and
    dispatch, and the captured graph must retain the complete original shape
    and provenance. Other convolutions are left untouched.
    """
    from xdsl.dialects import arith, math, tensor
    from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr, FloatAttr, IntegerAttr, StringAttr, TensorType, f32, i64
    from xdsl.dialects.linalg import ops as linalg
    from xdsl.ir import Block, BlockArgument, Region
    from xdsl.ir.affine import AffineDimExpr, AffineMap
    from xdsl.rewriter import InsertPoint, Rewriter

    from .passes_xdsl import carry_provenance, prune_dead_pure_tensor_ops

    if source_commit != TORCH_SOURCE_COMMIT or cpu_capability != "AVX2":
        raise ValueError("exact Torch 2.16 source commit and AVX2 dispatch required")
    if getattr(module.attributes.get("prov.quantization"), "data", None) != "int8_dyn_act_int8_weight":
        raise ValueError("captured TorchAO W8A8 source required")
    if MODULE_MARKER in module.attributes:
        if getattr(module.attributes[MODULE_MARKER], "data", None) != TORCH_SOURCE_COMMIT:
            raise ValueError("conflicting patch convolution source policy marker")
        return 0

    def attr(op, key):
        return getattr(op.attributes.get(key), "data", None)

    def shape(value, dims):
        return isinstance(value.type, TensorType) and value.type == TensorType(f32, dims)

    def through_reshapes(value):
        chain = []
        while isinstance(value.owner, (tensor.CollapseShapeOp, tensor.ExpandShapeOp)):
            op = value.owner
            if len(list(value.uses)) != 1 or attr(op, "prov.region_id") != "conv_0":
                return None, []
            chain.append(op)
            value = op.operands[0]
        return value, chain

    candidates = [
        op
        for op in module.walk()
        if isinstance(op, linalg.MatmulOp)
        and attr(op, "prov.region_id") == "conv_0"
        and attr(op, "prov.fqn") == "patch_embed.proj"
        and attr(op, "prov.aten") == "aten.convolution.default"
    ]
    if not candidates and not any(
        getattr(op.attributes.get("prov.fqn"), "data", None) == "patch_embed.proj" for op in module.walk()
    ):
        return 0
    if len(candidates) != 1:
        raise ValueError("one uniquely identified patch_embed.proj convolution required")
    mm = candidates[0]
    if (
        len(mm.inputs) != 2
        or len(mm.outputs) != 1
        or len(mm.results) != 1
        or not shape(mm.inputs[0], [256, 768])
        or not shape(mm.inputs[1], [768, 196])
        or not shape(mm.outputs[0], [256, 196])
        or not shape(mm.results[0], [256, 196])
    ):
        raise ValueError("patch convolution matrix geometry or dtype changed")
    weight, panel = mm.inputs
    weight_flat = weight.owner
    panel_flat = panel.owner
    if not isinstance(weight_flat, tensor.ExpandShapeOp) or not isinstance(panel_flat, tensor.ExpandShapeOp):
        raise ValueError("original flattened weight and im2col panel required")
    weight_source, weight_chain = through_reshapes(weight)
    panel_source, panel_chain = through_reshapes(panel)
    if not shape(weight_source, [256, 3, 16, 16]) or not shape(panel_source, [3, 16, 16, 1, 14, 14]):
        raise ValueError("original OIHW weight and C,H,W,N,OH,OW im2col layout required")
    if (
        not isinstance(weight_source, BlockArgument)
        or len(weight_chain) != 2
        or len(panel_chain) != 2
        or not shape(weight_chain[1].results[0], [196608])
        or not shape(panel_chain[1].results[0], [150528])
    ):
        raise ValueError("original contiguous flattening of weight and panel required")
    for chain, rank in ((weight_chain, 4), (panel_chain, 6)):
        if (
            str(chain[0].reassociation) != "[[0 : i64, 1 : i64]]"
            or str(chain[1].reassociation) != "[[" + ", ".join(f"{i} : i64" for i in range(rank)) + "]]"
        ):
            raise ValueError("original contiguous reshape reassociations required")
    if not isinstance(panel_source.owner, linalg.GenericOp):
        raise ValueError("original im2col gather required")
    gather = panel_source.owner
    if attr(gather, "prov.conv_path") != "im2col_matmul":
        raise ValueError("source im2col provenance required")
    d = [AffineDimExpr(i) for i in range(6)]
    expected_gather_maps = (
        AffineMap(6, 0, (d[3], d[0], d[4] * 16 + d[1], d[5] * 16 + d[2])),
        AffineMap.identity(6),
    )
    if (
        len(gather.inputs) != 1
        or not shape(gather.inputs[0], [1, 3, 224, 224])
        or tuple(a.data for a in gather.indexing_maps) != expected_gather_maps
        or tuple(a.data for a in gather.iterator_types) != (linalg.IteratorType.PARALLEL,) * 6
        or len(gather.body.block.ops) != 1
        or not isinstance(gather.body.block.last_op, linalg.YieldOp)
        or tuple(gather.body.block.last_op.arguments) != (gather.body.block.args[0],)
    ):
        raise ValueError("original stride-16 unpadded im2col gather required")
    if len(list(mm.results[0].uses)) != 1:
        raise ValueError("patch matmul must feed one reshape chain")
    zero = mm.outputs[0].owner
    if not isinstance(zero, tensor.SplatOp):
        raise ValueError("positive-zero source accumulator required")
    c = zero.operands[0].owner
    if (
        not isinstance(c, arith.ConstantOp)
        or not isinstance(c.value, FloatAttr)
        or c.value.type != f32
        or c.value.value.data != 0.0
        or pymath.copysign(1.0, c.value.value.data) != 1.0
    ):
        raise ValueError("positive-zero source accumulator required")

    adds = []
    for op in module.walk():
        if not isinstance(op, linalg.GenericOp) or attr(op, "prov.region_id") != "conv_0":
            continue
        if len(op.inputs) != 2 or len(op.outputs) != 1 or len(op.results) != 1:
            continue
        if not shape(op.results[0], [1, 256, 14, 14]) or not shape(op.inputs[1], [256]):
            continue
        root, chain = through_reshapes(op.inputs[0])
        if root is mm.results[0] and len(chain) == 4:
            adds.append(op)
    if len(adds) != 1:
        raise ValueError("one source bias add consuming the unique matmul result required")
    add = adds[0]
    bias = add.inputs[1]
    if not isinstance(bias, BlockArgument) or not shape(bias, [256]):
        raise ValueError("original function-argument f32 bias required")
    body = list(add.body.block.ops)
    if (
        len(body) != 2
        or not isinstance(body[0], arith.AddfOp)
        or not isinstance(body[1], linalg.YieldOp)
        or tuple(body[0].operands) != tuple(add.body.block.args[:2])
        or tuple(body[1].arguments) != (body[0].result,)
    ):
        raise ValueError("source bias add arithmetic changed")
    d4 = [AffineDimExpr(i) for i in range(4)]
    if (
        tuple(a.data for a in add.indexing_maps)
        != (AffineMap.identity(4), AffineMap(4, 0, (d4[1],)), AffineMap.identity(4))
        or tuple(a.data for a in add.iterator_types) != (linalg.IteratorType.PARALLEL,) * 4
    ):
        raise ValueError("source per-channel bias broadcasting layout changed")

    out_type = TensorType(f32, [256, 196])
    empty = tensor.EmptyOp((), out_type)
    mb = Block(arg_types=[f32, f32])
    mb.add_op(linalg.YieldOp(mb.args[0]))
    seed = linalg.GenericOp(
        inputs=(bias,),
        outputs=(empty.tensor,),
        body=Region(mb),
        indexing_maps=ArrayAttr(
            [
                AffineMapAttr(AffineMap(2, 0, (AffineDimExpr(0),))),
                AffineMapAttr(AffineMap.identity(2)),
            ]
        ),
        iterator_types=ArrayAttr([linalg.IteratorTypeAttr(linalg.IteratorType.PARALLEL)] * 2),
        result_types=(out_type,),
    )
    reassoc = ArrayAttr([ArrayAttr([IntegerAttr(i, i64) for i in group]) for group in ([0], [1], [2], [3, 4, 5])])
    panel_view = tensor.CollapseShapeOp(
        operands=[panel_source],
        result_types=[TensorType(f32, [3, 16, 16, 196])],
        properties={"reassociation": reassoc},
    )
    block = Block(arg_types=[f32, f32, f32])
    fused = math.FmaOp(block.args[0], block.args[1], block.args[2])
    block.add_ops([fused, linalg.YieldOp(fused.result)])
    m, n, h, w, ch = (AffineDimExpr(i) for i in range(5))
    ordered = linalg.GenericOp(
        inputs=(weight_source, panel_view.results[0]),
        outputs=(seed.results[0],),
        body=Region(block),
        indexing_maps=ArrayAttr(
            [
                AffineMapAttr(AffineMap(5, 0, (m, ch, h, w))),
                AffineMapAttr(AffineMap(5, 0, (ch, h, w, n))),
                AffineMapAttr(AffineMap(5, 0, (m, n))),
            ]
        ),
        iterator_types=ArrayAttr(
            [linalg.IteratorTypeAttr(linalg.IteratorType.PARALLEL)] * 2
            + [linalg.IteratorTypeAttr(linalg.IteratorType.REDUCTION)] * 3
        ),
        result_types=(out_type,),
    )
    carry_provenance(seed, mm, "torchao_cpu_avx2_patch_conv_bias_seed")
    carry_provenance(ordered, mm, "torchao_cpu_avx2_patch_conv_hwc_fma")
    ordered.attributes[MARKER] = StringAttr(TORCH_SOURCE_COMMIT)
    generated = [empty, seed, panel_view, ordered]
    Rewriter.insert_op(generated, InsertPoint.before(mm))
    if on_rewrite is not None:
        on_rewrite(mm, tuple(generated), ordered)
        on_rewrite(add, (), add.inputs[0].owner)
    mm.results[0].replace_all_uses_with(ordered.results[0])
    mm.parent_block().detach_op(mm)
    add.results[0].replace_all_uses_with(add.inputs[0])
    add.parent_block().detach_op(add)
    prune_dead_pure_tensor_ops(module)
    module.attributes[MODULE_MARKER] = StringAttr(TORCH_SOURCE_COMMIT)
    module.verify()
    return 1
