"""xDSL interface-to-Gemmini target rewrite pass."""

from __future__ import annotations

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Operation
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriter, PatternRewriteWalker, RewritePattern

from xdsl_dialects import gemmini as tgt
from xdsl_dialects import merlin_iface as src


class ConvertPattern(RewritePattern):
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter, /) -> None:
        attrs = dict(op.attributes)
        if isinstance(op, src.TensorOp):
            new = tgt.TensorOp.build(attributes=attrs, result_types=[op.result.type])
        elif isinstance(op, src.ResidentPackOp):
            typ = tgt.ResidentTensorType(op.src.type)
            new = tgt.PackOp.build(operands=[op.src], attributes=attrs, result_types=[typ])
        elif isinstance(op, src.MatmulOp):
            typ = tgt.AccumulatorType(op.result.type.element_type)
            new = tgt.MatmulOp.build(operands=list(op.operands), attributes=attrs, result_types=[typ])
        elif isinstance(op, src.CommitOp):
            new = tgt.CommitOp.build(operands=list(op.operands), attributes=attrs, result_types=[op.result.type])
        elif isinstance(op, src.EvictOp):
            new = tgt.ReleaseOp.build(operands=list(op.operands), attributes=attrs)
        elif isinstance(op, src.MovementOp):
            new = tgt.MovementOp.build(operands=list(op.operands), attributes=attrs, result_types=[op.result.type])
        elif isinstance(op, src.Conv2DOp):
            new = tgt.Conv2DOp.build(operands=list(op.operands), attributes=attrs, result_types=[op.result.type])
        elif isinstance(op, src.AttentionQKOp):
            new = tgt.AttentionQKOp.build(operands=list(op.operands), attributes=attrs, result_types=[op.result.type])
        else:
            return
        rewriter.replace_matched_op(new)


class ConvertIfaceToGemminiPass(ModulePass):
    name = "convert-iface-to-gemmini"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(ConvertPattern(), walk_regions_first=False).rewrite_module(op)
        op.verify()

