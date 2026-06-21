"""Lowering pass: ``merlin_iface`` -> ``gemmini`` target dialect.

Implemented as xDSL rewrite patterns driven by a ``PatternRewriteWalker`` (the
framework's standard conversion machinery).  Each interface op is rewritten into
its gemmini-dialect counterpart; ``resident``/``acc`` handle types become
``gemmini.resident_tensor``/``gemmini.accumulator`` and uses are rewired by the
rewriter.
"""
from __future__ import annotations

from xdsl.dialects.builtin import ModuleOp
from xdsl.pattern_rewriter import (GreedyRewritePatternApplier, PatternRewriter,
                                   PatternRewriteWalker, RewritePattern,
                                   op_type_rewrite_pattern)

from .dialects import (AccumulatorType, GCommitOp, GConvOp, GMatmulOp,
                       GMovementOp, GPackOp, GReleaseOp, IfCommitOp, IfConv2dOp,
                       IfEvictOp, IfMatmulOp, IfMovementOp, IfResidentPackOp,
                       IfTensorOp, ResidentTensorType)


class PackPat(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: IfResidentPackOp, rw: PatternRewriter):
        rw.replace_matched_op(GPackOp(
            operands=[op.src], properties={"layout": op.layout},
            result_types=[ResidentTensorType()]))


class MatmulPat(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: IfMatmulOp, rw: PatternRewriter):
        rw.replace_matched_op(GMatmulOp(
            operands=[op.lhs, op.rhs], result_types=[AccumulatorType()]))


class MovementPat(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: IfMovementOp, rw: PatternRewriter):
        rw.replace_matched_op(GMovementOp(
            operands=[op.src], properties={"tname": op.tname},
            result_types=[op.res.type]))


class ConvPat(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: IfConv2dOp, rw: PatternRewriter):
        rw.replace_matched_op(GConvOp(
            operands=[op.ifm, op.weight],
            properties={k: getattr(op, k) for k in (
                "tname", "kernel", "stride", "padding", "dilation",
                "epilogue", "output_dtype", "layout")},
            result_types=[op.res.type]))


class CommitPat(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: IfCommitOp, rw: PatternRewriter):
        props = {"tname": op.tname, "epilogue": op.epilogue,
                 "output_dtype": op.output_dtype}
        if op.acc_scale is not None:
            props["acc_scale"] = op.acc_scale
        rw.replace_matched_op(GCommitOp(
            operands=[op.acc], properties=props, result_types=[op.res.type]))


class EvictPat(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: IfEvictOp, rw: PatternRewriter):
        rw.replace_matched_op(GReleaseOp(operands=[op.handle]))


def lower_to_gemmini(module: ModuleOp) -> ModuleOp:
    """Rewrite a verified merlin_iface module into the gemmini target dialect."""
    walker = PatternRewriteWalker(GreedyRewritePatternApplier([
        PackPat(), MatmulPat(), MovementPat(), ConvPat(), CommitPat(), EvictPat()]))
    walker.rewrite_module(module)
    # leaf tensor decls (merlin_iface.tensor) stay as-is — they are the typed
    # value sources the emitters read shapes from.
    module.verify()
    return module
