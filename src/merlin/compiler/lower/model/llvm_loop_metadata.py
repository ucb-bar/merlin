"""Explicit upstream LLVM loop choices for xDSL-produced CPU control flow."""

from xdsl.context import Context
from xdsl.dialects import llvm


def disable_loop_unroll(latch):
    """Retain a caller-selected ordinary CPU loop through upstream LLVM.

    The caller identifies the actual latch. No arithmetic, dependence, alias or
    numerical permission is inferred. Existing loop metadata refuses to avoid
    discarding another pass's choices. Upstream translation validates the LLVM
    attribute, which this xDSL version retains as an unregistered attribute.
    """
    if not isinstance(latch, (llvm.BrOp, llvm.CondBrOp)):
        raise ValueError("loop metadata requires an LLVM latch branch")
    if "loop_annotation" in latch.attributes or "loop_annotation" in latch.properties:
        raise ValueError("existing loop metadata requires explicit composition")
    annotation = Context(allow_unregistered=True).get_attr("llvm.loop_annotation")
    latch.attributes["loop_annotation"] = annotation("llvm.loop_annotation", False, False, "unroll = <disable = true>")
