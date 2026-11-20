"""Tensor-source parsing with named batch contractions, reusing the shared parser."""

from .linalg_mlir import (
    layout_bufferization_dialect as layout_bufferization_dialect,
    layout_to_buffer_type as layout_to_buffer_type,
    make_context as _make_context,
    parse_mlir_text as _parse_mlir_text,
    strip_paren_results as strip_paren_results,
)


def make_context():
    """Extend the canonical layout/float-aware context with named batch matmul."""
    from .linalg_batch_matmul import BatchMatmulOp

    ctx = _make_context()
    ctx.load_op(BatchMatmulOp)
    return ctx


def parse_mlir_text(text: str, ctx=None):
    """Parse a tensor source using the canonical exact-bit float parser and lock."""
    return _parse_mlir_text(text, ctx=make_context() if ctx is None else ctx)
