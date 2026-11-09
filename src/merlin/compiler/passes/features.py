"""Shared optional edits of upstream pass lists; declarations live in selection.shared."""

from __future__ import annotations

import os

FUSE_AFTER_GENERALIZE_NAME = "fuse_elementwise_after_generalize"


def _fuse_after_generalize(passes: list[str]) -> list[str]:
    """Insert a second ``linalg-fuse-elementwise-ops`` immediately after generalization."""
    out = list(passes)
    anchor = "func.func(linalg-generalize-named-ops)"
    try:
        i = out.index(anchor)
    except ValueError:
        raise ValueError(
            f"{FUSE_AFTER_GENERALIZE_NAME}: anchor {anchor!r} not in the pipeline; refusing to guess "
            f"where the second fusion belongs"
        ) from None
    out.insert(i + 1, "func.func(linalg-fuse-elementwise-ops)")
    return out


PROMOTE_STACK_NAME = "promote_buffers_to_stack"


PROMOTE_STACK_BYTES_ENV = "MERLIN_PROMOTE_STACK_BYTES"


_PROMOTE_STACK_DEFAULT_BYTES = 16384


def _promote_stack_bytes() -> int:
    """Per-buffer promotion cap in bytes (env-overridable so the search can size it per model)."""
    raw = os.environ.get(PROMOTE_STACK_BYTES_ENV)
    if raw:
        try:
            v = int(raw)
        except ValueError:
            v = 0
        if v > 0:
            return v
    return _PROMOTE_STACK_DEFAULT_BYTES


def _promote_buffers_to_stack(passes: list[str], cap: int | None = None) -> list[str]:
    """Insert ``promote-buffers-to-stack`` right after the buffer hoisting.

    After hoisting (so a buffer already lifted out of a loop is promoted once, not per iteration) and
    before the loop lowering. Deallocation runs earlier in this pipeline (``__DEALLOC__``), and a
    stack buffer needs none -- promoting after it is what keeps the two consistent.

    ``cap`` names the per-buffer byte cap explicitly (the sized variants minted by
    :func:`ensure_promote_stack`); ``None`` keeps the env-or-default behaviour.
    """
    out = list(passes)
    anchor = "func.func(buffer-hoisting,buffer-loop-hoisting)"
    try:
        i = out.index(anchor)
    except ValueError:  # pipeline shape changed -> fail closed rather than insert somewhere wrong
        raise ValueError(
            f"{PROMOTE_STACK_NAME}: anchor {anchor!r} not in the pipeline; refusing to guess where "
            f"promote-buffers-to-stack belongs"
        ) from None
    nbytes = _promote_stack_bytes() if cap is None else int(cap)
    out.insert(i + 1, f"func.func(promote-buffers-to-stack{{max-alloc-size-in-bytes={nbytes}}})")
    return out


FUSE_ELEMENTWISE_NAME = "fuse_elementwise_post_contraction"


FUSE_ELEMENTWISE_STAGE: tuple[str, ...] = ("func.func(linalg-fuse-elementwise-ops)", "canonicalize", "cse")


FUSE_ELEMENTWISE_ANCHOR = "func.func(linalg-generalize-named-ops)"


def _fuse_elementwise_pipeline(passes: list[str]) -> list[str]:
    """Splice :data:`FUSE_ELEMENTWISE_STAGE` in just before :data:`FUSE_ELEMENTWISE_ANCHOR`.

    Two refusals, both fail-closed, because "the feature was enabled and changed nothing" is the
    failure mode this repo keeps re-learning:

    * the stage already being present means ``MERLIN_FUSE_POST`` put it there, so this returns the
      list unchanged rather than fusing twice (which is not merely wasteful -- a second
      ``canonicalize``/``cse`` pair changes the emitted code, so a manual A/B and a feature-driven one
      would not be comparing the same build);
    * the anchor being absent means this is not a pipeline the stage belongs in, and inserting at a
      guessed index would put the fusion on the wrong side of the transform interpreter. Raise, so the
      build fails loudly instead of quietly emitting the baseline while reporting the feature applied.
    """
    if FUSE_ELEMENTWISE_STAGE[0] in passes:
        return list(passes)
    if FUSE_ELEMENTWISE_ANCHOR not in passes:
        raise ValueError(
            f"{FUSE_ELEMENTWISE_NAME} was requested but the pass list carries no "
            f"{FUSE_ELEMENTWISE_ANCHOR!r} to anchor the fusion stage against, so there is no position "
            f"that is provably after the transform interpreter and before bufferization. Refusing to "
            f"insert at a guessed index and report the feature as applied."
        )
    at = passes.index(FUSE_ELEMENTWISE_ANCHOR)
    return [*passes[:at], *FUSE_ELEMENTWISE_STAGE, *passes[at:]]


def _reuse_tensor_destination(passes: list[str]) -> list[str]:
    """Run upstream destination forwarding after fusion, before bufferization.

    Tensor semantics and upstream use/alias analysis decide whether an empty
    producer destination can be replaced by the eventual insert-slice view.
    No shape, model name, floating arithmetic, or device ABI is rewritten here.
    """
    out = list(passes)
    for i, item in enumerate(out):
        if item.startswith("one-shot-bufferize"):
            if i and out[i - 1] == "eliminate-empty-tensors":
                return out
            out.insert(i, "eliminate-empty-tensors")
            return out
    return out
