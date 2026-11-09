"""Shared deallocation and fusion ordering for native pass pipelines.

Providers supply ordered pass occurrences. This policy expands the explicit
deallocation boundary and checks allocation movement without deduplicating passes.
"""

from __future__ import annotations

import os

SINK_DEALLOC_PASS = "func.func(optimize-allocation-liveness)"
SPLIT_FRESH_DEALLOCS_MARKER = "__merlin_split_fresh_deallocs__"
SLICE_UNUSED_ELEMENTWISE_OUTPUT_MARKER = "__merlin_slice_unused_elementwise_output__"
FUSE_ELEMENTWISE = "func.func(linalg-fuse-elementwise-ops)"
GENERALIZE_NAMED = "func.func(linalg-generalize-named-ops)"


def deallocation_passes(
    *, sink: bool | None = None, split: bool | None = None,
    slice_outputs: bool | None = None, disabled: bool | None = None,
) -> list[str]:
    """Resolve explicit policy values or the retained per-build defaults."""
    if disabled is None:
        disabled = bool(os.environ.get("MERLIN_NO_DEALLOC"))
    if disabled:
        return []
    if split is None:
        split = os.environ.get("MERLIN_SPLIT_FRESH_DEALLOCS", "1") == "1"
    if slice_outputs is None:
        slice_outputs = os.environ.get("MERLIN_SLICE_UNUSED_ELEMENTWISE_OUTPUT", "1") == "1"
    if sink is None:
        from merlin.llvmlower.optional_passes import switched

        sink = switched("sink-deallocs", bool(os.environ.get("MERLIN_SINK_DEALLOCS")))
    passes = ["ownership-based-buffer-deallocation", "buffer-deallocation-simplification",
              "bufferization-lower-deallocations"]
    if split:
        passes.insert(1, SPLIT_FRESH_DEALLOCS_MARKER)
    if sink or split or slice_outputs:
        passes.extend(["canonicalize", SINK_DEALLOC_PASS])
    if slice_outputs:
        passes.append(SLICE_UNUSED_ELEMENTWISE_OUTPUT_MARKER)
    return passes


def reorder_generalize_before_fuse(passes: list[str]) -> list[str]:
    """Move generalization before the first prior fusion, never across a schedule."""
    if GENERALIZE_NAMED not in passes or FUSE_ELEMENTWISE not in passes:
        return list(passes)
    gen = passes.index(GENERALIZE_NAMED)
    fuse = next((i for i, p in enumerate(passes) if p == FUSE_ELEMENTWISE and i < gen), None)
    if fuse is None:
        return list(passes)
    if any("transform-interpreter" in p for p in passes[fuse:gen]):
        raise ValueError(
            "refusing to move linalg-generalize-named-ops across a transform-interpreter: the "
            "schedule matches contractions by NAME and would then match nothing"
        )
    out = [p for i, p in enumerate(passes) if i != gen]
    return [*out[:fuse], GENERALIZE_NAMED, *out[fuse:]]


def splice_passes(
    passes: list[str], *, deallocation: list[str] | None = None,
    generalize_before_fuse: bool | None = None,
    allow_loop_allocs: bool = True,
) -> str:
    """Resolve ordered markers and reject hoisting after ownership derivation."""
    if generalize_before_fuse is None:
        generalize_before_fuse = bool(os.environ.get("MERLIN_GENERALIZE_BEFORE_FUSE"))
    if generalize_before_fuse:
        passes = reorder_generalize_before_fuse(passes)
    out: list[str] = []
    for item in passes:
        if item == "__DEALLOC__":
            out.extend(deallocation_passes() if deallocation is None else deallocation)
        else:
            out.append(item)
    dealloc = next((i for i, p in enumerate(out) if "ownership-based-buffer-deallocation" in p), None)
    if dealloc is not None and allow_loop_allocs:
        out = [
            p[:-1] + " allow-return-allocs-from-loops}"
            if p.startswith("one-shot-bufferize{") and p.endswith("}") and "allow-return-allocs-from-loops" not in p
            else p for p in out
        ]
    if dealloc is not None and any("buffer-hoisting" in p or "buffer-loop-hoisting" in p for p in out[dealloc + 1:]):
        raise ValueError("allocation hoisting invalidates ownership: schedule it before buffer deallocation")
    return ",".join(out)
