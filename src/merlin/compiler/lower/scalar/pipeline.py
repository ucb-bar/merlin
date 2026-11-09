"""Scalar Linalg lowering before shared LLVM finalization.

This module owns the scalar computation pipeline, not host dispatch or execution.
The caller expands ``__DEALLOC__`` after selecting its allocation-lifetime policy
and applies explicitly selected optimization features before invocation. No tool,
runtime, backend loader, or compiler driver is imported here.
"""

from __future__ import annotations

from merlin.compiler.lower.llvm.passes import finalization_passes


def scalar_passes(*, parallel: bool = False) -> list[str]:
    """Return the scalar or OpenMP pass sequence without choosing a target ISA.

    Serial output allocations can be hoisted to caller storage. The parallel
    route retains heap intermediates: buffer hoisting can create alloca scopes
    that become invalid multi-block regions during OpenMP conversion. Reduction
    dimensions stay sequential; only parallel Linalg iterators are work-shared.
    """
    passes = [
        "canonicalize",
        "cse",
        "func.func(linalg-fuse-elementwise-ops)",
        "func.func(linalg-generalize-named-ops)",
        "one-shot-bufferize{bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map}",
    ]
    if parallel:
        passes.extend(
            [
                "buffer-results-to-out-params{modify-public-functions}",
                "__DEALLOC__",
                "func.func(convert-linalg-to-parallel-loops)",
                "convert-scf-to-openmp",
                "canonicalize",
            ]
        )
    else:
        passes.extend(
            [
                # Expose invariant carried buffers before forwarding results.
                "canonicalize",
                "buffer-results-to-out-params{modify-public-functions hoist-static-allocs}",
                "func.func(buffer-hoisting,buffer-loop-hoisting)",
                # Ownership must be derived after the final allocation move.
                "__DEALLOC__",
                "func.func(convert-linalg-to-loops)",
            ]
        )
    passes.extend(
        [
            "convert-scf-to-cf",
            "expand-strided-metadata",
            "lower-affine",
            *finalization_passes(openmp=parallel),
        ]
    )
    return passes
