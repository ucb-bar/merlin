"""Shared terminal MLIR-to-LLVM conversion for scalar and extension pipelines.

Backends own optimization and vector/parallel scheduling before this boundary. This module
owns the conversion tail, including the optional vector and OpenMP dialect conversions, and
binds the selected target's index width to every LLVM converter. It imports no backend or
compiler driver, so an out-of-tree provider can compose the same suffix.
"""

from __future__ import annotations

COMMON_FINALIZATION_PASSES = (
    "convert-math-to-libm",
    "convert-math-to-llvm",
    "convert-index-to-llvm",
    "convert-arith-to-llvm",
    "finalize-memref-to-llvm",
    "convert-func-to-llvm",
    "convert-cf-to-llvm",
    "reconcile-unrealized-casts",
    "canonicalize",
    "cse",
    "symbol-dce",
)
VECTOR_FINALIZATION_PASSES = ("convert-vector-to-llvm", "convert-ub-to-llvm")
OPENMP_FINALIZATION_PASS = "convert-openmp-to-llvm"
INDEX_CONVERSION_PASSES = (
    "convert-index-to-llvm",
    "convert-arith-to-llvm",
    "finalize-memref-to-llvm",
    "convert-func-to-llvm",
    "convert-cf-to-llvm",
)


def finalization_passes(*, vector: bool = False, openmp: bool = False) -> tuple[str, ...]:
    """Return the terminal pass suffix for a backend's selected dialects."""
    return (
        *(VECTOR_FINALIZATION_PASSES if vector else ()),
        *((OPENMP_FINALIZATION_PASS,) if openmp else ()),
        *COMMON_FINALIZATION_PASSES,
    )


def append_finalization(
    passes: list[str] | tuple[str, ...], *, vector: bool = False, openmp: bool = False
) -> list[str]:
    """Append the shared suffix without changing the caller's optimization pass list."""
    return [*passes, *finalization_passes(vector=vector, openmp=openmp)]


def bind_index_width(pipeline: str, index_bits: int) -> str:
    """Bind every LLVM index converter, refusing missing or preconfigured passes."""
    if type(index_bits) is not int or index_bits <= 0:
        raise ValueError("selected index width must be a positive integer")
    selected = pipeline
    for name in INDEX_CONVERSION_PASSES:
        positions = []
        start = 0
        while (at := selected.find(name, start)) >= 0:
            before = at - 1
            while before >= 0 and selected[before].isspace():
                before -= 1
            end = at + len(name)
            while end < len(selected) and selected[end].isspace():
                end += 1
            if (before < 0 or selected[before] in "(,") and (end == len(selected) or selected[end] in ",){"):
                positions.append((at, end))
            start = at + len(name)
        if len(positions) != 1 or selected[positions[0][1] :].startswith("{"):
            raise ValueError(f"selected pipeline has no unique bare {name} pass")
        at, _end = positions[0]
        selected = selected[:at] + f"{name}{{index-bitwidth={index_bits}}}" + selected[at + len(name) :]
    return selected


def finalize_pipeline(pipeline: str, *, index_bits: int | None = None) -> str:
    """Validate a built-in backend's terminal conversion and bind its target index width.

    A user-supplied partial diagnostic pipeline may call :func:`bind_index_width` directly. Normal
    scalar, RVV, and chip-composed pipelines must pass this check before the runner is launched.
    """
    cursor = 0
    for name in COMMON_FINALIZATION_PASSES:
        at = pipeline.find(name, cursor)
        if at < 0:
            raise ValueError(f"selected backend pipeline lacks ordered LLVM finalization pass {name}")
        cursor = at + len(name)
    return bind_index_width(pipeline, index_bits) if index_bits is not None else pipeline
