"""Compatibility imports for the RVV provider's accumulator microkernel rewrite.

The executable rewrite and two-stage runner belong to the pinned vector provider.
Scalar compilation does not load this module or that provider.
"""

from __future__ import annotations

SCALARIZE_MARKER = "__merlin_scalarize_a__"


def _backend():
    from .backend_spi import load_vector_codegen

    return load_vector_codegen("rvv")


def rewrite_source() -> str:
    """Return the provider-owned A-scalarization rewrite source."""
    return _backend().accum_microkernel_rewrite_source()


def run_source(*, tag_bmm_tails: bool = False) -> str:
    """Return the provider-owned two-stage runner source."""
    return _backend().accum_microkernel_runner_source(tag_bmm_tails=tag_bmm_tails)
