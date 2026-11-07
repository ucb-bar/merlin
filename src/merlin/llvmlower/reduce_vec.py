"""Compatibility imports for provider-owned RVV reduction vectorization."""

from __future__ import annotations

FEATURE = "vectorize_amax_reduction"


def _implementation():
    from .backend_spi import load_vector_codegen

    return load_vector_codegen("rvv").reduce_vec_api()


def __getattr__(name: str):
    if name.startswith("__"):
        raise AttributeError(name)
    if name in {"reduction_arms", "splice_reduction_arms"}:
        from .backend_spi import load_vector_codegen

        return getattr(load_vector_codegen("rvv").reduction_schedule_api(), name)
    return getattr(_implementation(), name)


def ensure_registered():
    from .impr_features import get

    get(FEATURE)
    return FEATURE
