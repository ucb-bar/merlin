"""Compatibility imports for provider-owned RVV requantization fusion."""

from __future__ import annotations

FEATURE = "fuse_requant_into_contraction"
VEC_FEATURE = "fuse_requant_into_contraction_vec"


def _implementation():
    from .backend_spi import load_vector_codegen

    return load_vector_codegen("rvv").requant_fuse_api()


def __getattr__(name: str):
    if name.startswith("__"):
        raise AttributeError(name)
    return getattr(_implementation(), name)


def ensure_registered():
    from .impr_features import get

    for name in (FEATURE, VEC_FEATURE):
        get(name)
    return FEATURE
