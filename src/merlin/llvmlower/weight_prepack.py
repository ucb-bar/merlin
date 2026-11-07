"""Compatibility imports for provider-owned RVV weight layout prepacking."""

from __future__ import annotations

FEATURE = "prepack_weight_layout"


def _implementation():
    from .backend_spi import load_vector_codegen

    return load_vector_codegen("rvv").weight_prepack_api()


def __getattr__(name: str):
    if name.startswith("__"):
        raise AttributeError(name)
    return getattr(_implementation(), name)


def ensure_registered():
    from .impr_features import get

    get(FEATURE)
    return FEATURE
