"""Compatibility imports for provider-owned RVV im2col pack code."""

from __future__ import annotations

FEATURE = "im2col_panel_pack"


def _implementation():
    from .backend_spi import load_vector_codegen

    return load_vector_codegen("rvv").im2col_pack_api()


def __getattr__(name: str):
    if name.startswith("__"):
        raise AttributeError(name)
    return getattr(_implementation(), name)


def ensure_registered():
    result = _implementation().ensure_registered()
    from .impr_features import get

    get(FEATURE)
    return result
