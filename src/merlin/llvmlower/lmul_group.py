"""Compatibility imports for provider-owned RVV lmul group code."""

from __future__ import annotations


def _implementation():
    from .backend_spi import load_vector_codegen

    return load_vector_codegen("rvv").lmul_group_api()


def __getattr__(name: str):
    if name.startswith("__"):
        raise AttributeError(name)
    return getattr(_implementation(), name)
