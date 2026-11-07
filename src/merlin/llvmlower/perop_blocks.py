"""Compatibility imports for provider-owned RVV perop blocks code."""

from __future__ import annotations

from .compiler_api import par_tag_for as par_tag_for

CONV_ARM_FEATURE = "conv_register_block"


def _implementation():
    from .backend_spi import load_vector_codegen

    return load_vector_codegen("rvv").perop_blocks_api()


def __getattr__(name: str):
    if name.startswith("__"):
        raise AttributeError(name)
    return getattr(_implementation(), name)


def ensure_registered():
    return _implementation().ensure_registered()
