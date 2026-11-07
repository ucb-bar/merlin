"""Compatibility imports for provider-owned RVV weight panel code."""

from __future__ import annotations

FEATURE = "prepack_weight_panels"
PLAN_FILE = "weight_panel_plan.json"


def _implementation():
    from .backend_spi import load_vector_codegen

    return load_vector_codegen("rvv").weight_panel_api()


def __getattr__(name: str):
    if name.startswith("__"):
        raise AttributeError(name)
    return getattr(_implementation(), name)


def ensure_registered():
    return _implementation().ensure_registered()
