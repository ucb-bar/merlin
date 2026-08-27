"""Compatibility shim: ``kernels.rvv_knobs`` is now :mod:`merlin.kernels.knobs`.

The knob vocabulary is not vector-specific — a tile edge, a block extent and a dataflow choice are
knobs on any endpoint — so the module name outlived its accuracy. Kept for out-of-tree callers and for
saved configs that name the old path.
"""
from __future__ import annotations

from merlin.kernels.knobs import *          # noqa: F401,F403
from merlin.kernels import knobs as _knobs

__all__ = getattr(_knobs, "__all__", [n for n in dir(_knobs) if not n.startswith("_")])


def __getattr__(name: str):
    return getattr(_knobs, name)
