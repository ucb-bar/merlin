"""Lazy public compiler generators; selected components do not import sibling tools."""
from __future__ import annotations

from importlib import import_module

__all__ = ["target_repo", "xdsl", "mlir_scaffold", "zephyr_module", "llvm_plan", "oot_package"]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    value = import_module("." + name, __name__)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
