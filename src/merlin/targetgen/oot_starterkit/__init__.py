"""Lazy, target-neutral authoring plumbing and separately selectable transforms.

Importing a parser or serializer does not supply algorithmic transformation helpers.
Historical ``from ... import transforms`` and top-level helper imports retain their APIs.
"""
from __future__ import annotations

from importlib import import_module

_MODULES = ("transforms", "verify")
_HELPERS = {
    "CommandBufferBuilder": ("cmdbuf", "CommandBufferBuilder"),
    "parse_interface": ("iface", "parse_interface"),
    "validate": ("verify", "validate"),
    "verify_module": ("verify", "verify_module"),
}
__all__ = [*_MODULES, *_HELPERS]


def __getattr__(name):
    if name in _MODULES:
        value = import_module("." + name, __name__)
    elif name in _HELPERS:
        module, member = _HELPERS[name]
        value = getattr(import_module("." + module, __name__), member)
    else:
        raise AttributeError(name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
