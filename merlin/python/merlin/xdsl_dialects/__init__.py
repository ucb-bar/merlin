"""merlin's core dialects in xDSL (the default prototyping plane).

Five dialects with bare namespaces: ``contract``, ``schedule``, ``interface``,
``runtime``, ``dse``. Each module exposes ``DIALECT_NAME``, ``OPS``, ``TYPES``,
``get_dialect()``, and ``build_example()``; everything degrades gracefully when xDSL is
not installed (``HAS_XDSL``). The staged lowering lives in ``lowering/``.
"""
from __future__ import annotations

from importlib import import_module

from ._common import HAS_XDSL, make_context, roundtrip, text

_CORE_NAMES = ("contract", "schedule", "interface", "runtime", "dse")
__all__ = ["HAS_XDSL", "make_context", "roundtrip", "text", "CORE_DIALECT_MODULES",
           "get_all_dialects", "make_core_context", *_CORE_NAMES]


def __getattr__(name):
    # Importing a pure compiler helper must not load every dialect and runtime.
    # In an answer-masked compiler sandbox the reference runtime is deliberately absent.
    if name == "CORE_DIALECT_MODULES":
        value = tuple(import_module(f".{module}", __name__) for module in _CORE_NAMES)
    elif name in _CORE_NAMES:
        value = import_module(f".{name}", __name__)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))


def get_all_dialects():
    """All core Dialect objects (empty list when xDSL is absent)."""
    if not HAS_XDSL:
        return []
    return [m.get_dialect() for m in __getattr__("CORE_DIALECT_MODULES")]


def make_core_context():
    """A Context with Builtin + Func + every core dialect loaded."""
    return make_context(*get_all_dialects())
