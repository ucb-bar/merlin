"""merlin's core dialects in xDSL (the default prototyping plane).

Five dialects with bare namespaces: ``contract``, ``schedule``, ``interface``,
``runtime``, ``dse``. Each module exposes ``DIALECT_NAME``, ``OPS``, ``TYPES``,
``get_dialect()``, and ``build_example()``; everything degrades gracefully when xDSL is
not installed (``HAS_XDSL``). The staged lowering lives in ``lowering/``.
"""
from __future__ import annotations

from ._common import HAS_XDSL, make_context, roundtrip, text
from . import contract, schedule, interface, runtime, dse

CORE_DIALECT_MODULES = (contract, schedule, interface, runtime, dse)


def get_all_dialects():
    """All core Dialect objects (empty list when xDSL is absent)."""
    if not HAS_XDSL:
        return []
    return [m.get_dialect() for m in CORE_DIALECT_MODULES]


def make_core_context():
    """A Context with Builtin + Func + every core dialect loaded."""
    return make_context(*get_all_dialects())
