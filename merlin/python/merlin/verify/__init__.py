"""Compiler-pass verification: the static (lit/FileCheck) and formal (SMT) layers.

The capsule bench grades OUTCOMES — an independent golden, a decoded instruction stream, RTL. This
package verifies PASSES: what a single transform did to a single module, and whether it preserved
semantics. See ``docs/design/compiler_verification.md`` for the model and the working log.

Everything here degrades to a reported skip rather than a silent pass when a tool is absent, because
a verification layer that cannot run must never be indistinguishable from one that ran clean.
"""
from __future__ import annotations

try:  # pragma: no cover - import guard
    import xdsl  # noqa: F401
    HAS_XDSL = True
except Exception:  # pragma: no cover
    HAS_XDSL = False

try:  # pragma: no cover - import guard
    import z3  # noqa: F401
    HAS_Z3 = True
except Exception:  # pragma: no cover
    HAS_Z3 = False

__all__ = ["HAS_XDSL", "HAS_Z3"]
