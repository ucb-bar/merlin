"""Independent program and compiler transformation checks with explicit scope.

``HAS_XDSL`` and ``HAS_Z3`` report whether the optional engines are importable, so a check that
cannot run reports a skip instead of passing silently.
"""

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
