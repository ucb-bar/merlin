"""Scoped SpecIR imports, without owning reference-model semantics or discovery."""

from __future__ import annotations

import sys
import threading
from contextlib import contextmanager
from pathlib import Path

_IMPORT_LOCK = threading.RLock()


class SpecIRImportError(ImportError):
    """The requested checkout cannot supply an unambiguous SpecIR module graph."""


def _check_loaded(package: Path) -> None:
    for name, module in tuple(sys.modules.items()):
        if name != "specir" and not name.startswith("specir."):
            continue
        origin = getattr(getattr(module, "__spec__", None), "origin", None)
        locations = list(getattr(module, "__path__", ()))
        if origin and origin not in {"built-in", "frozen"}:
            locations.append(origin)
        if not locations or any(not Path(path).resolve().is_relative_to(package) for path in locations):
            raise SpecIRImportError(
                f"loaded {name} does not belong to SpecIR checkout {package.parent}; "
                "select the checkout in a fresh process (modules are never unloaded)"
            )


@contextmanager
def importable(root: str | Path | None):
    """Expose a selected checkout only for this scope; retain normal module identity.

    None uses ordinary installed resolution unchanged. Explicit roots cannot fall back
    to another installation. Cooperating callers serialize path changes, not arbitrary
    upstream side effects or unrelated imports. This is not a Python sandbox.
    """
    with _IMPORT_LOCK:
        before = list(sys.path)
        try:
            package = None
            if root is not None:
                base = Path(root).resolve()
                package = base / "specir"
                if not (package / "__init__.py").is_file():
                    raise SpecIRImportError(f"SpecIR checkout missing at {base}; configure SPECIR_ROOT")
                package = package.resolve()
                _check_loaded(package)
                sys.path.insert(0, str(base))
            yield
            # Only validate successful bodies: never replace their primary exception
            # with a secondary ownership error from partially imported modules.
            if package is not None:
                _check_loaded(package)
        finally:
            sys.path[:] = before
