"""Explicit checkout selection for optional external baseline frameworks."""

from __future__ import annotations

from pathlib import Path

from merlin.common.paths import ExternalPathUnset, ext_path


def require_checkout(path: str | Path, label: str) -> Path:
    """Require a selected repository root, never an ancestor Git checkout."""
    root = Path(path)
    if not root.is_absolute() or not root.is_dir() or not (root / ".git").exists():
        raise FileNotFoundError(f"{label} must name an existing absolute external Git checkout: {root}")
    return root


def checkout(name: str) -> Path:
    """Resolve the existing ``MERLIN_EXT_*`` selector without an in-tree fallback."""
    return require_checkout(ext_path(name), f"MERLIN_EXT_{name.upper()}")


def optional_checkout(name: str) -> Path | None:
    """Report an unconfigured baseline as unavailable rather than importing it."""
    try:
        return checkout(name)
    except ExternalPathUnset:
        return None
