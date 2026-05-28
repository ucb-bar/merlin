"""Shared utilities for intake scanners."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

_SKIP_DIR_NAMES: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".venv",
        "venv",
        "node_modules",
        "build",
        ".cache",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "target",
    }
)

_BINARY_SUFFIXES: frozenset[str] = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".pdf",
        ".bin",
        ".so",
        ".o",
        ".a",
        ".dylib",
        ".dll",
        ".exe",
        ".pyc",
        ".pyo",
        ".whl",
        ".tar",
        ".gz",
        ".zip",
        ".class",
        ".jar",
    }
)

_MAX_TEXT_SIZE_BYTES = 4 * 1024 * 1024


def iter_files(root: Path) -> Iterator[Path]:
    """Yield text-like files under ``root``, skipping common build/VCS dirs."""
    if not root.exists():
        return
    if root.is_file():
        if _is_text_candidate(root):
            yield root
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIR_NAMES]
        for name in filenames:
            path = Path(dirpath) / name
            if _is_text_candidate(path):
                yield path


def _is_text_candidate(path: Path) -> bool:
    if path.suffix.lower() in _BINARY_SUFFIXES:
        return False
    try:
        if path.stat().st_size > _MAX_TEXT_SIZE_BYTES:
            return False
    except OSError:
        return False
    return True


def read_text(path: Path) -> str | None:
    """Return file contents as text or ``None`` on failure."""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None


def relative_to(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def matches_any(name: str, patterns: tuple[str, ...]) -> bool:
    """Trivial glob over the basename: '*' matches anything, otherwise substring."""
    for pat in patterns:
        if pat == "*":
            return True
        if pat.startswith("*") and pat.endswith("*"):
            if pat.strip("*") in name:
                return True
        elif pat.startswith("*"):
            if name.endswith(pat[1:]):
                return True
        elif pat.endswith("*"):
            if name.startswith(pat[:-1]):
                return True
        elif pat == name:
            return True
    return False
