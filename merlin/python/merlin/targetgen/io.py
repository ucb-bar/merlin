"""Tiny filesystem helpers used by TargetGen ingest/evidence.

Read-only, deterministic, stdlib-only. These exist so evidence collection stays honest:
we read text and list files by suffix, we do not attempt to "understand" arbitrary RTL.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable


def read_text(path: str | Path, errors: str = "replace") -> str:
    """Read a file as UTF-8 text, tolerating undecodable bytes."""
    return Path(path).read_text(encoding="utf-8", errors=errors)


def first_lines(path: str | Path, n: int = 5) -> list[str]:
    """Return the first ``n`` lines of a file (stripped of trailing newlines).

    Returns an empty list if the file cannot be read.
    """
    try:
        out: list[str] = []
        with Path(path).open("r", encoding="utf-8", errors="replace") as fh:
            for _ in range(n):
                line = fh.readline()
                if not line:
                    break
                out.append(line.rstrip("\n"))
        return out
    except (OSError, UnicodeError):
        return []


def find_by_suffix(root: str | Path, suffixes: Iterable[str]) -> list[Path]:
    """Recursively list files under ``root`` whose suffix is in ``suffixes``.

    Suffixes are matched case-insensitively and may be given with or without the leading
    dot (e.g. ``"md"`` or ``".md"``). Results are sorted for determinism. A missing root
    yields an empty list.
    """
    root = Path(root)
    if not root.is_dir():
        return []
    wanted = {("" if s.startswith(".") else ".") + s.lower() for s in suffixes}
    hits = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in wanted]
    return sorted(hits)
