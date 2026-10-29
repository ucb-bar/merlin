"""Reuse a validated engine identity while every file it pins is unchanged.

Validating a prebuilt simulator's identity hashes its binary, its build receipt and every file the
receipt pins. A grade asks for that identity once per completed test case (to route its certificate
job), so the same unchanged files were hashed hundreds of times per grade: measured, about 60% of a
grade's completion processing, delaying its first verdict by about 20 minutes. A value is reused only
while the stat identity of every pinned file and directory tree is unchanged (any content change moves
the file's ctime, which an unprivileged process cannot set), never for a file changed within
:data:`SETTLED_NS`, and never for longer than :data:`MAX_AGE_S`, after which it is validated in full.
"""

from __future__ import annotations

import copy
import json
import os
import threading
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

MAX_AGE_S = 300.0
SETTLED_NS = 2_000_000_000
_CACHE: dict[tuple, tuple[tuple, float, Any]] = {}
_LOCK = threading.Lock()


def _stat_row(path: str) -> tuple:
    st = os.lstat(path)
    return (path, st.st_dev, st.st_ino, st.st_mode, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def signature(paths: Iterable[str]) -> tuple | None:
    """Stat identity of each path, and of every entry below a directory; ``None`` if any is unreadable."""
    rows = []
    try:
        for raw in sorted(set(paths)):
            rows.append(_stat_row(raw))
            if os.path.isdir(raw) and not os.path.islink(raw):
                for directory, dirnames, filenames in os.walk(raw):
                    dirnames.sort()
                    for name in sorted(dirnames) + sorted(filenames):
                        rows.append(_stat_row(os.path.join(directory, name)))
    except OSError:
        return None
    return tuple(rows)


def pinned_paths(document: Any) -> list[str]:
    """Every absolute ``path`` of a ``{path, sha256}`` pin anywhere in a JSON document."""
    found: list[str] = []
    stack = [document]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            path = item.get("path")
            if isinstance(path, str) and isinstance(item.get("sha256"), str) and os.path.isabs(path):
                found.append(path)
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    return found


def receipt_paths(receipt: str | Path) -> list[str]:
    """The receipt itself and every file it pins; an unreadable receipt pins nothing extra."""
    try:
        document = json.loads(Path(receipt).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        document = None
    return [str(receipt), *pinned_paths(document)]


def cached(key: tuple, paths: Iterable[str], compute: Callable[[], Any]) -> Any:
    """``compute()``, reused while ``paths`` keep the signature they had when it was computed."""
    paths = list(paths)
    before = signature(paths)
    now = time.monotonic()
    with _LOCK:
        entry = _CACHE.get(key)
    if before is not None and entry is not None and entry[0] == before and now - entry[1] < MAX_AGE_S:
        return copy.deepcopy(entry[2])
    value = compute()
    settled = before is not None and all(
        row[5] < time.time_ns() - SETTLED_NS and row[6] < time.time_ns() - SETTLED_NS for row in before
    )
    if settled and signature(paths) == before:
        with _LOCK:
            _CACHE[key] = (before, now, copy.deepcopy(value))
    return value


def clear() -> None:
    with _LOCK:
        _CACHE.clear()
