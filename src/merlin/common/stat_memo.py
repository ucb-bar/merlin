"""Reuse a value derived from files while every file it read keeps its stat identity.

A grade re-derives the same identities from the same immutable inputs once per completed test case:
the frozen compiler-source snapshot's tree hash, the selected FIRRTL file's digest, each submission
file's bytes and import graph. Measured on a live grade, those repeated reads held most of the grade's
interpreter time. A value here is reused only while the stat identity of every file it depends on is
unchanged: any change to a file's bytes or metadata moves its ctime, which an unprivileged process
cannot set. A file changed within :data:`SETTLED_NS` is never cached (file times advance in coarse
ticks), a tree containing a symbolic link is never cached (its target is not covered by the link's
stat), and every value is fully recomputed after :data:`MAX_AGE_S`.
"""

from __future__ import annotations

import copy
import os
import stat
import threading
import time
from collections.abc import Callable
from typing import Any

MAX_AGE_S = 300.0
SETTLED_NS = 2_000_000_000
#: Trees larger than this are hashed directly; their signature would cost about as much as the hash.
MAX_TREE_ENTRIES = 200_000
_MAX_ENTRIES = 4096
_MAX_BYTES_HELD = 256 << 20

_CACHE: dict[tuple, tuple[Any, float, Any, int]] = {}
_HELD = 0
_LOCK = threading.Lock()
_DISABLED = os.environ.get("MERLIN_STAT_MEMO", "1").strip() == "0"


def _row(path: str, info: os.stat_result) -> tuple:
    return (path, info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def file_signature(path) -> tuple | None:
    """The stat identity of one regular file; ``None`` for a link, a non-file or an unreadable path."""
    try:
        info = os.lstat(path)
    except OSError:
        return None
    if not stat.S_ISREG(info.st_mode):
        return None
    return (_row(os.fspath(path), info),)


def tree_signature(root, *, skip_names: frozenset = frozenset()) -> tuple | None:
    """The stat identity of ``root`` and of every entry below it; ``None`` if any entry is a link,
    unreadable, or the tree is larger than :data:`MAX_TREE_ENTRIES`.

    Entries named in ``skip_names`` (and everything below them) are left out, for a value that by
    definition never reads them."""
    root = os.fspath(root)
    rows = []
    try:
        top = os.lstat(root)
        if stat.S_ISLNK(top.st_mode):
            return None
        rows.append(_row(root, top))
        for directory, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(name for name in dirnames if name not in skip_names)
            for name in [*dirnames, *sorted(name for name in filenames if name not in skip_names)]:
                path = os.path.join(directory, name)
                info = os.lstat(path)
                if stat.S_ISLNK(info.st_mode):
                    return None
                rows.append(_row(path, info))
                if len(rows) > MAX_TREE_ENTRIES:
                    return None
    except OSError:
        return None
    return tuple(rows)


def _settled(signature: tuple) -> bool:
    horizon = time.time_ns() - SETTLED_NS
    return all(row[5] < horizon and row[6] < horizon for row in signature)


def _weight(value: Any) -> int:
    return len(value) if isinstance(value, (bytes, bytearray)) else 0


def cached(kind: str, key: Any, signature: Callable[[], tuple | None], compute: Callable[[], Any]) -> Any:
    """``compute()``, reused while ``signature()`` is unchanged since it was computed.

    The signature is taken before and after computing; the value is kept only when both agree and every
    row is settled, so a file written during the computation is never cached with stale content.
    Exceptions propagate and are never cached. Mutable values are returned as copies.
    """
    global _HELD
    if _DISABLED:
        return compute()
    entry_key = (kind, key)
    before = signature()
    now = time.monotonic()
    if before is not None:
        with _LOCK:
            entry = _CACHE.get(entry_key)
        if entry is not None and entry[0] == before and now - entry[1] < MAX_AGE_S:
            value = entry[2]
            return value if isinstance(value, (bytes, str, int, float, tuple, frozenset)) else copy.deepcopy(value)
    value = compute()
    if before is not None and _settled(before) and signature() == before:
        weight = _weight(value)
        with _LOCK:
            old = _CACHE.pop(entry_key, None)
            if old is not None:
                _HELD -= old[3]
            if len(_CACHE) >= _MAX_ENTRIES or _HELD + weight > _MAX_BYTES_HELD:
                _CACHE.clear()
                _HELD = 0
            stored = value if isinstance(value, (bytes, str, int, float, tuple, frozenset)) else copy.deepcopy(value)
            _CACHE[entry_key] = (before, now, stored, weight)
            _HELD += weight
    return value


def clear() -> None:
    global _HELD
    with _LOCK:
        _CACHE.clear()
        _HELD = 0
