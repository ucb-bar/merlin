"""Reuse a value computed from a directory tree for as long as the tree is unchanged.

Every completed test case of a grade promotes against the same frozen submission snapshot, and
hashing it again for each one held the grader's lock and interpreter for most of a second. A value
is keyed by :func:`tree_signature`, so any edit recomputes it; a tree that changed within
:data:`SETTLED_NS` is recomputed every time.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

CACHE: dict[str, tuple] = {}
CACHE_MAX = 32
#: A tree changed more recently than this is never cached: file timestamps advance in coarse ticks, so
#: two writes inside one tick can leave every stat field equal (the "racy git" case).
SETTLED_NS = 2_000_000_000


def tree_signature(root: Path):
    """The stat identity of every entry under ``root``, read without opening a file.

    Any change to a file's bytes or metadata moves its ctime, which an unprivileged process cannot set,
    and adding, removing or renaming an entry moves its directory's; so an equal signature means the
    tree is unchanged. ``None`` when the tree cannot be listed."""
    rows = []
    try:
        for directory, dirnames, filenames in os.walk(root):
            dirnames.sort()
            for name in [".", *sorted(filenames)]:
                path = os.path.join(directory, name) if name != "." else directory
                st = os.lstat(path)
                rows.append((os.path.relpath(path, root), st.st_dev, st.st_ino, st.st_mode, st.st_size,
                             st.st_mtime_ns, st.st_ctime_ns))
            for name in dirnames:
                if os.path.islink(os.path.join(directory, name)):
                    st = os.lstat(os.path.join(directory, name))
                    rows.append((os.path.relpath(os.path.join(directory, name), root), st.st_dev, st.st_ino,
                                 st.st_mode, st.st_size, st.st_mtime_ns, st.st_ctime_ns))
    except OSError:
        return None
    return tuple(rows)


def cached(root: Path, compute: Callable[[], Any]) -> Any:
    """``compute()``, reused while ``root`` keeps the signature it had when the value was computed."""
    root = Path(root)
    key = str(root.resolve()) if root.exists() else str(root)
    signature = tree_signature(root)
    entry = CACHE.get(key)
    if signature is not None and entry is not None and entry[0] == signature:
        return entry[1]
    value = compute()
    settled = signature is not None and all(
        row[5] < time.time_ns() - SETTLED_NS and row[6] < time.time_ns() - SETTLED_NS for row in signature
    )
    if settled and tree_signature(root) == signature:
        if len(CACHE) >= CACHE_MAX:
            CACHE.pop(next(iter(CACHE)))
        CACHE[key] = (signature, value)
    return value
