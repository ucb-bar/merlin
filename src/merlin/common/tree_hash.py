"""Content identity for source trees, independent of research harness installation."""

from __future__ import annotations

import hashlib
from pathlib import Path

_SKIP = {"build", "__pycache__", ".git"}


def hash_tree(root: Path) -> dict:
    """Hash relative source paths and bytes, excluding generated subtrees.

    The exclusion is relative to the tree being identified: a source checkout may
    itself live below out/build. Legacy digests for such locations excluded every
    file and must be recomputed; historical evidence is never rewritten.

    An unchanged tree's result is reused while every entry keeps its stat identity
    (see :mod:`merlin.common.stat_memo`); a grade hashes the same frozen trees once
    per completed test case.
    """
    from merlin.common import stat_memo

    root = Path(root)
    return stat_memo.cached(
        "hash_tree",
        str(root.absolute()),
        lambda: stat_memo.tree_signature(root, skip_names=frozenset(_SKIP)),
        lambda: _hash_tree(root),
    )


def _hash_tree(root: Path) -> dict:
    if not root.exists():
        return {"present": False, "sha256": None, "n_files": 0}
    digest = hashlib.sha256()
    count = 0
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if not path.is_file() or _SKIP & set(relative.parts):
            continue
        digest.update(relative.as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
        count += 1
    return {"present": True, "sha256": digest.hexdigest(), "n_files": count}
