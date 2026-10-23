"""Byte identity for an explicitly pinned plain source export (no Git metadata)."""

import hashlib
import json
from pathlib import Path


def export_digest(root):
    root = Path(root)
    entries = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if ".git" in relative.parts or "__pycache__" in relative.parts:
            continue
        if path.is_symlink():
            entries.append([relative.as_posix(), "symlink", str(path.readlink())])
        elif path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
            entries.append([relative.as_posix(), "file", digest.hexdigest()])
    return hashlib.sha256(json.dumps(entries, separators=(",", ":")).encode()).hexdigest()
