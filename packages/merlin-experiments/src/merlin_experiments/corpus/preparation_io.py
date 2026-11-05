"""Private record publication and ordinary source-tree checks for corpus assembly."""

from __future__ import annotations

import json
import os
from pathlib import Path

from ..spec import SpecError


def private_json(path: Path, document: dict) -> None:
    """Publish a host-only record without an intermediate world-readable file."""
    temporary = path.with_name(path.name + ".pending")
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(document, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def ordinary_tree(path: Path) -> None:
    """A release has no live indirection or unclassified filesystem entries."""
    if path.is_symlink() or not path.exists():
        raise SpecError("corpus release source is absent or symlinked")
    for member in [path, *path.rglob("*")] if path.is_dir() else [path]:
        if member.is_symlink() or not (member.is_file() or member.is_dir()):
            raise SpecError("corpus release source contains symlinked or nonregular entries")
