"""Read growing records cheaply and without touching them: offset tailing and stat signatures.

A live view polls files a running phase is still writing.  It must cost the run nothing, so:

* every file is opened read-only, never locked, renamed, truncated or written;
* a JSONL file is read from the byte offset the previous poll stopped at, and only up to its last
  newline -- a trailing line still being written is left for the next poll, never parsed half-way;
* a file that shrank or was replaced (another inode) is read again from the start, and a file that
  disappeared is simply absent this poll;
* :func:`signature` is the stat of a set of paths, so a caller can skip work whose inputs did not change.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

#: Bytes read per ``read`` call while catching up (bounds the transient buffer, not the file size).
CHUNK = 4 << 20


class JsonlTail:
    """One JSONL file read incrementally: :meth:`poll` returns only the rows completed since last time."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.offset = 0
        self.identity: tuple[int, int] | None = None
        self.rows_read = 0
        self.bad = 0
        self.bytes_read = 0
        self.resets = 0
        self.present = False

    def poll(self, consume: Callable[[dict[str, Any]], None], on_reset: Callable[[], None] | None = None) -> int:
        """Hand each row completed since the last poll to ``consume``; return how many.

        When the file was replaced or truncated, ``on_reset`` runs first and the whole file is read
        again, so a consumer rebuilds what it derived.  Rows are handed over one parsed chunk at a time
        (never the whole file at once), which bounds memory while catching up on a large file."""
        try:
            stat = os.stat(self.path)
        except OSError:
            self.present = False
            return 0
        self.present = True
        identity = (stat.st_dev, stat.st_ino)
        if self.identity is not None and (identity != self.identity or stat.st_size < self.offset):
            self.offset = 0
            self.resets += 1
            if on_reset is not None:
                on_reset()
        self.identity = identity
        if stat.st_size == self.offset:
            return 0
        count = 0
        try:
            with open(self.path, "rb") as handle:
                handle.seek(self.offset)
                pending = b""
                while True:
                    chunk = handle.read(CHUNK)
                    if not chunk:
                        break
                    self.bytes_read += len(chunk)
                    data = pending + chunk
                    cut = data.rfind(b"\n")
                    if cut < 0:
                        pending = data
                        continue
                    pending = data[cut + 1 :]
                    count += self._parse(data[: cut + 1], consume)
                    # Everything up to the last newline is consumed; a partial tail line waits.
                    self.offset = handle.tell() - len(pending)
        except OSError:
            return count
        return count

    def _parse(self, block: bytes, consume: Callable[[dict[str, Any]], None]) -> int:
        count = 0
        for line in block.split(b"\n"):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                self.bad += 1
                continue
            if isinstance(row, dict):
                consume(row)
                count += 1
                self.rows_read += 1
            else:
                self.bad += 1
        return count


class Tails:
    """The tails a view keeps between polls, by path (one-shot views use a fresh one)."""

    def __init__(self) -> None:
        self.files: dict[str, JsonlTail] = {}
        self.state: dict[str, Any] = {}

    def tail(self, path: Path) -> JsonlTail:
        key = str(Path(path))
        if key not in self.files:
            self.files[key] = JsonlTail(Path(path))
        return self.files[key]


def signature(paths: Iterable[Path]) -> tuple[tuple[str, int, int], ...]:
    """``(path, mtime_ns, size)`` for each existing path, sorted: equal signatures mean unchanged inputs."""
    out = []
    for path in paths:
        try:
            stat = os.stat(path)
        except OSError:
            continue
        out.append((str(path), stat.st_mtime_ns, stat.st_size))
    return tuple(sorted(out))


def tree_signature(root: Path, *, max_entries: int = 20000) -> tuple[tuple[str, int, int], ...]:
    """The signature of every file under ``root`` (bounded), for inputs read by a whole-tree reader."""
    paths: list[Path] = []
    for directory, _dirs, names in os.walk(root):
        for name in names:
            paths.append(Path(directory) / name)
            if len(paths) >= max_entries:
                return signature(paths)
    return signature(paths)


__all__ = ["JsonlTail", "Tails", "signature", "tree_signature"]
