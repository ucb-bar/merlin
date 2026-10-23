"""Bounded, cached, read-only record access shared by every console source.

A :class:`Reader` belongs to one experiment. It caches each parsed document by its stat identity, so a
stored run is parsed once and a live run only re-reads what changed. Every refresh has a byte budget;
exhausting it yields ``None`` plus a notice rather than a partial parse. Files are opened read-only and
never locked, renamed or written.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..tail import JsonlTail

_HEX = frozenset("0123456789abcdef")


def obj(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def rows(value: Any, limit: int = 4096) -> list:
    if isinstance(value, dict):
        value = list(value.values())
    return [row for row in value[:limit] if isinstance(row, dict)] if isinstance(value, list) else []


def num(value: Any) -> float | int | None:
    """A finite recorded number, else None (booleans are not numbers here)."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return value if math.isfinite(value) else None


def positive(value: Any) -> float | int | None:
    value = num(value)
    return value if value is not None and value > 0 else None


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in _HEX for c in value)


def clean(value: Any, limit: int = 200) -> str:
    """Record text that cannot carry terminal controls, newlines or unbounded length."""
    if value is None:
        return ""
    return "".join(c if c.isprintable() else " " for c in str(value))[:limit]


def epoch(value: Any) -> float | None:
    """A recorded time as epoch seconds.

    Accepts seconds, milliseconds and nanoseconds (told apart by magnitude), ``YYYYmmddTHHMMSSZ`` and
    ISO-8601 strings; anything else is None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        if not math.isfinite(value) or value <= 0:
            return None
        if value > 1e17:
            return value / 1e9
        if value > 1e14:
            return value / 1e6
        if value > 1e11:
            return value / 1e3
        return float(value)
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    try:
        return datetime.strptime(text, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC).timestamp()
    except ValueError:
        pass
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.timestamp()


def mtime(path: Path | str | None) -> float | None:
    if path is None:
        return None
    try:
        return os.stat(path).st_mtime
    except OSError:
        return None


def is_file(path: Path | str) -> bool:
    """Like ``Path.is_file`` but false, never raising, for unreadable paths."""
    try:
        return os.path.isfile(path)
    except (OSError, ValueError):
        return False


def is_dir(path: Path | str) -> bool:
    try:
        return os.path.isdir(path)
    except (OSError, ValueError):
        return False


def bounded_glob(root: Path, pattern: str, limit: int) -> list[Path]:
    """At most ``limit`` matches, without sorting a whole directory first."""
    out: list[Path] = []
    try:
        for path in Path(root).glob(pattern):
            out.append(path)
            if len(out) >= limit:
                break
    except OSError:
        pass
    return out


def scan_dir(root: Path, limit: int) -> Iterator[os.DirEntry]:
    """Entries of one directory, bounded, silently empty when it cannot be listed."""
    try:
        with os.scandir(root) as entries:
            for count, entry in enumerate(entries):
                if count >= limit:
                    return
                yield entry
    except OSError:
        return


class Reader:
    """One experiment's document cache with a per-refresh byte budget."""

    def __init__(self, *, budget: int = 128 << 20) -> None:
        self.budget = budget
        self.remaining = budget
        self.notices: set[str] = set()
        self._documents: dict[str, tuple[tuple, Any]] = {}
        self._tails: dict[str, JsonlTail] = {}

    def begin(self) -> None:
        """Start a refresh: restore the byte budget and clear its notices."""
        self.remaining = self.budget
        self.notices = set()

    @staticmethod
    def identity(path: Path) -> tuple | None:
        try:
            st = os.stat(path)
        except OSError:
            return None
        return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)

    def _load(self, path: Path, limit: int, parse) -> Any:
        path = Path(path)
        key = str(path)
        ident = self.identity(path)
        if ident is None:
            self._documents.pop(key, None)
            return None
        cached = self._documents.get(key)
        if cached and cached[0] == ident:
            return cached[1]
        size = ident[2]
        if size > limit:
            self.notices.add(f"{path.name} exceeds the {limit >> 20} MiB per-file read bound")
            return None
        if size > self.remaining:
            self.notices.add("refresh byte budget reached; some records wait for the next refresh")
            return None
        try:
            with open(path, "rb") as stream:
                data = stream.read(limit + 1)
        except OSError:
            return None
        self.remaining -= len(data)
        try:
            document = parse(data)
        except (ValueError, UnicodeError):
            self.notices.add(f"{path.name} is unreadable")
            document = None
        if self.identity(path) != ident:
            return None  # Changed while read: the next refresh reads it whole.
        self._documents[key] = (ident, document)
        return document

    def json(self, path: Path, *, limit: int = 8 << 20) -> Any:
        return self._load(path, limit, json.loads)

    def document(self, path: Path, *, limit: int = 8 << 20) -> dict | None:
        """A JSON object, or None for anything else."""
        value = self.json(path, limit=limit)
        return value if isinstance(value, dict) else None

    def yaml(self, path: Path, *, limit: int = 8 << 20) -> Any:
        def parse(data: bytes) -> Any:
            import yaml

            try:
                return yaml.load(data, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))
            except yaml.YAMLError as exc:
                raise ValueError("unreadable YAML") from exc

        return self._load(path, limit, parse)

    def text_tail(self, path: Path, nbytes: int = 64 << 10) -> list[str]:
        """The complete lines among the last ``nbytes`` of a text file (never cached)."""
        try:
            with open(path, "rb") as stream:
                stream.seek(0, os.SEEK_END)
                size = stream.tell()
                start = max(0, size - nbytes)
                stream.seek(start)
                data = stream.read(min(nbytes, self.remaining))
        except OSError:
            return []
        self.remaining -= len(data)
        lines = data.decode(errors="replace").splitlines()
        return lines[1:] if start > 0 and lines else lines

    def tail(self, path: Path) -> JsonlTail:
        key = str(path)
        if key not in self._tails:
            self._tails[key] = JsonlTail(Path(path))
        return self._tails[key]

    def poll(self, path: Path, consume, on_reset=None) -> int:
        """Rows completed since the previous poll of ``path`` (see :class:`~..tail.JsonlTail`)."""
        tail = self.tail(path)
        before = tail.bytes_read
        count = tail.poll(consume, on_reset)
        self.remaining -= tail.bytes_read - before
        return count


def first_existing(paths: Iterable[Path | None]) -> Path | None:
    for path in paths:
        if path is not None and Path(path).exists():
            return Path(path)
    return None


__all__ = [
    "Reader",
    "bounded_glob",
    "clean",
    "epoch",
    "first_existing",
    "is_dir",
    "is_file",
    "is_sha256",
    "mtime",
    "num",
    "obj",
    "positive",
    "rows",
    "scan_dir",
]
