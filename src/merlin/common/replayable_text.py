"""Bounded complete UTF-8 files for existing replayable line consumers.

This is raw transport, not an execution or semantic receipt. Every complete
iteration reopens and hashes all original bytes. A consumer which stops early
has not verified that iteration and must not publish a result from it.
"""

from __future__ import annotations

import codecs
import hashlib
import os
import stat
from collections.abc import Iterator
from pathlib import Path


def _stamp(value: os.stat_result) -> tuple[int, ...]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


class ReplayableText:
    """A pinned raw file with ``str.splitlines()`` compatible streaming passes.

    Callers explicitly supply their existing *decoded/raw* byte ceiling. There
    is no compression, filtering, line sampling or ceiling reinterpretation.
    ``record`` describes bytes only; callers own process and source provenance.
    """

    def __init__(self, path: Path, *, max_bytes: int, chunk_bytes: int = 65536):
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError("raw byte budget must be a nonnegative integer")
        if type(chunk_bytes) is not int or not 0 < chunk_bytes <= 1 << 20:
            raise ValueError("stream chunk must be between one byte and one MiB")
        self.path = Path(path).absolute()
        self.max_bytes = max_bytes
        self.chunk_bytes = chunk_bytes
        if self.path.is_symlink():
            raise ValueError("raw stream must be an ordinary selected file")
        value = self.path.stat()
        if not stat.S_ISREG(value.st_mode) or value.st_size > max_bytes:
            raise ValueError("raw stream exceeds its independent byte budget or is not regular")
        self._stamp = _stamp(value)
        self._sha256 = self._digest()

    def _open(self):
        descriptor = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        stream = os.fdopen(descriptor, "rb")
        if _stamp(os.fstat(stream.fileno())) != self._stamp:
            stream.close()
            raise ValueError("raw stream identity changed")
        return stream

    def _finish(self, stream, count: int, digest: str) -> None:
        if (
            _stamp(os.fstat(stream.fileno())) != self._stamp
            or _stamp(self.path.stat()) != self._stamp
            or count != self._stamp[2]
            or count > self.max_bytes
            or (hasattr(self, "_sha256") and digest != self._sha256)
        ):
            raise ValueError("raw stream changed or is incomplete")
        # A same-size rewrite can occur within the filesystem timestamp tick
        # while the reader still holds buffered original bytes. Reopen and
        # hash current content rather than authenticating it by timestamps.
        current, total = hashlib.sha256(), 0
        with self._open() as reopened:
            while raw := reopened.read(self.chunk_bytes):
                total += len(raw)
                if total > self.max_bytes:
                    raise ValueError("raw stream exceeds its independent byte budget")
                current.update(raw)
            if (
                total != count
                or current.hexdigest() != digest
                or _stamp(os.fstat(reopened.fileno())) != self._stamp
                or _stamp(self.path.stat()) != self._stamp
            ):
                raise ValueError("raw stream changed during its complete pass")

    def _digest(self) -> str:
        digest, count = hashlib.sha256(), 0
        with self._open() as stream:
            while raw := stream.read(self.chunk_bytes):
                count += len(raw)
                if count > self.max_bytes:
                    raise ValueError("raw stream exceeds its independent byte budget")
                digest.update(raw)
            result = digest.hexdigest()
            self._finish(stream, count, result)
        return result

    def verify(self) -> None:
        """Reopen complete raw bytes; timestamps alone never authenticate them."""
        self._digest()

    def record(self) -> dict[str, str | int]:
        return {
            "schema": "merlin.replayable_raw_utf8.v1",
            "path": str(self.path),
            "sha256": self._sha256,
            "bytes": self._stamp[2],
            "max_raw_bytes": self.max_bytes,
        }

    def splitlines(self) -> Iterator[str]:
        """Yield exactly the lines of strict UTF-8 ``str.splitlines()``.

        Keep the final segment between chunks so CRLF and multibyte line
        separators remain exact even when split across reads. Full exhaustion
        checks the complete original hash and file identity before returning.
        """
        digest, count = hashlib.sha256(), 0
        decoder = codecs.getincrementaldecoder("utf-8")("strict")
        pending = ""
        with self._open() as stream:
            while raw := stream.read(self.chunk_bytes):
                count += len(raw)
                if count > self.max_bytes:
                    raise ValueError("raw stream exceeds its independent byte budget")
                digest.update(raw)
                lines = (pending + decoder.decode(raw)).splitlines(keepends=True)
                pending = lines.pop() if lines else ""
                for line in lines:
                    yield line.splitlines()[0]
            pending += decoder.decode(b"", final=True)
            yield from pending.splitlines()
            self._finish(stream, count, digest.hexdigest())
