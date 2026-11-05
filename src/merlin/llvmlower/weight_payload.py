"""Lazy byte spans for the existing generated weight-blob layout.

This is build-time host storage, not a device memory or scheduling contract.
Readers borrow source files/arrays until consumption; callers still own content
freshness and immutable invocation inputs. No content identity is memoized.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Compiler I/O work budget, not hardware geometry or a transfer-cost estimate.
CHUNK_BYTES = 1 << 20


@dataclass(frozen=True)
class _Span:
    size: int
    read: Callable[[int, int, int], Iterator[bytes | memoryview]]
    begin: int = 0


class WeightPayload:
    """Ordered file/array/zero spans; slicing and alignment never copy payloads."""

    def __init__(self):
        self._spans: list[_Span] = []
        self._size = 0

    def __len__(self):
        return self._size

    def __bytes__(self):
        """Explicit legacy materialization; ordinary generation consumes chunks."""
        return b"".join(self.chunks())

    def extend(self, other: WeightPayload):
        self._spans.extend(other._spans)
        self._size += len(other)

    def _add(self, size, reader):
        if type(size) is not int or size < 0:
            raise ValueError("weight byte span needs a nonnegative integer size")
        self._spans.append(_Span(size, reader))
        self._size += size
        return self

    @classmethod
    def zeros(cls, size: int):
        def read(begin, count, chunk):
            block = bytes(min(count, chunk))
            while count:
                length = min(count, len(block))
                yield block[:length]
                count -= length

        return cls()._add(size, read)

    @classmethod
    def file(cls, path: Path, begin: int, size: int):
        path = Path(path).resolve(strict=True)
        selected = path.stat()
        if type(begin) is not int or begin < 0 or type(size) is not int or size < 0 or begin + size > selected.st_size:
            raise ValueError("weight file span exceeds source bytes")

        def identity(stat):
            return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

        def block(stream, index):
            stream.seek(begin + index * CHUNK_BYTES)
            expected = min(CHUNK_BYTES, size - index * CHUNK_BYTES)
            data = stream.read(expected)
            if len(data) != expected:
                raise ValueError("weight file truncated during consumption")
            return data

        # Timestamps are only an additional drift signal. Store bounded block
        # digests so restored/coalesced filesystem times never establish identity.
        with path.open("rb", buffering=0) as stream:
            digests = tuple(hashlib.sha256(block(stream, i)).digest()
                            for i in range((size + CHUNK_BYTES - 1) // CHUNK_BYTES))

        def read(offset, count, chunk):
            import os

            original_offset, original_count = offset, count
            with path.open("rb") as stream:
                if identity(os.fstat(stream.fileno())) != identity(selected):
                    raise ValueError("weight file changed before consumption")
                while count:
                    index, within = divmod(offset, CHUNK_BYTES)
                    data = block(stream, index)
                    if hashlib.sha256(data).digest() != digests[index]:
                        raise ValueError("weight file bytes changed before consumption")
                    length = min(count, len(data) - within)
                    view = memoryview(data)
                    for start in range(within, within + length, chunk):
                        yield view[start:min(start + chunk, within + length)]
                    count -= length
                    offset += length
                if (identity(os.fstat(stream.fileno())) != identity(selected)
                        or identity(path.stat()) != identity(selected)):
                    raise ValueError("weight file changed during consumption")
            # A yielded chunk may outlive a caller's write. Reopen rather than
            # reusing buffered bytes when checking the consumed source blocks.
            with path.open("rb", buffering=0) as stream:
                first = original_offset // CHUNK_BYTES
                last = (original_offset + original_count + CHUNK_BYTES - 1) // CHUNK_BYTES
                for index in range(first, last):
                    if hashlib.sha256(block(stream, index)).digest() != digests[index]:
                        raise ValueError("weight file bytes changed during consumption")

        return cls()._add(size, read)

    @classmethod
    def array(cls, size: int, load: Callable[[], np.ndarray]):
        """Load and validate one caller-owned array only while its span is read."""
        def read(begin, count, chunk):
            array = np.ascontiguousarray(load())
            if array.nbytes != size:
                raise ValueError("weight array disagrees with its declared byte span")
            if not count:
                return
            view = memoryview(array).cast("B")
            for offset in range(begin, begin + count, chunk):
                yield view[offset:min(offset + chunk, begin + count)]

        return cls()._add(size, read)

    def __getitem__(self, selection):
        if not isinstance(selection, slice) or selection.step not in (None, 1):
            raise ValueError("weight byte selection requires a contiguous slice")
        begin = 0 if selection.start is None else selection.start
        end = len(self) if selection.stop is None else selection.stop
        if type(begin) is not int or type(end) is not int or begin < 0 or end < begin or end > len(self):
            raise ValueError("weight byte slice exceeds payload")
        result, cursor = WeightPayload(), 0
        for span in self._spans:
            left, right = max(begin, cursor), min(end, cursor + span.size)
            if left < right:
                result._spans.append(_Span(right - left, span.read, span.begin + left - cursor))
                result._size += right - left
            cursor += span.size
        return result

    def chunks(self, chunk_bytes: int = CHUNK_BYTES):
        if type(chunk_bytes) is not int or chunk_bytes < 1:
            raise ValueError("weight I/O chunk budget must be a positive integer")
        for span in self._spans:
            count = 0
            for data in span.read(span.begin, span.size, chunk_bytes):
                if not isinstance(data, (bytes, memoryview)) or len(data) > chunk_bytes or not len(data):
                    raise ValueError("weight source returned an invalid bounded chunk")
                count += len(data)
                if count > span.size:
                    raise ValueError("weight source exceeded its selected span")
                yield data
            if count != span.size:
                raise ValueError("weight source did not cover its complete span")
