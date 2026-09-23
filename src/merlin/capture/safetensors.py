"""Bounded safetensors header IO shared by capture rewrites and weight packers.

This reader validates header framing and JSON, not tensor layout, payload identity,
dtype support or execution authority. Those checks belong to each consuming operation.
It never reads or converts the tensor payload and preserves metadata and entry order.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import BinaryIO

MAX_HEADER_BYTES = 16 * 1024 * 1024  # Merlin's metadata bound, not a file-format limit.


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate safetensors header key: {key!r}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"non-finite safetensors header value: {value}")


def read_header(stream: BinaryIO, *, max_header_bytes: int = MAX_HEADER_BYTES) -> tuple[dict, int]:
    """Read from offset zero and leave the stream at the payload; return header and offset."""
    if type(max_header_bytes) is not int or max_header_bytes < 0:
        raise ValueError("header byte bound must be a nonnegative integer")
    if stream.tell() != 0:
        raise ValueError("safetensors header reader requires offset zero")
    prefix = stream.read(8)
    if len(prefix) != 8:
        raise ValueError("truncated safetensors length prefix")
    size = struct.unpack("<Q", prefix)[0]
    if size > max_header_bytes:
        raise ValueError("safetensors header exceeds configured byte bound")
    payload = stream.read(size)
    if len(payload) != size:
        raise ValueError("truncated safetensors header")
    try:
        header = json.loads(payload, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("malformed safetensors header JSON") from exc
    if not isinstance(header, dict):
        raise ValueError("safetensors header must be an object")
    return header, 8 + size


def load_header(path: str | Path) -> tuple[dict, int]:
    """Read one file's header without keeping the file open or reading its tensor data."""
    with Path(path).open("rb") as stream:
        return read_header(stream)
