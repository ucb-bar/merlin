"""Read one pinned captured state tensor without executing or packing a model.

This verifies capture bytes, not permission to prepack a source argument. The
caller must independently prove the raw-source/manifest binding, preservation
through normalization, and read-only use. Candidate role/recipe claims are not
accepted here. No dtype conversion or interpretation of tensor bytes occurs.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from math import prod
from pathlib import Path
import struct
from collections.abc import Sequence


# File-format storage types, not accelerator facts. Keep this explicit: unknown
# or sub-byte formats require their own byte-preserving contract, never a cast.
_TYPES = {
    "I8": ("i8", "int8", 1), "I16": ("i16", "int16", 2),
    "I32": ("i32", "int32", 4), "I64": ("i64", "int64", 8),
    "U8": ("ui8", "uint8", 1), "U16": ("ui16", "uint16", 2),
    "U32": ("ui32", "uint32", 4), "U64": ("ui64", "uint64", 8),
    "F16": ("f16", "float16", 2), "BF16": ("bf16", "bfloat16", 2),
    "F32": ("f32", "float32", 4), "F64": ("f64", "float64", 8),
    "BOOL": ("i1", "bool", 1),
}
_MAX_MANIFEST_BYTES = 4 * 1024 * 1024
_MAX_HEADER_BYTES = 16 * 1024 * 1024
_CHUNK_BYTES = 64 * 1024


def _digest_pin(value: str) -> None:
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError("capture pins require lowercase SHA-256 hex digests")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate capture JSON key: {key!r}")
        result[key] = value
    return result


def _json_object(payload: bytes) -> dict:
    def invalid_constant(value):
        raise ValueError(f"non-finite capture JSON value: {value}")
    try:
        value = json.loads(payload, object_pairs_hook=_unique_object,
                           parse_constant=invalid_constant)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("malformed capture JSON") from error
    if not isinstance(value, dict):
        raise ValueError("capture JSON root must be an object")
    return value


def _shape(value) -> tuple[int, ...]:
    if (not isinstance(value, (list, tuple))
            or any(type(dim) is not int or dim < 0 for dim in value)):
        raise ValueError("capture shape requires nonnegative integer extents")
    return tuple(value)


@dataclass(frozen=True)
class CapturedConstant:
    """An immutable byte snapshot. Its construction alone grants no authority."""
    entry_argument_index: int
    manifest_path: str
    manifest_sha256: str
    safetensors_path: str
    safetensors_sha256: str
    tensor_key: str
    manifest_kind: str
    source_shape: tuple[int, ...]
    source_dtype: str
    storage_dtype: str
    header_sha256: str
    file_offset_bytes: int
    payload_sha256: str
    logical_payload: bytes

    def to_evidence(self) -> dict:
        return {
            "schema": "verified_capture_constant_bytes_v1",
            "entry_argument_index": self.entry_argument_index,
            "manifest_path": self.manifest_path, "manifest_sha256": self.manifest_sha256,
            "safetensors_path": self.safetensors_path,
            "safetensors_sha256": self.safetensors_sha256,
            "tensor_key": self.tensor_key, "manifest_kind": self.manifest_kind,
            "source_shape": list(self.source_shape), "source_dtype": self.source_dtype,
            "storage_dtype": self.storage_dtype, "header_sha256": self.header_sha256,
            "file_offset_bytes": self.file_offset_bytes,
            "payload_bytes": len(self.logical_payload), "payload_sha256": self.payload_sha256,
            "scope": "pinned captured state bytes only; source binding and prepack authorization unproven",
            "prepack_authorized": False,
        }


def verify_capture_constant(*, manifest_path: str | Path, manifest_sha256: str,
                            safetensors_path: str | Path, safetensors_sha256: str,
                            entry_argument_index: int, source_shape: Sequence[int],
                            source_dtype: str, max_payload_bytes: int) -> CapturedConstant:
    """Verify two pinned files and load only one explicitly bounded tensor.

    Every tensor header is checked for a contiguous, nonoverlapping valid file
    partition. The full blob is hashed in chunks; only the selected payload is
    retained. The selected bytes come from that same hashing pass, preventing a
    separate post-verification read from silently observing different bytes.
    """
    _digest_pin(manifest_sha256)
    _digest_pin(safetensors_sha256)
    if type(entry_argument_index) is not int or entry_argument_index < 0:
        raise ValueError("entry argument index must be a nonnegative integer")
    if type(max_payload_bytes) is not int or max_payload_bytes <= 0:
        raise ValueError("selected payload requires an explicit positive byte budget")
    wanted_shape = _shape(source_shape)
    manifest_path, safetensors_path = Path(manifest_path), Path(safetensors_path)
    with manifest_path.open("rb") as stream:
        manifest_bytes = stream.read(_MAX_MANIFEST_BYTES + 1)
    if len(manifest_bytes) > _MAX_MANIFEST_BYTES:
        raise ValueError("capture manifest exceeds bounded JSON size")
    if hashlib.sha256(manifest_bytes).hexdigest() != manifest_sha256:
        raise ValueError("capture manifest SHA-256 mismatch")
    manifest = _json_object(manifest_bytes)
    if any(not key.isascii() or not key.isdecimal() or str(int(key)) != key for key in manifest):
        raise ValueError("capture manifest requires unique canonical argument indices")
    entry = manifest.get(str(entry_argument_index))
    if (not isinstance(entry, dict) or entry.get("kind") not in ("param", "buffer")
            or "error" in entry or entry.get("stub", False) is not False
            or not isinstance(entry.get("weight"), str) or not entry["weight"]):
        raise ValueError("argument is missing a concrete captured parameter/buffer")
    if _shape(entry.get("shape")) != wanted_shape:
        raise ValueError("manifest shape differs from source argument")
    key = entry["weight"]
    with safetensors_path.open("rb") as stream:
        length_bytes = stream.read(8)
        if len(length_bytes) != 8:
            raise ValueError("truncated safetensors length prefix")
        header_length = struct.unpack("<Q", length_bytes)[0]
        if not 2 <= header_length <= _MAX_HEADER_BYTES:
            raise ValueError("safetensors header exceeds bounded JSON size")
        header_bytes = stream.read(header_length)
        if len(header_bytes) != header_length:
            raise ValueError("truncated safetensors header")
        header = _json_object(header_bytes)
        file_length = stream.seek(0, 2)
        payload_base = 8 + header_length
        ranges = []
        for name, tensor in header.items():
            if name == "__metadata__":
                if not isinstance(tensor, dict) or any(not isinstance(v, str) for v in tensor.values()):
                    raise ValueError("invalid safetensors metadata")
                continue
            if not isinstance(tensor, dict) or set(tensor) != {"dtype", "shape", "data_offsets"}:
                raise ValueError("malformed safetensors tensor header")
            dtype = tensor["dtype"]
            if not isinstance(dtype, str) or dtype not in _TYPES:
                raise ValueError("unsupported safetensors storage dtype")
            shape = _shape(tensor["shape"])
            offsets = tensor["data_offsets"]
            if (not isinstance(offsets, list) or len(offsets) != 2
                    or any(type(n) is not int for n in offsets)):
                raise ValueError("malformed safetensors data offsets")
            begin, end = offsets
            if not 0 <= begin <= end <= file_length - payload_base:
                raise ValueError("safetensors data offsets outside file bounds")
            if end - begin != prod(shape) * _TYPES[dtype][2]:
                raise ValueError("safetensors shape/dtype byte length mismatch")
            ranges.append((begin, end))
        cursor = 0
        for begin, end in sorted(ranges):
            if begin != cursor:
                raise ValueError("safetensors payload overlaps or contains an unindexed hole")
            cursor = end
        if cursor != file_length - payload_base:
            raise ValueError("safetensors payload contains unindexed trailing bytes")
        tensor = header.get(key)
        if key == "__metadata__" or not isinstance(tensor, dict):
            raise ValueError("manifest weight key is absent from safetensors")
        dtype = tensor["dtype"]
        normalized_dtype, manifest_dtype, _ = _TYPES[dtype]
        if (source_dtype != normalized_dtype or entry.get("dtype") != manifest_dtype
                or _shape(tensor["shape"]) != wanted_shape):
            raise ValueError("capture storage type/shape differs from source or manifest")
        begin, end = tensor["data_offsets"]
        if end - begin > max_payload_bytes:
            raise ValueError("selected capture payload exceeds explicit byte budget")
        start, stop = payload_base + begin, payload_base + end
        # Capture the selected bytes from the exact stream being hashed. Also
        # compare the parsed header with the hashed header to reject a changed
        # preliminary view, rather than trusting file stat timestamps.
        stream.seek(0)
        digest = hashlib.sha256()
        prefix = length_bytes + header_bytes
        payload = bytearray()
        position = 0
        while chunk := stream.read(_CHUNK_BYTES):
            digest.update(chunk)
            prefix_end = min(len(prefix), position + len(chunk))
            if position < prefix_end and chunk[:prefix_end-position] != prefix[position:prefix_end]:
                raise ValueError("safetensors header changed during verification")
            lo, hi = max(start, position), min(stop, position + len(chunk))
            if lo < hi:
                payload.extend(chunk[lo-position:hi-position])
            position += len(chunk)
        if (position != file_length or digest.hexdigest() != safetensors_sha256
                or len(payload) != end - begin):
            raise ValueError("safetensors SHA-256 or file length mismatch")
    logical_payload = bytes(payload)
    return CapturedConstant(
        entry_argument_index, str(manifest_path.resolve()), manifest_sha256,
        str(safetensors_path.resolve()), safetensors_sha256, key, entry["kind"],
        wanted_shape, source_dtype, dtype, hashlib.sha256(header_bytes).hexdigest(),
        start, hashlib.sha256(logical_payload).hexdigest(), logical_payload)
