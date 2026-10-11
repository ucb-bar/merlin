"""Verify compact target output evidence against an independently validated array."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np


_OUTPUT_STORAGE_BYTES = {
    "f64": 8, "f32": 4, "f16": 2, "bf16": 2,
    "i64": 8, "i32": 4, "i16": 2, "i8": 1, "i1": 1,
}


def observe_output_digests(
    text: str, outputs: Sequence[tuple[Sequence[int], str]], *,
    require_first_typed: bool = False,
) -> dict:
    """Bind indexed target digests to a captured output ABI, without a golden.

    Counts, order, storage extents and digest syntax must match. A typed first
    digest, when requested, must describe and equal the raw first result on the
    selected little-endian runtime. This observation says nothing about the
    numerical values; ``verify_all_output_sha256`` needs independent arrays.
    """
    if not outputs:
        raise ValueError("output digest observation needs a nonempty output ABI")
    records = [line.split() for line in text.splitlines()
               if line.startswith("OUT_SHA256_INDEX ")]
    if len(records) != len(outputs):
        raise ValueError("indexed output digest count does not match the captured ABI")
    indexed = []
    for index, (record, (shape, dtype)) in enumerate(zip(records, outputs, strict=True)):
        width = _OUTPUT_STORAGE_BYTES.get(dtype)
        if (width is None or not isinstance(shape, (list, tuple))
                or any(type(dim) is not int or dim < 0 for dim in shape)):
            raise ValueError(f"unsupported output ABI at index {index}")
        elements = 1
        for dim in shape:
            elements *= dim
        if (
            len(record) != 6
            or record[:3] != ["OUT_SHA256_INDEX", str(index), "rawbytes"]
            or record[3:5] != [str(elements), str(elements * width)]
            or len(record[5]) != 64
            or any(char not in "0123456789abcdef" for char in record[5])
        ):
            raise ValueError(f"indexed output digest disagrees with output ABI at {index}")
        indexed.append({"index": index, "shape": list(shape), "dtype": dtype,
                        "elements": elements, "bytes": elements * width,
                        "sha256": record[5]})
    typed_records = [line.split() for line in text.splitlines()
                     if line.startswith("OUT_SHA256 ")]
    if require_first_typed and len(typed_records) != 1:
        raise ValueError("expected one typed first-output digest")
    if len(typed_records) > 1:
        raise ValueError("duplicate typed first-output digest")
    typed = None
    if typed_records:
        first = indexed[0]
        encoding = {"f32": "f32le", "i64": "i64le", "i1": "i1byte"}.get(first["dtype"])
        record = typed_records[0]
        if (encoding is None or len(record) != 5
                or record != ["OUT_SHA256", encoding, str(first["elements"]),
                              str(first["bytes"]), first["sha256"]]):
            raise ValueError("typed first-output digest disagrees with indexed raw digest")
        typed = {"encoding": encoding, "elements": first["elements"],
                 "bytes": first["bytes"], "sha256": first["sha256"]}
    return {"indexed": indexed, "first_typed": typed,
            "numerical_qualification": "not_established"}


def verify_output_sha256(text: str, reference: np.ndarray) -> dict:
    """Require one canonical typed digest matching all reference elements.

    The caller must separately establish the reference's numerical validity and
    provenance; a matching digest alone does not validate model semantics.
    """
    records = [line.split() for line in text.splitlines() if line.startswith("OUT_SHA256 ")]
    if len(records) != 1:
        raise ValueError("expected exactly one OUT_SHA256 record")
    record = records[0]
    if (
        len(record) != 5
        or record[0] != "OUT_SHA256"
        or record[1] not in {"f32le", "i64le", "i1byte"}
        or any(not field or any(c not in "0123456789" for c in field) for field in record[2:4])
        or len(record[4]) != 64
        or any(c not in "0123456789abcdef" for c in record[4])
    ):
        raise ValueError("invalid OUT_SHA256 record")
    array = np.asarray(reference)
    encoding = record[1]
    if encoding == "f32le":
        if array.dtype.kind != "f" or array.dtype.itemsize != 4:
            raise ValueError("f32le reference must contain f32 values")
        raw = array.astype("<f4", copy=False).tobytes(order="C")
    elif encoding == "i64le":
        if array.dtype.kind != "i" or array.dtype.itemsize != 8:
            raise ValueError("i64le reference must contain signed i64 values")
        raw = array.astype("<i8", copy=False).tobytes(order="C")
    else:
        if array.dtype.kind != "b" or array.dtype.itemsize != 1:
            raise ValueError("i1byte reference must contain boolean values")
        raw = array.astype(np.bool_, copy=False).tobytes(order="C")
    elements, byte_count, observed = record[2:]
    expected = hashlib.sha256(raw).hexdigest()
    if int(elements) != array.size or int(byte_count) != len(raw):
        raise ValueError("output digest element or byte count mismatch")
    if observed != expected:
        raise ValueError("full output SHA256 mismatch")
    return {"encoding": encoding, "elements": array.size, "bytes": len(raw), "sha256": expected}


def verify_all_output_sha256(text: str, references: list[np.ndarray] | tuple[np.ndarray, ...]) -> list[dict]:
    """Check each indexed raw-storage digest against the matching result array.

    The caller must bind result order and dtype to an independently captured
    session ABI. This verifier checks bytes, not model numerical semantics.
    """
    records = [line.split() for line in text.splitlines() if line.startswith("OUT_SHA256_INDEX ")]
    if len(records) != len(references):
        raise ValueError("indexed output digest count does not match reference results")
    verified = []
    for index, (record, reference) in enumerate(zip(records, references, strict=True)):
        if (
            len(record) != 6
            or record[:3] != ["OUT_SHA256_INDEX", str(index), "rawbytes"]
            or any(not field or any(c not in "0123456789" for c in field) for field in record[3:5])
            or len(record[5]) != 64
            or any(c not in "0123456789abcdef" for c in record[5])
        ):
            raise ValueError("invalid indexed output digest record")
        array = np.asarray(reference)
        if array.dtype.hasobject or not array.dtype.isnative:
            raise ValueError("indexed output reference needs native-endian value storage")
        raw = array.tobytes(order="C")
        if int(record[3]) != array.size or int(record[4]) != len(raw):
            raise ValueError("indexed output digest extent mismatch")
        expected = hashlib.sha256(raw).hexdigest()
        if record[5] != expected:
            raise ValueError("indexed output digest mismatch")
        verified.append({"index": index, "elements": array.size,
                         "bytes": len(raw), "sha256": expected})
    return verified
