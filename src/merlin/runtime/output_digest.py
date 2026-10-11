"""Verify compact target output evidence against an independently validated array."""

from __future__ import annotations

import hashlib

import numpy as np


def verify_output_sha256(text: str, reference: np.ndarray) -> dict:
    """Require one canonical typed digest matching all reference elements.

    The caller must separately establish the reference's numerical validity and
    provenance; a matching digest alone does not validate model semantics.
    """
    records = [line.split() for line in text.splitlines() if line.startswith("OUT_SHA256")]
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
