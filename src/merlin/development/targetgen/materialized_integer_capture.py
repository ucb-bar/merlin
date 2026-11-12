"""Complete typed references for an explicitly selected native integer capture.

This reopens original receipt-bound source operands and results. It performs no
capture, arithmetic qualification, policy selection or target admission.
"""

from __future__ import annotations

import io
from math import prod

import numpy as np

from merlin.common import strict_json

SCHEMA = "merlin.materialized_public_capture.v3"
MODE = "native_integer_v1"
REFERENCE = "golden.json"
_DTYPES = {"i8": np.dtype("int8"), "i32": np.dtype("int32")}


def selected(selection):
    if selection.get("schema") != SCHEMA:
        return False
    if selection.get("source_mode") != MODE:
        raise ValueError("integer public capture requires explicit native_integer_v1 source mode")
    return True


def abi(metadata):
    """Require unchanged integral source storage, without inventing a recipe."""
    if (
        metadata.get("source_mode") != MODE
        or metadata.get("capture_quantization") != "already_materialized"
        or any(metadata.get(key) is not None for key in ("scheme", "recipe", "recipe_sha256", "precision_conversion"))
        or metadata.get("quantization_stats", {}).get("applied") is not False
        or metadata.get("ok") is not True
        or metadata.get("opaque") != 0
    ):
        raise ValueError("integer public capture must retain its unchanged complete native source")
    for key in ("input_abi", "output_abi"):
        rows = metadata.get(key)
        if not isinstance(rows, list) or not rows:
            raise ValueError("integer public capture needs a complete nonempty typed ABI")
        for row in rows:
            if (
                not isinstance(row, dict)
                or row.get("dtype") not in _DTYPES
                or not isinstance(row.get("shape"), list)
                or any(type(n) is not int or n < 1 for n in row["shape"])
            ):
                raise ValueError("integer public capture supports only complete signed i8/i32 tensors")
    return metadata["input_abi"], metadata["output_abi"]


def outputs(metadata, raw):
    _, rows = abi(metadata)
    golden = strict_json.loads(raw)
    values = golden if len(rows) > 1 else [golden]
    if not isinstance(values, list) or len(values) != len(rows):
        raise ValueError("integer public capture omits a complete output")
    for row, value in zip(rows, values, strict=True):
        pending = [value]
        while pending:
            scalar = pending.pop()
            if isinstance(scalar, list):
                pending.extend(scalar)
            elif type(scalar) is not int:
                raise ValueError("integer public reference contains a converted non-integer scalar")
        array = np.asarray(value)
        bounds = np.iinfo(_DTYPES[row["dtype"]])
        if (
            array.dtype.kind not in "iu"
            or list(array.shape) != row["shape"]
            or array.size != prod(row["shape"])
            or np.any(array < bounds.min)
            or np.any(array > bounds.max)
        ):
            raise ValueError("integer public reference differs from its complete exact signed output ABI")
    return golden


def inputs(metadata, raw):
    rows, _ = abi(metadata)
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        if set(archive.files) != {f"in{i}" for i in range(len(rows))}:
            raise ValueError("integer public capture omits complete source input leaves")
        values = []
        for i, row in enumerate(rows):
            value = archive[f"in{i}"]
            if value.dtype != _DTYPES[row["dtype"]] or list(value.shape) != row["shape"]:
                raise ValueError("integer public input storage differs from its exact signed ABI")
            values.append(value.tolist())
        return values


def reference_members(identities):
    """Protect all copied expected-output leaves, including legacy lossy output0."""
    if REFERENCE not in identities:
        raise ValueError("integer public capture requires complete receipt-bound golden.json")
    return tuple(sorted(name for name in identities if name.startswith(("golden.", "goldens."))))
