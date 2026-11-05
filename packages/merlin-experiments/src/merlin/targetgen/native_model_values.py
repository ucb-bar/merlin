"""Decode frozen logical tensor leaves without selecting execution policy."""
from __future__ import annotations

import math
from typing import Any


def logical_values(raw: bytes, *, tensor: str, dtype: str, shape: tuple[int, ...], error_type) -> Any:
    """Decode only captured leaf bytes; the submitted ELF still computes every operation."""
    import numpy as np

    types = {
        "i1": "u1",
        "i8": "i1",
        "u8": "u1",
        "i16": "<i2",
        "u16": "<u2",
        "i32": "<i4",
        "u32": "<u4",
        "i64": "<i8",
        "u64": "<u8",
        "index": "<i8",
        "f16": "<f2",
        "f32": "<f4",
        "f64": "<f8",
        "bf16": "<u2",
    }
    if dtype not in types:
        raise error_type(f"{tensor}: unsupported captured leaf dtype {dtype!r}")
    values = np.frombuffer(raw, dtype=np.dtype(types[dtype]))
    expected = math.prod(shape) if shape else 1
    if values.size != expected:
        raise error_type(
            f"{tensor}: captured {values.size} element(s), ABI declares {expected} {dtype} element(s)"
        )
    if dtype == "bf16":
        values = (values.astype(np.uint32) << 16).view(np.float32)
    return values.reshape(shape).tolist() if shape else values[0].item()
