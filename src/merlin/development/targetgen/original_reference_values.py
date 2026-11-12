"""Exact typed storage for bounded original-source reference checks.

This first implementation supports signed byte-aligned integers and IEEE f32.
Other floating formats refuse; their storage never becomes a float32 surrogate.
These software datatype checks establish no hardware or framework arithmetic.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

from merlin.common.quant_formats import get


def format_record(dtype: str) -> dict:
    fmt = get(dtype)
    record = {
        "kind": fmt.kind,
        "element_bits": fmt.element_bits,
        "signed": fmt.signed,
        "exp_bits": fmt.exp_bits,
        "mant_bits": fmt.mant_bits,
    }
    if fmt.kind == "int_affine" and fmt.signed and fmt.element_bits in {8, 16, 32, 64}:
        return record
    if record == {"kind": "float_ieee", "element_bits": 32, "signed": True, "exp_bits": 8, "mant_bits": 23}:
        return record
    raise ValueError(f"original typed reference does not implement format {dtype!r}")


def round_f32(value: float) -> float:
    """Round a finite binary64 value to f32 using Python's IEEE struct codec.

    Products of two f32 values are exact in binary64. The reference rounds each
    product and each addition separately; it never contracts multiply and add.
    """
    try:
        result = struct.unpack("<f", struct.pack("<f", value))[0]
    except (OverflowError, struct.error) as error:
        raise ValueError("finite f32 reference intermediate overflow") from error
    if not math.isfinite(result):
        raise ValueError("finite f32 reference rejects NaN/Inf")
    return result


def integer_project(value: int, dtype: str, arithmetic: str) -> int:
    fmt = format_record(dtype)
    if type(value) is not int or fmt["kind"] != "int_affine":
        raise ValueError("integer projection requires an exact integer and signed integer format")
    bits = fmt["element_bits"]
    lower, upper = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    if arithmetic == "bounded_exact":
        if not lower <= value <= upper:
            raise ValueError(f"bounded integer reference intermediate/readout overflow in {dtype}")
        return value
    if arithmetic != "modular_wrap":
        raise ValueError("unsupported signed integer reference arithmetic")
    return (value - lower) % (1 << bits) + lower


@dataclass(frozen=True)
class TypedReferenceTensor:
    """A complete named dense logical tensor in its original storage format.

    No implicit cast, stride conversion, byte-order assumption or tensor library
    is used. Construction is not numerical/source admission.
    """

    name: str
    dtype: str
    shape: tuple[int, ...]
    data: bytes
    byteorder: str

    def verify(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("typed reference tensor requires an original slot name")
        if type(self.shape) is not tuple or any(type(n) is not int or n < 1 for n in self.shape):
            raise ValueError("typed reference tensor requires explicit positive static extents")
        if type(self.data) is not bytes or self.byteorder not in {"little", "big"}:
            raise ValueError("typed reference tensor requires raw bytes and explicit byte order")
        bits = format_record(self.dtype)["element_bits"]
        if len(self.data) != math.prod(self.shape) * (bits // 8):
            raise ValueError("typed reference tensor has incomplete original storage")

    def values(self) -> tuple[int | float, ...]:
        self.verify()
        fmt = format_record(self.dtype)
        size = fmt["element_bits"] // 8
        if fmt["kind"] == "float_ieee":
            order = "<" if self.byteorder == "little" else ">"
            values = tuple(row[0] for row in struct.iter_unpack(order + "f", self.data))
            if any(not math.isfinite(value) for value in values):
                raise ValueError("finite f32 reference rejects NaN/Inf input/output")
            return values
        return tuple(
            int.from_bytes(self.data[n : n + size], self.byteorder, signed=True) for n in range(0, len(self.data), size)
        )

    @classmethod
    def from_values(cls, name, dtype, shape, values, *, byteorder):
        """Encode explicitly typed values; an input outside its storage refuses."""
        fmt = format_record(dtype)
        shape = tuple(shape)
        if any(type(n) is not int or n < 1 for n in shape):
            raise ValueError("typed reference tensor requires explicit positive static extents")
        values = tuple(values)
        if len(values) != math.prod(shape):
            raise ValueError("typed reference tensor requires every original element")
        if byteorder not in {"little", "big"}:
            raise ValueError("typed reference tensor requires explicit byte order")
        size = fmt["element_bits"] // 8
        if fmt["kind"] == "float_ieee":
            if any(type(value) not in {int, float} or not math.isfinite(value) for value in values):
                raise ValueError("finite f32 reference requires finite numeric values")
            order = "<" if byteorder == "little" else ">"
            data = b"".join(struct.pack(order + "f", round_f32(value)) for value in values)
        else:
            if any(type(value) is not int for value in values):
                raise ValueError("integer reference storage forbids implicit floating casts")
            data = b"".join(
                integer_project(value, dtype, "bounded_exact").to_bytes(size, byteorder, signed=True)
                for value in values
            )
        tensor = cls(name, dtype, shape, data, byteorder)
        tensor.verify()
        return tensor
