"""Derive a small integer-only host encoder from a selected float boundary.

The table is generated from the declared format and admissible codes. It does
not choose a quantization policy or make the transformation quality preserving.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

import numpy as np

from merlin.runtime.fp8_formats import codes_to_f32, float_format_params, storage_bits

from .precision_values import FloatBoundarySpec


def _f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


@dataclass(frozen=True)
class FiniteF32Codebook:
    """Positive finite grid and RNE transitions for a symmetric byte format."""

    positive_codes: tuple[int, ...]
    midpoint_bits: tuple[int, ...]
    zero_gate_bits: int
    admitted_mask: tuple[int, ...]
    canonicalize_negative_zero: bool = False

    def record(self) -> dict:
        record = {
            "positive_codes": list(self.positive_codes),
            "midpoint_bits": list(self.midpoint_bits),
            "zero_gate_bits": self.zero_gate_bits,
            "admitted_mask": list(self.admitted_mask),
        }
        if self.canonicalize_negative_zero:
            record["canonicalize_negative_zero"] = True
        return record


def render_finite_f32_codebook_c(name: str, table: FiniteF32Codebook) -> list[str]:
    """Render one checked codebook for the shared C finite-float encoder."""
    if not isinstance(name, str) or not name.isascii() or not name.isidentifier():
        raise ValueError("C codebook name must be an ASCII identifier")
    if (
        not isinstance(table, FiniteF32Codebook)
        or not 1 < len(table.positive_codes) <= 127
        or len(table.midpoint_bits) != len(table.positive_codes) - 1
        or len(table.admitted_mask) != 8
        or type(table.zero_gate_bits) is not int
        or not 0 < table.zero_gate_bits < 0x7F800000
        or any(type(code) is not int or not 0 < code < 128 for code in table.positive_codes)
        or any(type(bits) is not int or not 0 < bits < 0x7F800000 for bits in table.midpoint_bits)
        or any(type(word) is not int or not 0 <= word <= 0xFFFFFFFF for word in table.admitted_mask)
        or type(table.canonicalize_negative_zero) is not bool
        or any(left >= right for left, right in zip(table.positive_codes, table.positive_codes[1:]))
        or any(left >= right for left, right in zip(table.midpoint_bits, table.midpoint_bits[1:]))
        or table.midpoint_bits[0] <= table.zero_gate_bits
    ):
        raise ValueError("invalid finite-float C codebook")

    def rows(values: tuple[int, ...], width: int) -> list[str]:
        return [
            "    " + ", ".join(f"0x{value:0{width}x}u" for value in values[start:start + 12]) + ","
            for start in range(0, len(values), 12)
        ]

    codes = f"{name}_positive_codes"
    midpoints = f"{name}_midpoint_bits"
    return [
        f"static const uint8_t {codes}[] = {{",
        *rows(table.positive_codes, 2),
        "};",
        f"static const uint32_t {midpoints}[] = {{",
        *rows(table.midpoint_bits, 8),
        "};",
        f"static const merlin_f32_u8_codebook {name} = {{",
        f"    {codes}, {midpoints},",
        f"    {len(table.positive_codes)}u, 0x{table.zero_gate_bits:08x}u,",
        "    {" + ", ".join(f"0x{word:08x}u" for word in table.admitted_mask) + "},",
        f"    {int(table.canonicalize_negative_zero)}u",
        "};",
    ]


def finite_f32_codebook(spec: FloatBoundarySpec) -> FiniteF32Codebook:
    """Build the finite-normal, subnormal-flushing byte encoder's C table.

    Grid thresholds are mapped back through the selected power-of-two input
    scale. All resulting midpoints must be exactly representable in binary32.
    The C encoder compares positive binary32 bit patterns against these exact
    transitions, then applies the source sign and the admitted-code mask.
    """
    if (
        not isinstance(spec, FloatBoundarySpec)
        or spec.direction != "encode"
        or storage_bits(spec.format) != 8
        or not spec.flush_below_normal
    ):
        raise ValueError("host codebook requires a flushing eight-bit encode boundary")
    eb, mb, bias, scheme = float_format_params(spec.format)
    if 1 + eb + mb != 8 or scheme not in {"ieee", "e4m3fn"}:
        raise ValueError("host codebook requires a signed symmetric float grid")
    top = (1 << eb) - 1
    maximum = (
        (top - 1 if scheme == "ieee" else top)
        if spec.max_normal_exponent_code is None
        else spec.max_normal_exponent_code
    )
    codes = []
    for code in range(1, 128):
        exponent = (code >> mb) & top
        if spec.min_normal_exponent_code <= exponent <= maximum:
            decoded = float(codes_to_f32(np.array([code], dtype=np.uint8), spec.format)[0])
            if math.isfinite(decoded) and decoded > 0:
                codes.append((code, decoded))
    if not codes or any(left[1] >= right[1] for left, right in zip(codes, codes[1:])):
        raise ValueError("selected float grid is empty or not monotone")
    inverse_scale = -spec.pre_encode_scale_exponent
    first = math.ldexp(codes[0][1], inverse_scale)
    first_bits = _f32_bits(first)
    if first_bits != _f32_bits(math.ldexp(2.0 ** (spec.min_normal_exponent_code - bias), inverse_scale)):
        raise ValueError("selected float grid has a different normal threshold")
    if struct.unpack("<f", struct.pack("<I", first_bits))[0] != first:
        raise ValueError("scaled normal threshold is not exact in binary32")
    midpoints = []
    for left, right in zip(codes, codes[1:]):
        midpoint = math.ldexp((left[1] + right[1]) / 2.0, inverse_scale)
        bits = _f32_bits(midpoint)
        if struct.unpack("<f", struct.pack("<I", bits))[0] != midpoint:
            raise ValueError("float boundary midpoint is not exact in binary32")
        midpoints.append(bits)
    if any(left >= right for left, right in zip(midpoints, midpoints[1:])):
        raise ValueError("float boundary transitions are not monotone")
    admitted = [0xFFFFFFFF] * 8 if not spec.admitted_code_ranges else [0] * 8
    for low, high in spec.admitted_code_ranges:
        for code in range(low, high + 1):
            admitted[code >> 5] |= 1 << (code & 31)
    return FiniteF32Codebook(
        tuple(code for code, _ in codes),
        tuple(midpoints),
        first_bits,
        tuple(admitted),
        spec.canonicalize_negative_zero,
    )
