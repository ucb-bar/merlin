"""Checked logical-value conversion at an explicitly selected precision boundary.

The caller supplies the numerical and admitted-code facts. This module does
not choose a precision policy or lay out device memory. Quantization remains
an observable transformation, separate from semantic equivalence rewrites.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

import numpy as np

from merlin.common.fp8_formats import (
    canonical_float,
    codes_to_f32,
    float_format_params,
    float_to_codes,
    storage_bits,
)

# ---- values ------------------------------------------------------------------------------------


def contraction_restore_exponent(lhs_encode_exponent: int, rhs_encode_exponent: int) -> int:
    """Restore units after a dot of operands encoded with constant binary scales.

    Encoding x*2**a and y*2**b scales their products by 2**(a+b).
    The returned exponent undoes that scale after the reduction. Quantization,
    accumulator rounding and the target's rescale range remain explicit.
    """
    if any(type(value) is not int or not -32 <= value <= 32 for value in (lhs_encode_exponent, rhs_encode_exponent)):
        raise ValueError("operand encoding exponents must be integers in [-32, 32]")
    return -lhs_encode_exponent - rhs_encode_exponent


@dataclass(frozen=True)
class FloatBoundarySpec:
    """One explicit value conversion; pre-encode scaling is exact before clipping."""

    policy: str
    format: str
    direction: str
    min_normal_exponent_code: int = 1
    max_normal_exponent_code: int | None = None
    flush_below_normal: bool = False
    admitted_code_ranges: tuple[tuple[int, int], ...] = ()
    pre_encode_scale_exponent: int = 0  # Multiply source by 2**e before encoding.
    canonicalize_negative_zero: bool = False  # Explicit lossy source-boundary choice.
    post_decode_scale_exponent: int = 0  # Restore logical units after a scaled contraction.

    def __post_init__(self) -> None:
        if not self.policy or self.direction not in {"encode", "decode"}:
            raise ValueError("boundary needs a policy and encode/decode direction")
        eb, _mb, _bias, scheme = float_format_params(self.format)
        top = (1 << eb) - 1
        finite_top = top - 1 if scheme == "ieee" else top
        if (
            type(self.min_normal_exponent_code) is not int
            or self.max_normal_exponent_code is not None
            and type(self.max_normal_exponent_code) is not int
        ):
            raise ValueError("normal exponent bounds must be integers")
        maximum = finite_top if self.max_normal_exponent_code is None else self.max_normal_exponent_code
        if not 1 <= self.min_normal_exponent_code <= maximum <= finite_top:
            raise ValueError("boundary requires a finite normal exponent interval")
        if type(self.flush_below_normal) is not bool:
            raise ValueError("flush_below_normal must be boolean")
        # Only flushing enforces a raised floor; otherwise encode would still
        # emit lower normal and subnormal codes outside the declared interval.
        if self.min_normal_exponent_code != 1 and not self.flush_below_normal:
            raise ValueError("a raised normal exponent floor requires flush_below_normal")
        if type(self.pre_encode_scale_exponent) is not int or not -32 <= self.pre_encode_scale_exponent <= 32:
            raise ValueError("pre-encode scale exponent must be an integer in [-32, 32]")
        if type(self.canonicalize_negative_zero) is not bool:
            raise ValueError("negative-zero policy must be boolean")
        if self.direction == "decode" and self.canonicalize_negative_zero:
            raise ValueError("decode cannot canonicalize encoded negative zero")
        if self.direction == "decode" and self.pre_encode_scale_exponent:
            raise ValueError("decode cannot have a pre-encode scale")
        if type(self.post_decode_scale_exponent) is not int or not -64 <= self.post_decode_scale_exponent <= 64:
            raise ValueError("post-decode scale exponent must be an integer in [-64, 64]")
        if self.direction == "encode" and self.post_decode_scale_exponent:
            raise ValueError("encode cannot have a post-decode scale")
        bit_max = (1 << storage_bits(self.format)) - 1
        previous = -1
        for low, high in self.admitted_code_ranges:
            if type(low) is not int or type(high) is not int or not previous < low <= high <= bit_max:
                raise ValueError("admitted code ranges must be ordered, disjoint and in range")
            previous = high
        if self.direction == "decode" and self.admitted_code_ranges:
            raise ValueError("decode does not admit an input-code restriction")

    def record(self) -> dict:
        record = {
            "policy": self.policy,
            "format": canonical_float(self.format),
            "direction": self.direction,
            "min_normal_exponent_code": self.min_normal_exponent_code,
            "max_normal_exponent_code": self.max_normal_exponent_code,
            "flush_below_normal": self.flush_below_normal,
            "admitted_code_ranges": [list(pair) for pair in self.admitted_code_ranges],
        }
        if self.pre_encode_scale_exponent:
            record["pre_encode_scale_exponent"] = self.pre_encode_scale_exponent
        if self.canonicalize_negative_zero:
            record["canonicalize_negative_zero"] = True
        if self.post_decode_scale_exponent:
            record["post_decode_scale_exponent"] = self.post_decode_scale_exponent
        return record

    def encode(self, source: np.ndarray) -> np.ndarray:
        """Return logical code elements, before any target physical packing."""
        if self.direction != "encode" or not isinstance(source, np.ndarray) or source.dtype != np.float32:
            raise ValueError("encode requires a float32 source array and an encode policy")
        if not np.isfinite(source).all():
            raise ValueError("nonfinite source is outside the selected boundary policy")
        eb, mb, bias, scheme = float_format_params(self.format)
        top = (1 << eb) - 1
        maximum = (
            (top - 1 if scheme == "ieee" else top)
            if self.max_normal_exponent_code is None
            else self.max_normal_exponent_code
        )
        largest_mantissa = (1 << mb) - 2 if scheme == "e4m3fn" and maximum == top else (1 << mb) - 1
        max_value = (1.0 + largest_mantissa / (1 << mb)) * 2.0 ** (maximum - bias)
        # A finite binary32 value times 2**e is exact in binary64 for this
        # bounded exponent. Clip before converting back to binary32, so the
        # selected finite FP8 range saturates without an intermediate Inf.
        scaled = np.ldexp(source.astype(np.float64), self.pre_encode_scale_exponent)
        values = np.clip(scaled, -max_value, max_value).astype(np.float32)
        if self.flush_below_normal:
            min_value = 2.0 ** (self.min_normal_exponent_code - bias)
            values = np.where(np.abs(values) < min_value, np.copysign(np.float32(0), values), values)
        codes = float_to_codes(values, self.format).reshape(source.shape)
        if self.canonicalize_negative_zero:
            codes = np.where(codes == 1 << (storage_bits(self.format) - 1), 0, codes)
        if self.admitted_code_ranges:
            admitted = np.zeros(codes.shape, dtype=np.bool_)
            for low, high in self.admitted_code_ranges:
                admitted |= (codes >= low) & (codes <= high)
            if not admitted.all():
                raise ValueError("encoded operand is outside the selected target input domain")
        bits = storage_bits(self.format)
        dtype = np.dtype(f"<u{(bits + 7) // 8}")
        return codes.astype(dtype)

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Decode logical code elements; the target must separately check its result domain."""
        if (
            self.direction != "decode"
            or not isinstance(codes, np.ndarray)
            or codes.dtype.kind != "u"
            or codes.dtype.itemsize * 8 < storage_bits(self.format)
            or (codes >= (1 << storage_bits(self.format))).any()
        ):
            raise ValueError("decode requires unsigned in-range code elements and a decode policy")
        decoded = codes_to_f32(codes, self.format).reshape(codes.shape)
        if self.post_decode_scale_exponent:
            # Scale finite values in binary64, then round once to binary32.
            # Exceptional values retain their original sign/payload bits.
            finite = np.isfinite(decoded)
            decoded = decoded.copy()
            with np.errstate(over="ignore", under="ignore"):
                decoded[finite] = np.ldexp(decoded[finite].astype(np.float64), self.post_decode_scale_exponent).astype(
                    np.float32
                )
        return decoded


# ---- codegen -----------------------------------------------------------------------------------
# Derive a small integer-only host encoder from a selected float boundary.
#
# The table is generated from the declared format and admissible codes. It does
# not choose a quantization policy or make the transformation quality preserving.


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
            "    " + ", ".join(f"0x{value:0{width}x}u" for value in values[start : start + 12]) + ","
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
