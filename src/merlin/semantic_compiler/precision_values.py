"""Checked logical-value conversion at an explicitly selected precision boundary.

The caller supplies the numerical and admitted-code facts. This module does
not choose a precision policy or lay out device memory. Quantization remains
an observable transformation, separate from semantic equivalence rewrites.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from merlin.runtime.fp8_formats import (
    canonical_float,
    codes_to_f32,
    float_format_params,
    float_to_codes,
    storage_bits,
)


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
    underflow_rounding: str = "flush-before-rounding"
    column_scale_exponents: tuple[tuple[int, int, int], ...] = ()
    scale_axis: int = 1

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
        if self.underflow_rounding not in {"flush-before-rounding", "round-normal-significand-then-flush"}:
            raise ValueError("unsupported underflow rounding policy")
        if self.underflow_rounding != "flush-before-rounding" and (not self.flush_below_normal or self.direction != "encode"):
            raise ValueError("normal-significand underflow rounding requires a flushing encode")
        previous_column = 0
        for start, stop, exponent in self.column_scale_exponents:
            if (any(type(value) is not int for value in (start, stop, exponent))
                    or start != previous_column or stop <= start or not -64 <= exponent <= 64):
                raise ValueError("column scales must form a contiguous partition with bounded exponents")
            if self.direction == "encode" and not -32 <= exponent <= 32:
                raise ValueError("column encode scale outside [-32,32]")
            previous_column = stop
        if self.column_scale_exponents and (self.pre_encode_scale_exponent or self.post_decode_scale_exponent):
            raise ValueError("choose scalar or column scale exponents")
        if type(self.scale_axis) is not int or self.scale_axis not in (0, 1):
            raise ValueError("scale axis must be 0 or 1")
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
        if self.underflow_rounding != "flush-before-rounding":
            record["underflow_rounding"] = self.underflow_rounding
        if self.column_scale_exponents:
            record["column_scale_exponents"] = [list(panel) for panel in self.column_scale_exponents]
            if self.scale_axis != 1:
                record["scale_axis"] = self.scale_axis
        return record

    def scalar_panel(self, exponent: int) -> "FloatBoundarySpec":
        from dataclasses import replace
        return replace(self, column_scale_exponents=(), **{
            "pre_encode_scale_exponent" if self.direction == "encode" else "post_decode_scale_exponent": exponent})

    def _column_convert(self, values: np.ndarray) -> np.ndarray:
        if values.ndim != 2 or values.shape[self.scale_axis] != self.column_scale_exponents[-1][1]:
            raise ValueError("column scale partition differs from complete tensor shape")
        slices = [(slice(start, stop), slice(None)) if self.scale_axis == 0 else (slice(None), slice(start, stop))
                  for start, stop, _exponent in self.column_scale_exponents]
        return np.concatenate([getattr(self.scalar_panel(exponent), self.direction)(values[window])
                               for window, (_start, _stop, exponent) in zip(slices, self.column_scale_exponents, strict=True)], axis=self.scale_axis)

    def encode(self, source: np.ndarray) -> np.ndarray:
        """Return logical code elements, before any target physical packing."""
        if self.direction != "encode" or not isinstance(source, np.ndarray) or source.dtype != np.float32:
            raise ValueError("encode requires a float32 source array and an encode policy")
        if not np.isfinite(source).all():
            raise ValueError("nonfinite source is outside the selected boundary policy")
        if self.column_scale_exponents:
            return self._column_convert(source)
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
            threshold = min_value
            if self.underflow_rounding == "round-normal-significand-then-flush":
                threshold *= 1.0 - 2.0 ** (-mb - 2)
                values = np.where((np.abs(values) >= threshold) & (np.abs(values) < min_value),
                                  np.copysign(np.float32(min_value), values), values)
            values = np.where(np.abs(values) < threshold, np.copysign(np.float32(0), values), values)
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
        if self.column_scale_exponents:
            return self._column_convert(codes)
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
