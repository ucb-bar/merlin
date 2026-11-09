"""Independent evaluator for admitted integer, strict f32 arithmetic and opaque-bit operations.

This is a verification tool. Native selection never calls it, and unknown
numerical policies or operations fail instead of acquiring guessed semantics.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from itertools import product as index_product
from math import isnan, prod
from struct import pack, unpack

from .model import IndexMap, KernelRequest, SemanticNode, TensorType, pointwise_identity_maps

_I32_MIN = -(1 << 31)
_I32_MAX = (1 << 31) - 1
_BF16_RELU_ATTRS = {"input_domain": "finite-normal-or-zero-bf16", "numeric_semantics": "bf16-sign-clamp-v1"}


def _i32(value: int) -> int:
    if type(value) is not int or not _I32_MIN <= value <= _I32_MAX:
        raise ValueError("exact-i32 value is outside its admitted domain")
    return value


def _wrap_i32(value: int) -> int:
    return (value + (1 << 31)) % (1 << 32) - (1 << 31)


def _matmul_maps() -> tuple[IndexMap, ...]:
    return (
        IndexMap(3, ((1, 0, 0), (0, 0, 1)), (0, 0)),
        IndexMap(3, ((0, 0, 1), (0, 1, 0)), (0, 0)),
        IndexMap(3, ((1, 0, 0), (0, 1, 0)), (0, 0)),
        IndexMap(3, ((1, 0, 0), (0, 1, 0)), (0, 0)),
    )


def _f32_add_rne(left: int, right: int) -> int:
    """Add two binary32 bit patterns exactly, then round once to nearest even."""
    signs = (left >> 31, right >> 31)
    fields = ((left >> 23 & 255, left & 0x7FFFFF), (right >> 23 & 255, right & 0x7FFFFF))
    if any(exponent == 255 and fraction for exponent, fraction in fields):
        raise ValueError("f32 addition NaN payload is not qualified by this reference")
    if any(exponent == 255 for exponent, _ in fields):
        if all(exponent == 255 for exponent, _ in fields) and signs[0] != signs[1]:
            raise ValueError("f32 addition NaN result is not qualified by this reference")
        return left if fields[0][0] == 255 else right

    def component(bits: int) -> tuple[int, int]:
        exponent = bits >> 23 & 255
        fraction = bits & 0x7FFFFF
        return ((1 << 23) | fraction, exponent - 150) if exponent else (fraction, -149)

    left_mantissa, left_exponent = component(left)
    right_mantissa, right_exponent = component(right)
    exponent = min(left_exponent, right_exponent)
    exact = (-1 if signs[0] else 1) * (left_mantissa << (left_exponent - exponent)) + (-1 if signs[1] else 1) * (
        right_mantissa << (right_exponent - exponent)
    )
    if exact == 0:
        return 0x80000000 if left_mantissa == right_mantissa == 0 and signs == (1, 1) else 0
    sign = 0x80000000 if exact < 0 else 0
    magnitude = abs(exact)

    def round_shift(value: int, shift: int) -> int:
        if shift <= 0:
            return value << -shift
        quotient, remainder = divmod(value, 1 << shift)
        halfway = 1 << (shift - 1)
        return quotient + (remainder > halfway or remainder == halfway and bool(quotient & 1))

    top_exponent = magnitude.bit_length() - 1 + exponent
    if top_exponent >= -126:
        significand = round_shift(magnitude, top_exponent - 23 - exponent)
        if significand == 1 << 24:
            significand >>= 1
            top_exponent += 1
        if top_exponent > 127:
            return sign | 0x7F800000
        return sign | ((top_exponent + 127) << 23) | (significand - (1 << 23))
    fraction = round_shift(magnitude, -149 - exponent)
    return sign | (0x00800000 if fraction == 1 << 23 else fraction)


def _f32_div_rne(left: int, right: int) -> int:
    """Divide binary32 values as an exact rational, then round once to nearest even."""
    sign = (left ^ right) & 0x80000000
    left_exponent, right_exponent = (left >> 23) & 255, (right >> 23) & 255
    left_fraction, right_fraction = left & 0x7FFFFF, right & 0x7FFFFF
    if (left_exponent == 255 and left_fraction) or (right_exponent == 255 and right_fraction):
        raise ValueError("f32 division NaN payload is not qualified by this reference")
    left_zero = left_exponent == 0 and left_fraction == 0
    right_zero = right_exponent == 0 and right_fraction == 0
    if (left_zero and right_zero) or (left_exponent == right_exponent == 255):
        raise ValueError("f32 division NaN result is not qualified by this reference")
    if right_zero or left_exponent == 255:
        return sign | 0x7F800000
    if left_zero or right_exponent == 255:
        return sign

    def component(exponent: int, fraction: int) -> tuple[int, int]:
        return ((1 << 23) | fraction, exponent - 150) if exponent else (fraction, -149)

    left_mantissa, left_power = component(left_exponent, left_fraction)
    right_mantissa, right_power = component(right_exponent, right_fraction)
    power = left_power - right_power
    numerator = left_mantissa << max(power, 0)
    denominator = right_mantissa << max(-power, 0)

    def round_ratio(numer: int, denom: int) -> int:
        quotient, remainder = divmod(numer, denom)
        return quotient + (2 * remainder > denom or 2 * remainder == denom and bool(quotient & 1))

    exponent = numerator.bit_length() - denominator.bit_length()
    if numerator < denominator << exponent if exponent >= 0 else numerator << -exponent < denominator:
        exponent -= 1
    if exponent >= -126:
        shift = 23 - exponent
        significand = round_ratio(
            numerator << max(shift, 0),
            denominator << max(-shift, 0),
        )
        if significand == 1 << 24:
            significand >>= 1
            exponent += 1
        if exponent > 127:
            return sign | 0x7F800000
        return sign | ((exponent + 127) << 23) | (significand - (1 << 23))
    fraction = round_ratio(numerator << 149, denominator)
    return sign | (0x00800000 if fraction == 1 << 23 else fraction)


def _f32_reduce_maps(rank: int, axis: int) -> tuple[IndexMap, ...]:
    identity = IndexMap(
        rank, tuple(tuple(int(dim == loop) for loop in range(rank)) for dim in range(rank)), (0,) * rank
    )
    projection = IndexMap(
        rank,
        tuple(tuple(int(dim == loop) for loop in range(rank)) for dim in range(rank) if dim != axis),
        (0,) * (rank - 1),
    )
    return identity, projection, projection


def _checked_f32_pointwise_maps(node: SemanticNode, operands: tuple[TensorValue, ...]) -> tuple[IndexMap, ...]:
    """Recheck a static singleton projection without trusting the source recognizer."""
    rank = len(node.type.shape)
    if len(node.index_maps) != len(operands) + 1:
        raise ValueError("f32 pointwise needs an index map for every tensor")
    output_map = node.index_maps[-1]
    if output_map != pointwise_identity_maps(rank, 1)[0]:
        raise ValueError("f32 pointwise output map is not identity")
    for operand, mapping in zip(operands, node.index_maps[:-1], strict=True):
        if mapping.loop_rank != rank or len(mapping.coefficients) != len(operand.type.shape):
            raise ValueError("f32 pointwise input map has the wrong rank")
        last_dimension = -1
        for extent, row, offset in zip(
            operand.type.shape,
            mapping.coefficients,
            mapping.offsets,
            strict=True,
        ):
            if not any(row) and offset == 0 and extent == 1:
                continue
            positions = [dimension for dimension, coefficient in enumerate(row) if coefficient == 1]
            if (
                offset != 0
                or len(positions) != 1
                or row != tuple(int(dimension == positions[0]) for dimension in range(rank))
                or positions[0] <= last_dimension
                or extent != node.type.shape[positions[0]]
            ):
                raise ValueError("f32 pointwise input map is not a singleton projection")
            last_dimension = positions[0]
    return node.index_maps[:-1]


def _flat_index(indices: tuple[int, ...], shape: tuple[int, ...]) -> int:
    flat = 0
    for index, extent in zip(indices, shape, strict=True):
        flat = flat * extent + index
    return flat


@dataclass(frozen=True)
class TensorValue:
    type: TensorType
    elements: tuple[int, ...]

    def __post_init__(self) -> None:
        admitted = {
            ("i1", "boolean"),
            ("i32", "exact-i32"),
            ("i32", "i32-wrap-k-ascending"),
            ("i64", "signed-i64"),
            ("i8", "signed-i8"),
            ("i8", "raw-byte-copy"),
        }
        raw_float = self.type.dtype in {"fp8_e4m3", "bf16", "f32", "f64"}
        if not raw_float and (self.type.dtype, self.type.numerical_policy) not in admitted:
            raise ValueError("reference evaluator has no admitted numerical policy")
        if len(self.elements) != prod(self.type.shape):
            raise ValueError("reference tensor size differs from its type")
        for element in self.elements:
            if self.type.dtype == "i1":
                if type(element) is not int or element not in (0, 1):
                    raise ValueError("boolean tensor elements must be zero or one")
            elif raw_float or self.type.numerical_policy == "raw-byte-copy":
                maximum = {"i8": 255, "fp8_e4m3": 255, "bf16": 65535, "f32": (1 << 32) - 1, "f64": (1 << 64) - 1}[
                    self.type.dtype
                ]
                if type(element) is not int or not 0 <= element <= maximum:
                    raise ValueError("raw-bit value is outside its admitted domain")
            elif self.type.dtype == "i8":
                if type(element) is not int or not -128 <= element <= 127:
                    raise ValueError("signed-i8 value is outside its admitted domain")
            elif self.type.dtype == "i64":
                if type(element) is not int or not -(1 << 63) <= element < (1 << 63):
                    raise ValueError("signed-i64 value is outside its admitted domain")
            else:
                _i32(element)


