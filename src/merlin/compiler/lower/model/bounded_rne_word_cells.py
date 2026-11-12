"""Exact ordered-binary32 preimages of a typed saturated RNE signed byte.

This is a numeric observer representation, not an output approximation or a
target instruction. It supplies no floating-mode, source-use or storage policy.
"""

import struct
from dataclasses import dataclass, field

from merlin.compiler.lower.model.source_expression_interval import (
    ClosedScalarObserver,
    IntervalEffectContract,
    close_scalar_i8_observer,
    validate_closed_scalar_observer,
)


def _key(value: float) -> int:
    word = struct.unpack("<I", struct.pack("<f", value))[0]
    return (~word & 0xFFFFFFFF) if word >> 31 else word ^ 0x80000000


def _ranges() -> tuple[tuple[int, int], ...]:
    result = []
    for integer in range(-128, 128):
        # Half-integer boundaries in this domain are exactly binary32.
        lower = _key(float("-inf")) if integer == -128 else _key(integer - 0.5) + (integer & 1)
        upper = _key(float("inf")) if integer == 127 else _key(integer + 0.5) - (integer & 1)
        result.append((lower, upper))
    assert all(a[1] + 1 == b[0] for a, b in zip(result, result[1:]))
    return tuple(result)


def _validate_observer(observer, effects):
    validate_closed_scalar_observer(observer)
    rebuilt = close_scalar_i8_observer(observer.cut, observer.endpoint, effects=effects)
    if rebuilt.expression != observer.expression or rebuilt.quant_factor_bits != observer.quant_factor_bits:
        raise ValueError("source observer numerical fields differ from the actual typed source")


@dataclass(frozen=True)
class BoundedRneWordCells:
    ranges: tuple[tuple[int, int], ...]
    expression_sha256: str
    quant_factor_word: int
    _observer: ClosedScalarObserver = field(repr=False)
    _effects: IntervalEffectContract = field(repr=False)


def prepare_bounded_rne_word_cells(observer: ClosedScalarObserver, *, effects: IntervalEffectContract):
    """Retain complete typed source/effect witnesses and exact observer cells.

    The source theorem is the existing complete [-128,127] clamp/ties-even i8
    recognition. Every non-NaN binary32 word belongs to exactly one preimage;
    both signed zeros belong to integer zero, and infinities saturate. NaN keys
    belong to none. Odd results exclude exact half ties, even results include
    them. Deleting a second observer invocation requires explicit nontrapping
    arithmetic and unobserved flags; unsupported modes retain original source.
    """
    if type(observer) is not ClosedScalarObserver or type(effects) is not IntervalEffectContract:
        raise ValueError("complete typed saturated source observer and effect contract required")
    effects.validate()
    _validate_observer(observer, effects)
    return BoundedRneWordCells(
        _ranges(), observer.expression.canonical_sha256, observer.quant_factor_bits, observer, effects
    )


def validate_bounded_rne_word_cells(cells: BoundedRneWordCells):
    if type(cells) is not BoundedRneWordCells:
        raise ValueError("typed immutable source observer cells required")
    cells._effects.validate()
    _validate_observer(cells._observer, cells._effects)
    if (
        cells.ranges != _ranges()
        or cells.expression_sha256 != cells._observer.expression.canonical_sha256
        or cells.quant_factor_word != cells._observer.quant_factor_bits
    ):
        raise ValueError("source observer cells or retained numerical binding changed")


def binary32_word_in_rne_cell(word: int, integer: int, cells: BoundedRneWordCells) -> bool:
    """Reference membership; runtime emission performs the same integer test."""
    validate_bounded_rne_word_cells(cells)
    if type(word) is not int or not 0 <= word <= 0xFFFFFFFF or type(integer) is not int or not -128 <= integer <= 127:
        raise ValueError("binary32 word and signed-byte observer result required")
    key = (~word & 0xFFFFFFFF) if word >> 31 else word ^ 0x80000000
    lower, upper = cells.ranges[integer + 128]
    return lower <= key <= upper
