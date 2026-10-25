"""Exact integer residual coefficients lowered to bounded constant K tiles.

Radix digits describe the coefficient for audit. Physical K concatenation uses
additive digits: positional digits alone would require unmodeled accumulator
shifts. Every returned tile is an ordinary signed multiply/add of the residual.
"""

from __future__ import annotations

from fractions import Fraction


def integer_coefficient(value):
    try:
        coefficient = Fraction(value)
    except (ValueError, OverflowError, TypeError) as error:
        raise ValueError("K-extension coefficient must be a finite exact integer") from error
    if coefficient.denominator != 1:
        raise ValueError("K-extension coefficient must be an exact integer")
    return coefficient.numerator


def decompose_coefficient(value, *, digit_bits):
    """Return signed-magnitude radix digits, least significant first."""
    coefficient = integer_coefficient(value)
    if type(digit_bits) is not int or digit_bits < 2:
        raise ValueError("K-extension requires a declared signed digit width")
    radix = 1 << (digit_bits - 1)
    remaining, digits = abs(coefficient), []
    while remaining:
        remaining, digit = divmod(remaining, radix)
        digits.append(digit if coefficient >= 0 else -digit)
    return {"coefficient": coefficient, "radix": radix, "digits": digits or [0]}


def lower_k_extension(
    value, *, source_k, residual_width, digit_bits, max_extended_k, accumulator_bits, source_bound, residual_range
):
    """Produce constant diagonal digit tiles and an all-prefix extended-K proof.

    A tile contributes ``residual @ (digit * identity)``. The additive digits
    sum to the exact coefficient; their absolute sum proves all new partials.
    Target panel layout and instruction emission remain provider-owned.
    """
    record = decompose_coefficient(value, digit_bits=digit_bits)
    coefficient = record["coefficient"]
    if any(type(x) is not int or x <= 0 for x in (source_k, residual_width, max_extended_k, accumulator_bits)):
        raise ValueError("K-extension extents and accumulator width must be positive")
    maximum = record["radix"] - 1
    count = (abs(coefficient) + maximum - 1) // maximum
    extended = source_k + count * residual_width
    if extended > max_extended_k:
        raise ValueError("K-extension exceeds the declared extended-K capacity")
    low, high = residual_range
    if type(low) is not int or type(high) is not int or low > high:
        raise ValueError("K-extension needs a declared integer residual range")
    # The original prefix bound is supplied by a source proof, not a final sum.
    if type(source_bound) is not int or source_bound < 0:
        raise ValueError("K-extension needs a nonnegative source partial bound")
    bound = source_bound + abs(coefficient) * max(abs(low), abs(high))
    if bound >= 1 << (accumulator_bits - 1):
        raise ValueError("extended-K partial sum can overflow the accumulator")
    sign = -1 if coefficient < 0 else 1
    full, tail = divmod(abs(coefficient), maximum)
    tiles = [sign * maximum] * full + ([sign * tail] if tail else [])
    return {
        **record,
        "tile_digits": tiles,
        "tile_form": "digit_times_identity",
        "residual_width": residual_width,
        "source_k": source_k,
        "extended_k": extended,
        "partial_absolute_bound": bound,
        "accumulator_bits": accumulator_bits,
    }
