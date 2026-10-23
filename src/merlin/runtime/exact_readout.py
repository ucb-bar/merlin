"""Full-width int32 tiles followed by one exact host quantization.

The rational coefficient already includes division by the output scale. No
floating multiplication, intermediate saturation or accumulator cast occurs.
"""

from fractions import Fraction

import numpy as np


def round_rational_codes(values, zero_points, quant_min, quant_max):
    """The shared, single quantization boundary for rational references and tiles."""
    values, zeros = np.broadcast_arrays(np.asarray(values, dtype=object), zero_points)
    output = [
        min(quant_max, max(quant_min, round(v) + _integer(z))) for v, z in zip(values.flat, zeros.flat, strict=True)
    ]
    return np.asarray(output, dtype=np.int64).reshape(values.shape)


def _integer(value):
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, str) and value and value.lstrip("-").isascii() and value.lstrip("-").isdigit():
        return int(value)
    raise ValueError("exact readout parameters must be integers or decimal integer strings")


def requantize_int32(
    accumulator,
    *,
    scale_numerator,
    scale_denominator,
    zero_point=0,
    quant_min=-128,
    quant_max=127,
    rounding="half_to_even",
    axis=-1,
):
    """Return int64 storage codes with the input shape, using exact RNE then clamp.

    Scalars broadcast; one-dimensional parameters broadcast over ``axis``
    (the last axis by default). Every input must be an integer in int32 range.
    Python integers/decimal strings permit arbitrarily large rational factors.
    ``zero_point`` is added AFTER RNE, including at ties with an odd zero point.
    """
    if rounding != "half_to_even":
        raise ValueError("exact readout requires half_to_even rounding")
    low, high = _integer(quant_min), _integer(quant_max)
    if not -(2**31) <= low <= high <= 2**31 - 1:
        raise ValueError("quant_min/quant_max must fit int32")
    values = np.asarray(accumulator, dtype=object)
    integers = np.array([_integer(v) for v in values.flat], dtype=object).reshape(values.shape)
    if any(v < -(2**31) or v > 2**31 - 1 for v in integers.flat):
        raise ValueError("full-width readout requires int32 accumulators without overflow")

    def parameter(value, kind):
        data = np.asarray(value, dtype=object)
        if data.ndim > 1 or not data.size:
            raise ValueError("readout parameters must be nonempty scalars or channel vectors")
        data = np.array([_integer(v) for v in data.flat], dtype=object)
        if kind == "denominator" and any(v <= 0 for v in data):
            raise ValueError("scale denominators must be positive")
        if kind == "zero_point" and any(v < low or v > high for v in data):
            raise ValueError("zero_point must be within quant_min/quant_max")
        if data.size == 1:
            return np.broadcast_to(data.reshape(()), values.shape)
        if not values.ndim or not -values.ndim <= axis < values.ndim or data.size != values.shape[axis]:
            raise ValueError("readout channel parameter extent does not match the accumulator axis")
        shape = [1] * values.ndim
        shape[axis] = data.size
        return np.broadcast_to(data.reshape(shape), values.shape)

    numerators = parameter(scale_numerator, "numerator")
    denominators = parameter(scale_denominator, "denominator")
    zeros = parameter(zero_point, "zero_point")
    if any(d <= 0 for d in denominators.flat):
        raise ValueError("scale denominators must be positive")
    if any(z < low or z > high for z in zeros.flat):
        raise ValueError("zero_point must be within quant_min/quant_max")
    products = np.asarray(
        [Fraction(v * n, d) for v, n, d in zip(integers.flat, numerators.flat, denominators.flat, strict=True)],
        dtype=object,
    ).reshape(values.shape)
    return round_rational_codes(products, zeros, low, high)


def prove_fp32_products(scale, accumulator_min, accumulator_max):
    """Sufficient interval proof of exact int->fp32 conversion AND multiplication.

    This deliberately declines some exact sparse domains. Output equivalence
    alone never proves this property. The returned premises belong in placement.
    """
    from merlin.targetgen.exact_arithmetic import rational, round_binary

    lo, hi = _integer(accumulator_min), _integer(accumulator_max)
    value = rational(scale)
    if lo > hi or lo < -(2**31) or hi > 2**31 - 1:
        raise ValueError("invalid reachable int32 accumulator interval")
    bound = max(abs(lo), abs(hi))
    if bound > 2**24 or rational(round_binary(value)) != value:
        raise ValueError("fp32 product exactness is not proven: conversion or scale rounds")
    numerator, denominator = abs(value.numerator), value.denominator
    if denominator & (denominator - 1):
        raise ValueError("fp32 product exactness is not proven: nonbinary scale")
    while numerator and numerator % 2 == 0:
        numerator //= 2
    if (bound * numerator).bit_length() > 24 and not (bound == 2**24 and numerator == 1):
        raise ValueError("fp32 product exactness is not proven: product mantissa rounds")
    # Exact endpoint and minimum-step checks also exclude overflow/underflow.
    for point in (value, value * lo, value * hi):
        if rational(round_binary(point)) != point:
            raise ValueError("fp32 product exactness is not proven: exponent range")
    return dict(
        status="proven",
        method="integer_mantissa_bound",
        accumulator_min=lo,
        accumulator_max=hi,
        scale_numerator=str(value.numerator),
        scale_denominator=str(value.denominator),
        max_significand=bound * numerator,
    )
