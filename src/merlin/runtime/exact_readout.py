"""Declared readout policies: IEEE f32 product/RNE or exact rational host RNE.

These policies differ even for power-of-two scales above the f32 integer domain.
Neither policy is selected by a target identity or by an oracle's header macro.
"""

from fractions import Fraction

import numpy as np

EXACT_READOUT_PARAMETERS = frozenset(
    {"scale_numerator", "scale_denominator", "zero_point", "quant_min", "quant_max", "rounding"}
)


def readout_policy(parameters):
    """Resolve the declared numeric strategy, preserving legacy exact commands.

    Old commands declared exact host arithmetic through rational/quantization
    parameters. Translate those only when no explicit strategy was supplied;
    new f32 commands may carry the same parameters without changing policy.
    """
    strategy = parameters.get("readout_strategy")
    if strategy is None:
        strategy = "exact_rational_host" if not EXACT_READOUT_PARAMETERS.isdisjoint(parameters) else "f32_product_rne"
    if strategy not in ("exact_rational_host", "f32_product_rne"):
        raise ValueError(f"unknown declared readout strategy {strategy!r}")
    return strategy


def form_multiplier(numerators, denominators=1):
    """RNE of the exact declared ratio to f32, with NumPy broadcasting.

    Compose scale factors as exact rationals BEFORE calling this function. This
    avoids both stepwise f32 arithmetic and double rounding through binary64.
    An already supplied float denotes its exact binary value, not a decimal
    approximation to a different rational. Nonfinite/overflowing scales refuse.
    """
    from merlin.targetgen.exact_arithmetic import rational, round_binary

    supplied = np.asarray(numerators)
    if supplied.dtype == np.dtype(np.float32) and np.isscalar(denominators) and denominators == 1:
        if not np.isfinite(supplied).all():
            raise ValueError("multiplier must be finite f32")
        # A register value is already the exact result of f32 formation. Keep
        # repeated readouts vectorized rather than re-proving each scale bit.
        return supplied[()] if supplied.ndim == 0 else supplied
    nums, dens = np.broadcast_arrays(np.asarray(numerators, dtype=object), np.asarray(denominators, dtype=object))
    rounded = []
    for numerator, denominator in zip(nums.flat, dens.flat, strict=True):
        numerator = _integer(numerator) if isinstance(numerator, str) else numerator
        denominator = _integer(denominator) if isinstance(denominator, str) else denominator
        denominator = rational(denominator)
        if denominator <= 0:
            raise ValueError("multiplier denominators must be positive")
        rounded.append(round_binary(rational(numerator) / denominator))
    result = np.asarray(rounded, dtype=np.float32).reshape(nums.shape)
    return result[()] if result.ndim == 0 else result


def _readout_pipeline(acc, multiplier, *, convert, multiply, rne, saturate):
    """Keep the evaluator and emitted scalar code's rounding boundaries together."""
    return saturate(rne(multiply(convert(acc), multiplier)))


def _output_dtype(out_dtype):
    if isinstance(out_dtype, str) and out_dtype[:1] in ("i", "u") and out_dtype[1:].isdigit():
        out_dtype = ("int" if out_dtype[0] == "i" else "uint") + out_dtype[1:]
    dtype = np.dtype(out_dtype)
    if dtype.kind not in "iu":
        raise ValueError("readout output dtype must be integral")
    return dtype


def f32_readout(acc, multiplier, out_dtype="i32", *, relu=False, qmin=None, qmax=None, zero_point=0):
    """RNE(f32(f32(acc) * M)), then saturate to the declared integer type.

    ``M`` is formed by :func:`form_multiplier`. Arrays broadcast normally; ReLU
    applies to the accumulator before conversion. Optional quantization bounds
    and an integer zero point describe the final storage boundary. ``out_dtype=None``
    returns rounded f32 values for command stages that defer saturation.
    Products may overflow to infinity and saturate; NaNs have no integer meaning.
    """
    multiplier = form_multiplier(multiplier)
    values = np.maximum(acc, 0) if relu else acc

    def saturate(rounded):
        if np.isnan(rounded).any():
            raise ValueError("f32 readout has a NaN product")
        if out_dtype is None:
            if qmin is not None or qmax is not None or zero_point != 0:
                raise ValueError("quantization parameters require an output dtype")
            if not np.isfinite(rounded).all():
                raise ValueError("unsaturated f32 readout must be finite")
            return rounded
        dtype = _output_dtype(out_dtype)
        limits = np.iinfo(dtype)
        low = limits.min if qmin is None else max(limits.min, _integer(qmin))
        high = limits.max if qmax is None else min(limits.max, _integer(qmax))
        if low > high:
            raise ValueError("invalid readout clamp range")
        # f64 exactly holds every rounded f32 and every <=32-bit clamp endpoint.
        shifted = np.asarray(rounded, dtype=np.float64) + _integer(zero_point)
        if dtype.itemsize <= np.dtype(np.int32).itemsize:
            return np.asarray(np.clip(shifted, low, high), dtype=dtype)
        # Wide endpoints need integer clamps: float(i64.max) rounds out of range.
        return np.asarray(
            [low if x <= low else high if x >= high else int(x) for x in shifted.flat], dtype=dtype
        ).reshape(shifted.shape)

    with np.errstate(over="ignore", invalid="ignore"):
        return _readout_pipeline(
            values,
            multiplier,
            convert=lambda x: np.asarray(x, dtype=np.float32),
            multiply=lambda a, b: np.multiply(a, b, dtype=np.float32),
            rne=np.rint,
            saturate=saturate,
        )


def f32_readout_scalar(acc, multiplier, out_dtype="i32", **kwargs):
    """Scalar form of :func:`f32_readout`, returning a Python integer."""
    return int(f32_readout(acc, multiplier, out_dtype, **kwargs))


def emit_f32_readout(acc, multiplier, out_dtype, *, fp):
    """Emit the same pipeline using a caller's scalar f32/integer builder.

    The builder supplies IEEE RNE operations. Clamp before integer conversion
    to avoid an undefined overflowing float-to-int operation.
    """
    limits = np.iinfo(_output_dtype(out_dtype))
    low, high = int(limits.min), int(limits.max)
    return _readout_pipeline(
        acc,
        fp.fconst(float(form_multiplier(multiplier))),
        convert=fp.sitofp,
        multiply=fp.fmul,
        rne=fp.round_near_even,
        saturate=lambda x: fp.fb.clamp(fp.fptosi(fp.fclamp(x, float(low), float(high))), low, high),
    )


def pred_pow2_vectors(multiplier, acc_lo, acc_hi, e_range):
    """Generate pred(2**e) and adjacent integer-tie regression accumulators.

    Monotone binary searches invert the *rounded* f32 product, including its
    int-to-f32 plateaus. Return both ends of each reachable preimage and their
    immediate neighbours, deterministically, without scanning the int32 domain.
    Negative products and negative multipliers are included symmetrically.
    """
    low, high = _integer(acc_lo), _integer(acc_hi)
    if low > high or low < -(2**31) or high >= 2**31:
        raise ValueError("invalid reachable int32 accumulator interval")
    scale = form_multiplier(multiplier)
    if np.ndim(scale) or not scale:
        raise ValueError("regression vectors require a nonzero scalar multiplier")
    direction = 1 if scale > 0 else -1

    def product(acc):
        return direction * float(np.float32(np.float32(acc) * scale))

    def first(target, *, strict=False):
        left, right = low, high + 1
        while left < right:
            mid = (left + right) // 2
            if product(mid) < target or (strict and product(mid) == target):
                left = mid + 1
            else:
                right = mid
        return left

    points = set()
    for exponent in e_range:
        power = np.float32(float(Fraction(2) ** int(exponent)))
        predecessor = np.nextafter(power, np.float32(0))
        targets = (predecessor, float(power) - 0.5, float(power) + 0.5)
        for magnitude in targets:
            for sign in (-1, 1):
                target = sign * float(magnitude)
                begin, end = first(target), first(target, strict=True) - 1
                if begin <= end:
                    points.update(x for edge in (begin, end) for x in (edge - 1, edge, edge + 1) if low <= x <= high)
    return np.asarray(sorted(points), dtype=np.int64)


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
    """Prove the IEEE f32 path equals exact-rational RNE throughout an interval.

    Exact conversion AND exact multiplication are sufficient premises, stronger
    than output-code equivalence. This deliberately declines sparse domains.
    A product at pred(2**e) is valid: IEEE RNE rounds it correctly, even where a
    simulator header macro does not. This proof certifies no such macro/oracle.
    The returned arithmetic premises belong in placement.
    """
    from merlin.targetgen.exact_arithmetic import rational

    lo, hi = _integer(accumulator_min), _integer(accumulator_max)
    value = rational(scale)
    if lo > hi or lo < -(2**31) or hi > 2**31 - 1:
        raise ValueError("invalid reachable int32 accumulator interval")
    bound = max(abs(lo), abs(hi))
    if bound > 2**24 or rational(form_multiplier(value)) != value:
        raise ValueError("fp32 product exactness is not proven: conversion or scale rounds")
    numerator, denominator = abs(value.numerator), value.denominator
    if denominator & (denominator - 1):
        raise ValueError("fp32 product exactness is not proven: nonbinary scale")
    while numerator and numerator % 2 == 0:
        numerator //= 2
    if (bound * numerator).bit_length() > 24 and not (bound == 2**24 and numerator == 1):
        raise ValueError("fp32 product exactness is not proven: product mantissa rounds")
    # Exact endpoint and minimum-step checks also exclude overflow/underflow.
    for accumulator in (1, lo, hi):
        point = value * accumulator
        with np.errstate(over="ignore"):
            product = np.float32(np.float32(accumulator) * form_multiplier(value))
        if not np.isfinite(product) or rational(product) != point:
            raise ValueError("fp32 product exactness is not proven: exponent range")
    return dict(
        status="proven",
        method="integer_mantissa_bound",
        readout_policy="f32_product_rne",
        equals_policy="exact_rational_host",
        integer_rounding="half_to_even",
        accumulator_min=lo,
        accumulator_max=hi,
        scale_numerator=str(value.numerator),
        scale_denominator=str(value.denominator),
        max_significand=bound * numerator,
    )
