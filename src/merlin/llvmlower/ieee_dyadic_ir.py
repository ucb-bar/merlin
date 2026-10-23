"""IEEE bit decoding and integer RNE of a bounded significand with runtime exponent.

No example values narrow the domain. The caller bounds the signed significand
by 2**60; shifts, normalization, subnormal RNE and overflow are integer work.
"""

from merlin.llvmlower.fixed_point_ir import FORMAT


def finite_dtype_grid(token):
    fraction, exponent, bias = FORMAT[token]
    minimum = 1 - bias - fraction
    maximum = ((1 << (fraction + 1)) - 1) << (((1 << exponent) - 2 - bias - fraction) - minimum)
    return minimum, maximum


def decode(value, emit):
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import IntegerType, i64

    from merlin.llvmlower._ir import constants

    fraction, exponent, bias = FORMAT[str(value.type)]
    if fraction > 24:
        raise ValueError("significand product proof is limited to binary32 or narrower")
    storage = IntegerType(1 + fraction + exponent)

    c = constants(emit, i64)
    raw = emit(arith.ExtUIOp(emit(arith.BitcastOp(value, storage)), i64))
    field = emit(arith.AndIOp(emit(arith.ShRUIOp(raw, c(fraction))), c((1 << exponent) - 1)))
    negative = emit(arith.CmpiOp(emit(arith.AndIOp(raw, c(1 << (fraction + exponent)))), c(0), "ne"))
    normal = emit(arith.CmpiOp(field, c(0), "ne"))
    significand = emit(
        arith.OrIOp(
            emit(arith.AndIOp(raw, c((1 << fraction) - 1))), emit(arith.SelectOp(normal, c(1 << fraction), c(0)))
        )
    )
    special = emit(arith.CmpiOp(field, c((1 << exponent) - 1), "eq"))
    significand = emit(arith.SelectOp(special, c(0), significand))
    signed = emit(arith.SelectOp(negative, emit(arith.SubiOp(c(0), significand)), significand))
    power = emit(arith.SubiOp(emit(arith.SelectOp(normal, field, c(1))), c(bias + fraction)))
    return signed, power, negative


def encode(value, power, result_type, emit, *, zero_negative=None):
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import IntegerType, i128

    from merlin.llvmlower._ir import constants

    fraction, exponent, bias = FORMAT[str(result_type)]
    storage = IntegerType(1 + fraction + exponent)
    if value.type != i128:
        value = emit(arith.ExtSIOp(value, i128))
    if power.type != i128:
        power = emit(arith.ExtSIOp(power, i128))

    c = constants(emit, i128)

    def b(cls, a, d):
        return emit(cls(a, d))

    def choose(p, a, d):
        return emit(arith.SelectOp(p, a, d))

    def cmp(a, d, predicate):
        return emit(arith.CmpiOp(a, d, predicate))

    zero, one = c(0), c(1)
    negative = cmp(value, zero, "slt")
    magnitude = choose(negative, b(arith.SubiOp, zero, value), value)
    leading = zero
    for step in (32, 16, 8, 4, 2, 1):
        trial = b(arith.AddiOp, leading, c(step))
        leading = choose(cmp(b(arith.ShRUIOp, magnitude, trial), zero, "ne"), trial, leading)
    actual_exponent = b(arith.AddiOp, leading, power)
    drop = b(arith.SubiOp, leading, c(fraction))
    minimum = b(arith.SubiOp, c(1 - bias - fraction), power)
    drop = choose(cmp(drop, minimum, "slt"), minimum, drop)
    # A significand below 2**60 is strictly below half a quantum when drop>61.
    too_small = cmp(drop, c(61), "sgt")
    safe_drop = choose(too_small, c(61), drop)
    right = choose(cmp(safe_drop, zero, "sge"), safe_drop, zero)
    left = choose(cmp(safe_drop, zero, "slt"), b(arith.SubiOp, zero, safe_drop), zero)
    numerator = b(arith.ShLIOp, magnitude, left)
    quotient = b(arith.ShRUIOp, numerator, right)
    remainder = b(arith.SubiOp, numerator, b(arith.ShLIOp, quotient, right))
    divisor = b(arith.ShLIOp, one, right)
    twice = b(arith.ShLIOp, remainder, one)
    tie = cmp(twice, divisor, "eq")
    odd = cmp(b(arith.AndIOp, quotient, one), zero, "ne")
    increment = b(arith.OrIOp, cmp(twice, divisor, "ugt"), b(arith.AndIOp, tie, odd))
    rounded = b(arith.AddiOp, quotient, choose(increment, one, zero))
    carry = cmp(rounded, c(1 << (fraction + 1)), "uge")
    rounded = choose(carry, b(arith.ShRUIOp, rounded, one), rounded)
    actual_exponent = b(arith.AddiOp, actual_exponent, choose(carry, one, zero))
    normal = cmp(actual_exponent, c(1 - bias), "sge")
    encoded_exponent = choose(
        normal, b(arith.AddiOp, actual_exponent, c(bias)), choose(cmp(rounded, c(1 << fraction), "uge"), one, zero)
    )
    raw = b(
        arith.OrIOp, b(arith.ShLIOp, encoded_exponent, c(fraction)), b(arith.AndIOp, rounded, c((1 << fraction) - 1))
    )
    raw = choose(cmp(encoded_exponent, c((1 << exponent) - 1), "uge"), c(((1 << exponent) - 1) << fraction), raw)
    raw = choose(b(arith.OrIOp, too_small, cmp(magnitude, zero, "eq")), zero, raw)
    if zero_negative is not None:
        negative = choose(cmp(magnitude, zero, "eq"), zero_negative, negative)
    raw = b(arith.OrIOp, raw, choose(negative, c(1 << (fraction + exponent)), zero))
    return emit(arith.BitcastOp(emit(arith.TruncIOp(raw, storage)), result_type))
