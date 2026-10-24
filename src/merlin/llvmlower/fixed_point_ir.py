"""IEEE conversion of bounded exact signed integer/dyadic source values."""

from fractions import Fraction

FORMAT = {"f16": (10, 5, 15), "bf16": (7, 8, 127), "f32": (23, 8, 127), "f64": (52, 11, 1023)}


def encode_fixed_integer(value, exponent, result_type, emit, *, denominator=1):
    """Emit one exact RNE to IEEE bits; |value| < 2**120 and denominator <=2**31.

    The caller proves the integer producer and retained recombination bounds.
    All rounding decisions use integer quotient/remainder. Signed zero for a
    nonzero underflow follows its exact sign; exact zero has positive sign.
    """
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import IntegerAttr, IntegerType

    from merlin.llvmlower._ir import constants

    if str(result_type) not in FORMAT or not 0 < denominator < (1 << 31):
        raise ValueError("unsupported exact IEEE boundary")
    fraction_bits, exponent_bits, bias = FORMAT[str(result_type)]
    wide = value.type
    if not isinstance(wide, IntegerType) or wide.width.data < 128 or wide.width.data & (wide.width.data - 1):
        raise ValueError("exact IEEE conversion requires a bounded power-of-two integer width of at least 128")
    width = wide.width.data
    bits = IntegerType(1 + fraction_bits + exponent_bits)

    c = constants(emit, wide)

    def binary(cls, a, b):
        return emit(cls(a, b))

    def choose(predicate, a, b):
        return emit(arith.SelectOp(predicate, a, b))

    zero, one = c(0), c(1)
    negative = emit(arith.CmpiOp(value, zero, "slt"))
    magnitude = choose(negative, binary(arith.SubiOp, zero, value), value)
    # Leading bit search on a positive, bounded integer, with defined zero behavior.
    leading = zero
    for step in (1 << n for n in reversed(range(width.bit_length() - 1))):
        trial = binary(arith.AddiOp, leading, c(step))
        nonzero = emit(arith.CmpiOp(binary(arith.ShRUIOp, magnitude, trial), zero, "ne"))
        leading = choose(nonzero, trial, leading)
    divisor_bits = denominator.bit_length() - 1
    leading = binary(arith.SubiOp, leading, c(divisor_bits))
    nonnegative = emit(arith.CmpiOp(leading, zero, "sge"))
    pos = choose(nonnegative, leading, zero)
    neg = choose(nonnegative, zero, binary(arith.SubiOp, zero, leading))
    divisor = c(denominator)
    below = emit(arith.CmpiOp(binary(arith.ShLIOp, magnitude, neg), binary(arith.ShLIOp, divisor, pos), "ult"))
    leading = choose(below, binary(arith.SubiOp, leading, one), leading)
    exponent_value = binary(arith.AddiOp, leading, c(exponent))
    drop = binary(arith.SubiOp, leading, c(fraction_bits))
    minimum_drop = 1 - bias - fraction_bits - exponent
    if minimum_drop + divisor_bits >= width - 7:
        # The input bound makes every magnitude strictly smaller than half a subnormal.
        raw = choose(negative, c(1 << (fraction_bits + exponent_bits)), zero)
        return emit(arith.BitcastOp(emit(arith.TruncIOp(raw, bits)), result_type))
    drop = choose(emit(arith.CmpiOp(drop, c(minimum_drop), "slt")), c(minimum_drop), drop)
    right = choose(emit(arith.CmpiOp(drop, zero, "sge")), drop, zero)
    left = choose(emit(arith.CmpiOp(drop, zero, "slt")), binary(arith.SubiOp, zero, drop), zero)
    numerator = binary(arith.ShLIOp, magnitude, left)
    divisor = binary(arith.ShLIOp, divisor, right)
    if denominator == 1:
        quotient = binary(arith.ShRUIOp, numerator, right)
        remainder = binary(arith.SubiOp, numerator, binary(arith.ShLIOp, quotient, right))
    else:
        # Base-2**32 division by a <=31-bit literal uses at most 63-bit
        # partial dividends. All divisions stay at native i64 width; no
        # compiler-rt i128 or wider division symbol is required on the host.
        from xdsl.dialects.builtin import i64

        small_divisor = emit(arith.ConstantOp(IntegerAttr(denominator, i64)))
        residue = emit(arith.ConstantOp(IntegerAttr(0, i64)))
        shift32 = emit(arith.ConstantOp(IntegerAttr(32, i64)))
        quotient_full = zero
        for offset in reversed(range(0, width, 32)):
            word = binary(arith.AndIOp, binary(arith.ShRUIOp, numerator, c(offset)), c((1 << 32) - 1))
            word = emit(arith.TruncIOp(word, i64))
            chunk = binary(arith.OrIOp, binary(arith.ShLIOp, residue, shift32), word)
            digit = binary(arith.DivUIOp, chunk, small_divisor)
            residue = binary(arith.RemUIOp, chunk, small_divisor)
            extended = emit(arith.ExtUIOp(digit, wide))
            quotient_full = binary(arith.OrIOp, quotient_full, binary(arith.ShLIOp, extended, c(offset)))
        quotient = binary(arith.ShRUIOp, quotient_full, right)
        remainder = binary(arith.SubiOp, numerator, binary(arith.MuliOp, quotient, divisor))
    twice = binary(arith.ShLIOp, remainder, one)
    above = emit(arith.CmpiOp(twice, divisor, "ugt"))
    tie = emit(arith.CmpiOp(twice, divisor, "eq"))
    odd = emit(arith.CmpiOp(binary(arith.AndIOp, quotient, one), zero, "ne"))
    increment = binary(arith.OrIOp, above, binary(arith.AndIOp, tie, odd))
    rounded = binary(arith.AddiOp, quotient, choose(increment, one, zero))
    carry = emit(arith.CmpiOp(rounded, c(1 << (fraction_bits + 1)), "uge"))
    rounded = choose(carry, binary(arith.ShRUIOp, rounded, one), rounded)
    exponent_value = binary(arith.AddiOp, exponent_value, choose(carry, one, zero))
    normal = emit(arith.CmpiOp(exponent_value, c(1 - bias), "sge"))
    subnormal_carry = emit(arith.CmpiOp(rounded, c(1 << fraction_bits), "uge"))
    encoded_exponent = choose(normal, binary(arith.AddiOp, exponent_value, c(bias)), choose(subnormal_carry, one, zero))
    fraction = binary(arith.AndIOp, rounded, c((1 << fraction_bits) - 1))
    raw = binary(arith.OrIOp, binary(arith.ShLIOp, encoded_exponent, c(fraction_bits)), fraction)
    overflow = emit(arith.CmpiOp(encoded_exponent, c((1 << exponent_bits) - 1), "uge"))
    raw = choose(overflow, c(((1 << exponent_bits) - 1) << fraction_bits), raw)
    raw = choose(emit(arith.CmpiOp(magnitude, zero, "eq")), zero, raw)
    raw = binary(arith.OrIOp, raw, choose(negative, c(1 << (fraction_bits + exponent_bits)), zero))
    return emit(arith.BitcastOp(emit(arith.TruncIOp(raw, bits)), result_type))


def finite_grid(values):
    """A complete dyadic payload determines a common exponent and exact bound."""
    import math

    values = list(values)
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError("nonfinite source data require an explicit IEEE component policy")
    fractions = [Fraction.from_float(float(value)) for value in values]
    exponents = [-value.denominator.bit_length() + 1 for value in fractions if value]
    exponent = min(exponents, default=0)
    # Remove powers of two in large integral payloads as well.
    if fractions and all(value.denominator == 1 for value in fractions):
        powers = [(abs(value.numerator) & -abs(value.numerator)).bit_length() - 1 for value in fractions if value]
        exponent = min(powers, default=0)
    bound = max((abs(int(value / Fraction(2) ** exponent)) for value in fractions), default=0)
    if bound.bit_length() > 512:
        raise ValueError("finite source exponent span exceeds the proved 512-bit fixed-point packing bound")
    return exponent, bound
