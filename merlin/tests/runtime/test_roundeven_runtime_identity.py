"""All-word arithmetic identity; native C/ABI/execution are separate checks.

Requires the declared verify extra. This theorem does not parse or certify C.
"""

import pytest
import z3


@pytest.mark.parametrize("width,fraction,bias,sort", [(32, 23, 127, z3.Float32()), (64, 52, 1023, z3.Float64())])
def test_rounding_word_identity_for_every_non_nan_input(width, fraction, bias, sort):
    bits = z3.BitVec("bits", width)
    sign = bits & (1 << (width - 1))
    magnitude = bits ^ sign
    exponent = z3.LShR(magnitude, fraction)
    shift = (bias + fraction) - exponent
    one = z3.BitVecVal(1, width)
    rounded = sign | ((magnitude + (one << (shift - 1)) - 1 + (z3.LShR(magnitude, shift) & 1))
                      & ~((one << shift) - 1))
    # The branches establish a valid shift before the rounding expression is
    # selected, preserve integral values, and handle zero and the half-to-one bin.
    result = z3.If(z3.UGE(exponent, bias + fraction), bits,
        z3.If(z3.ULT(exponent, bias - 1), sign,
            z3.If(exponent == bias - 1,
                sign | z3.If((magnitude & ((1 << fraction) - 1)) != 0,
                             z3.BitVecVal(bias << fraction, width), z3.BitVecVal(0, width)), rounded)))
    value = z3.fpBVToFP(bits, sort)
    expected = z3.fpToIEEEBV(z3.fpRoundToIntegral(z3.RNE(), value))
    solver = z3.Solver()
    solver.set(timeout=30000)
    solver.add(z3.Not(z3.fpIsNaN(value)), result != expected)
    assert solver.check() == z3.unsat
