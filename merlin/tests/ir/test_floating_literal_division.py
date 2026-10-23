# ruff: noqa: E501 -- MLIR fixtures retain one operation per line
"""Literal division retains an exact integer numerator and a rational readout."""

import pytest
from mlir_fixtures import forward, generic, limbs

from merlin.llvmlower.floating_components import rewrite_floating_components


# The literal reciprocal is carried as a normalized signed-byte significand
# (scaled by 2**6 so it fills the int8 operand), and the single exact readout
# owns the binary exponent: the common exponent is therefore 6 lower than the
# unnormalized one. The rational denominator (3 for 1/3) is unchanged.
@pytest.mark.parametrize("divisor,exponent,denominator", [(2.0, -9, 1), (-2.0, -9, 1), (3.0, -8, 3), (0.5, -7, 1)])
def test_literal_division_has_a_complete_numerator_producer(divisor, exponent, denominator):
    typ = "tensor<4xf16>"
    literal = f"%c = arith.constant {divisor} : f16\n%s = tensor.splat %c : {typ}"
    body = generic([("%x", typ), ("%s", typ)], typ, "%v = arith.divf %a, %b : f16")
    module = forward(f"%x:{typ}", typ, literal + "\n" + body)
    bounds, proofs = rewrite_floating_components(module, [(-2, 7)])
    assert len(proofs) == 1 and proofs[0]["source_data_arithmetic"]
    assert proofs[0]["common_exponent"] == exponent
    assert not any(op.name == "arith.divf" for op in module.walk())
    limbs(module, [0], bounds)
