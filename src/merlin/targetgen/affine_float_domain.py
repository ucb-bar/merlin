"""A full byte-domain proof for retained floating affine intermediates.

Fixed observers alone do not make capture admission payload independent. The
frontend leaves non-contraction affine operations in floating point, so prove
that every multiply/add is exact before consulting any evaluated output.
This conservative lattice bound is independent of the graded tensor values.
"""


def prove_affine_domain(coefficients, offset, *, precision=24):
    from fractions import Fraction

    values = [Fraction(value) for value in (*coefficients, offset)]
    nonzero = [value for value in values if value]
    if not nonzero:
        return dict(grid_exponent=0, integer_bound=0, precision=precision)
    if any(value.denominator & (value.denominator - 1) for value in nonzero):
        raise ValueError("declared affine domain has a non-dyadic intermediate")
    exponent = min(-(value.denominator.bit_length() - 1) for value in nonzero)
    grid = Fraction(2) ** exponent
    bound = sum(128 * abs(value / grid) for value in values[:-1]) + abs(values[-1] / grid)
    # The absolute bound also bounds every individual product and partial sum.
    # Requiring the common lattice to fit the significand proves exactness of
    # each intermediate, rather than just its final rounded output code.
    if bound.denominator != 1 or bound.numerator.bit_length() > precision:
        raise ValueError("declared affine domain requires intermediate floating rounding")
    if exponent < -149 or bound * grid > Fraction(2) ** 128 - Fraction(2) ** 104:
        raise ValueError("declared affine domain exceeds finite binary32 storage")
    return dict(grid_exponent=exponent, integer_bound=int(bound), precision=precision)
