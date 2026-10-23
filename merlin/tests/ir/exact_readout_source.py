"""Exact readout fixture built from model2mlir's emitter (m2m/ir/exact_quant.py).

The scalar RNE primitives and the elementwise generic are m2m's own. Only the composition below
repeats m2m's ``readout_plan``/``build_readout``: those import ``m2m.capture``, which needs torch,
and the Merlin test venv has none. The test oracle is the rational reference in the test file.
"""

from fractions import Fraction

from m2m.ir.decompositions import _broadcast_map, _elementwise
from m2m.ir.exact_quant import Scalar
from xdsl.dialects.builtin import Float32Type, IntegerType


def readout_plan(scales, beta, bias_type, output_scale):
    """Prove a finite integer width for every possible runtime operand."""
    from merlin.targetgen.exact_arithmetic import rational

    fractions = [Fraction(s) for s in scales]
    if not fractions or any(s == 0 for s in fractions):
        raise ValueError("exact readout requires nonzero finite coefficients")
    # Signed coefficients carry alpha exactly, without negating the i32
    # accumulator (which could be -2**31). Width bounds use magnitudes.
    base = min(-s.denominator.bit_length() + 1 for s in fractions)
    bias_unit = None
    beta_coeff = 0
    if bias_type is not None:
        if bias_type != Float32Type():
            raise ValueError("exact readout requires float32 bias")
        bias_unit = -149
        bf = rational(beta)
        base = min(base, bias_unit - bf.denominator.bit_length() + 1)
        beta_coeff = int(bf * Fraction(2) ** (bias_unit - base))
    coeffs = [int(s / Fraction(2) ** base) for s in fractions]
    # int32 input; a finite f32 bias is < 2**128 or <2**277 subnormal units.
    bound = max(abs(v) for v in coeffs) * (1 << 31) + abs(beta_coeff) * (1 << 277)
    ratio = Fraction(2) ** base / (Fraction(output_scale) if output_scale is not None else 1)
    width = max(
        64,
        (2 * bound * abs(ratio.numerator)).bit_length() + 2,
        (1024 * ratio.denominator).bit_length() + 2,
        bound.bit_length() + 2,
    )
    # Avoid unbounded compiler artifacts for pathological exponents. This is a
    # declaration-wide refusal, never a test of the calibration sample.
    if width > 4096:
        raise ValueError("exact readout requires more than 4096 integer bits")
    return base, coeffs, beta_coeff, bias_unit, ratio, width


def build_readout(
    acc,
    scales,
    *,
    bias=None,
    beta=1.0,
    output_scale=None,
    zero_point=0,
    quant_min=-128,
    quant_max=127,
    relu=False,
    unsigned=False,
):
    """Lower exact affine readout using integer arithmetic and one final RNE."""
    from xdsl.dialects.builtin import IndexType, TensorType
    from xdsl.dialects.linalg.ops import IndexOp
    from xdsl.ir.affine import AffineMap

    if beta == 0:
        bias = None  # addmm beta=0 ignores self, including NaN/inf.
    if output_scale is not None:
        bounds = (0, 255) if unsigned else (-128, 127)
        if not bounds[0] <= quant_min < quant_max <= bounds[1] or not bounds[0] <= zero_point <= bounds[1]:
            raise ValueError("invalid readout range or zero point for byte storage")
        if not Fraction(output_scale) > 0:
            raise ValueError("readout output scale must be finite and positive")
    base, coeffs, beta_coeff, bias_unit, ratio, width = readout_plan(
        scales, beta, None if bias is None else bias.type.element_type, output_scale
    )
    typ, i64 = IntegerType(width), IntegerType(64)
    shape = list(acc.type.get_shape())
    if len(scales) > 1 and (not shape or shape[-1] != len(scales)):
        raise ValueError("exact readout scales must match the last axis")
    out_type = Float32Type() if output_scale is None else IntegerType(8)

    def build(args, oe):
        b = Scalar()
        coeff = b.c(coeffs[0], typ)
        if len(set(coeffs)) > 1:
            index = b.emit(IndexOp(len(shape) - 1))
            for ordinal, value in enumerate(coeffs[1:], 1):
                match = b.cmp(index, b.c(ordinal, IndexType()), "eq")
                coeff = b.select(match, b.c(value, typ), coeff)
        value = b.mul(b.cast(args[0], typ), coeff)
        if bias is not None:
            bm, be, finite = b.decode(args[1])
            b.assert_(finite, "exact readout requires finite bias")
            shift = b.cast(b.sub(be, b.c(bias_unit, i64)), typ)
            fixed = b.shl(b.cast(bm, typ), shift)
            value = b.add(value, b.mul(fixed, b.c(beta_coeff, typ)))
        if output_scale is None:
            result = b.to_float(value, base, oe)
        else:
            lhs = b.mul(value, b.c(2 * ratio.numerator, typ))

            def compare(mid):
                threshold = b.mul(b.cast(b.add(b.mul(mid, b.c(2, i64)), b.c(1, i64)), typ), b.c(ratio.denominator, typ))
                return b.cmp(lhs, threshold, "sgt"), b.cmp(lhs, threshold, "eq")

            low = min(quant_max, max(quant_min, zero_point)) if relu else quant_min
            code = b.search(b.c(zero_point, i64), low, quant_max, compare)
            result = b.cast(code, oe)
        return b.ops, result

    maps = [AffineMap.identity(len(shape))]
    if bias is not None:
        mapping = _broadcast_map(bias.type.get_shape(), shape)
        if mapping is None:
            raise ValueError("exact readout bias does not broadcast to accumulator")
        maps.append(mapping)
    return _elementwise(
        [acc, *([bias] if bias is not None else [])], TensorType(out_type, shape), build, input_maps=maps
    )
