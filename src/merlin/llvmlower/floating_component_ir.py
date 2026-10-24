"""Exact host preparation for finite IEEE component producers."""

from functools import partial

from xdsl.dialects import arith
from xdsl.dialects.builtin import IntegerType

from merlin.llvmlower import _ir
from merlin.llvmlower.fixed_point_ir import FORMAT

point = partial(_ir.point, host=True)


def pack_ieee_integer(value, exponent, integer_type, emit):
    """Decode finite IEEE bits to a proved integral common-exponent coefficient."""
    i64 = integer_type
    fraction_bits, exponent_bits, bias = FORMAT[str(value.type)]
    storage = IntegerType(1 + fraction_bits + exponent_bits)
    raw = emit(arith.BitcastOp(value, storage))
    raw = emit(arith.ExtUIOp(raw, i64))

    constant = _ir.constants(emit, i64)

    def binary(cls, a, b):
        return emit(cls(a, b))

    field = binary(
        arith.AndIOp, binary(arith.ShRUIOp, raw, constant(fraction_bits)), constant((1 << exponent_bits) - 1)
    )
    normal = emit(arith.CmpiOp(field, constant(0), "ne"))
    significand = binary(arith.AndIOp, raw, constant((1 << fraction_bits) - 1))
    implicit = emit(arith.SelectOp(normal, constant(1 << fraction_bits), constant(0)))
    significand = binary(arith.OrIOp, significand, implicit)
    effective = emit(arith.SelectOp(normal, field, constant(1)))
    shift = binary(arith.SubiOp, effective, constant(bias + fraction_bits + exponent))
    nonnegative = emit(arith.CmpiOp(shift, constant(0), "sge"))
    left = emit(arith.SelectOp(nonnegative, shift, constant(0)))
    right = emit(arith.SelectOp(nonnegative, constant(0), binary(arith.SubiOp, constant(0), shift)))
    # Zero has no grid constraint. Its decoded exponent can be far outside
    # the chosen coefficient width, where even shifting zero is poison.
    is_zero = emit(arith.CmpiOp(significand, constant(0), "eq"))
    left = emit(arith.SelectOp(is_zero, constant(0), left))
    right = emit(arith.SelectOp(is_zero, constant(0), right))
    magnitude = binary(arith.ShRUIOp, binary(arith.ShLIOp, significand, left), right)
    negative = emit(
        arith.CmpiOp(binary(arith.AndIOp, raw, constant(1 << (fraction_bits + exponent_bits))), constant(0), "ne")
    )
    return emit(arith.SelectOp(negative, binary(arith.SubiOp, constant(0), magnitude), magnitude))
