"""Canonicalize constant scalar squares before freestanding libm lowering.

Retained model captures can contain math.powf(x, 2) in normalization bodies.
Freestanding LLVM compilation retains the libm call; express the square as the
single floating multiplication its source operation computes. Other exponents,
dynamic exponents and shaped constants are deliberately outside this rewrite.
"""

from xdsl.dialects import arith, math
from xdsl.dialects.builtin import Float32Type, Float64Type, FloatAttr


def canonicalize_scalar_squares(module) -> int:
    count = 0
    for op in list(module.walk()):
        if not isinstance(op, math.PowFOp):
            continue
        if not isinstance(op.result.type, (Float32Type, Float64Type)):
            continue
        exponent = op.rhs.owner
        if not isinstance(exponent, arith.ConstantOp):
            continue
        value = exponent.value
        if not isinstance(value, FloatAttr) or value.value.data != 2.0:
            continue
        square = arith.MulfOp(op.lhs, op.lhs, flags=op.fastmath)
        square.attributes.update(op.attributes)
        block = op.parent
        assert block is not None
        block.insert_op_before(square, op)
        op.result.replace_all_uses_with(square.result)
        block.erase_op(op)
        count += 1
    return count
