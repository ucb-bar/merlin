"""Conservative scalar-ring and statically known view normalization."""

from __future__ import annotations

from xdsl.dialects import arith, builtin, memref
from xdsl.ir import SSAValue
from xdsl.rewriter import Rewriter

from .certificates import IntegerWitness, ShapeWitness


def constant_integer(value: SSAValue) -> int | None:
    owner = value.owner
    if isinstance(owner, arith.ConstantOp) and isinstance(owner.value, builtin.IntegerAttr):
        return owner.value.value.data
    return None


def signed_wrap(value: int, width: int) -> int:
    if width <= 0:
        raise ValueError("integer width must be positive")
    modulus = 1 << width
    value %= modulus
    return value - modulus if value >= modulus // 2 else value


def normalize_shapes(module: builtin.ModuleOp) -> tuple[ShapeWitness, ...]:
    """Eliminate identical memref casts and materialize static dimensions.

    No reinterpretation, guessed strides, dynamic-to-static assumptions or
    element enumeration. The identity cast also applies to contiguous views
    presented through an already identical memref descriptor.
    """
    witnesses = []
    for index, op in enumerate(tuple(module.walk())):
        if isinstance(op, memref.CastOp) and op.source.type == op.dest.type:
            shape = op.source.type.get_shape() if isinstance(op.source.type, builtin.MemRefType) else ()
            witnesses.append(ShapeWitness(index, "identity_memref_cast", tuple(shape)))
            Rewriter.replace_op(op, [], [op.source])
        elif isinstance(op, memref.DimOp) and isinstance(op.source.type, builtin.MemRefType):
            axis = constant_integer(op.index)
            shape = op.source.type.get_shape()
            if axis is not None and 0 <= axis < len(shape) and shape[axis] >= 0:
                replacement = arith.ConstantOp.from_int_and_width(shape[axis], builtin.IndexType())
                witnesses.append(ShapeWitness(index, "static_memref_dimension", tuple(shape), axis))
                Rewriter.replace_op(op, replacement)
    return tuple(witnesses)


def simplify_integers(module: builtin.ModuleOp) -> tuple[IntegerWitness, ...]:
    """Fold scalar wrapping integers and neutral operands, without poison flags.

    Index arithmetic, tensors, floating arithmetic and nsw/nuw operations are
    deliberately excluded. Constants are canonicalized modulo the declared
    bit width; reductions are never reassociated.
    """
    witnesses = []
    for index, op in enumerate(tuple(module.walk())):
        if not isinstance(op, arith.AddiOp | arith.SubiOp | arith.MuliOp):
            continue
        dtype = op.result.type
        if not isinstance(dtype, builtin.IntegerType) or dtype.signedness.data != builtin.Signedness.SIGNLESS:
            continue
        if op.overflow_flags != arith.IntegerOverflowAttr("none"):
            continue
        lhs, rhs = constant_integer(op.lhs), constant_integer(op.rhs)
        width = dtype.width.data
        if lhs is not None and rhs is not None:
            if isinstance(op, arith.AddiOp):
                value = lhs + rhs
            elif isinstance(op, arith.SubiOp):
                value = lhs - rhs
            else:
                value = lhs * rhs
            result = signed_wrap(value, width)
            replacement = arith.ConstantOp.from_int_and_width(result, dtype)
            witnesses.append(IntegerWitness(index, op.name, "constant_ring", width, lhs, rhs, result))
            Rewriter.replace_op(op, replacement)
            continue
        retained: int | None = None
        if isinstance(op, arith.AddiOp):
            retained = 1 if lhs == 0 else 0 if rhs == 0 else None
        elif isinstance(op, arith.SubiOp):
            retained = 0 if rhs == 0 else None
        elif isinstance(op, arith.MuliOp):
            retained = 1 if lhs == 1 else 0 if rhs == 1 else None
        if retained is not None:
            witnesses.append(IntegerWitness(index, op.name, "neutral_operand", width, lhs, rhs, None, retained))
            Rewriter.replace_op(op, [], [op.operands[retained]])
    return tuple(witnesses)
