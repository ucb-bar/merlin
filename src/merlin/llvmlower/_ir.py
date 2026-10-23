"""Shared xDSL builders for the exact component rewrites.

Each helper spells one construction the rewrites built by hand, op for op and
in the same order, so the emitted IR does not depend on which helper built it.
"""

from xdsl.dialects import arith, tensor
from xdsl.dialects.builtin import AffineMapAttr, DenseIntOrFPElementsAttr, IntegerAttr, TensorType, UnitAttr, i64
from xdsl.dialects.linalg import ops as L
from xdsl.ir import Block, BlockArgument, Region
from xdsl.ir.affine import AffineDimExpr, AffineMap

PARALLEL = L.IteratorTypeAttr(L.IteratorType.PARALLEL)
REDUCTION = L.IteratorTypeAttr(L.IteratorType.REDUCTION)


def identity(rank):
    return AffineMapAttr(AffineMap.identity(rank))


def dims(rank, axes):
    return AffineMapAttr(AffineMap(rank, 0, tuple(AffineDimExpr(d) for d in axes)))


# A pointwise product as a K=1 batched contraction over [N, 1, 1] operands.
BATCHED_MAPS = [dims(4, axes) for axes in [(0, 1, 3), (0, 3, 2), (0, 1, 2)]]
BATCHED_ITERATORS = [PARALLEL] * 3 + [REDUCTION]


def generic_input(value):
    """The linalg.generic input that a block argument reads, else None."""
    if isinstance(value, BlockArgument):
        parent = value.owner.parent_op()
        if getattr(parent, "name", None) == "linalg.generic" and value.index < len(parent.inputs):
            return parent.inputs[value.index]
    return None


def input_literal(value):
    """The scalar literal behind value, reading a generic block argument as its input."""
    from merlin.xdsl_dialects.lowering.group_numerics import _constant_of

    source = generic_input(value)
    return _constant_of(value if source is None else source)


def emitter(block):
    """Append an op to block and return its first result."""

    def emit(op):
        block.add_op(op)
        return op.results[0]

    return emit


def constants(emit, typ):
    """Integer literal builder of one element type."""
    return lambda value: emit(arith.ConstantOp(IntegerAttr(value, typ)))


def generic(inputs, output, block, maps, iterators, *, host=False):
    """One linalg.generic writing output; host marks exact host arithmetic."""
    op = L.GenericOp(
        inputs=inputs,
        outputs=[output],
        body=Region(block),
        indexing_maps=maps,
        iterator_types=iterators,
        result_types=[output.type],
    )
    if host:
        op.attributes["merlin.exact_host_arithmetic"] = UnitAttr()
    return op


def point(inputs, typ, fn, operations, *, maps=None, iterators=None, host=False):
    """Append (empty, generic) computing fn(args, emit) per element; identity parallel maps by default."""
    empty = tensor.EmptyOp([], typ)
    block = Block(arg_types=[v.type.element_type for v in inputs] + [typ.element_type])
    block.add_op(L.YieldOp(fn(block.args[:-1], emitter(block))))
    rank = len(typ.get_shape())
    op = generic(
        inputs,
        empty.tensor,
        block,
        maps or [identity(rank)] * (len(inputs) + 1),
        iterators or [PARALLEL] * rank,
        host=host,
    )
    operations.extend([empty, op])
    return op.results[0]


def copy(value, operations):
    """Append an identity-map elementwise copy of value."""
    return point([value], value.type, lambda args, emit: args[0], operations)


def mac(inputs, seed, maps, iterators):
    """seed += a * b, sign-extending narrower operands to the accumulator type."""
    acc = seed.type.element_type
    block = Block(arg_types=[v.type.element_type for v in inputs] + [acc])
    emit = emitter(block)
    a, b = (value if value.type == acc else emit(arith.ExtSIOp(value, acc)) for value in block.args[:2])
    block.add_op(L.YieldOp(emit(arith.AddiOp(emit(arith.MuliOp(a, b)), block.args[2]))))
    return generic(inputs, seed, block, maps, iterators)


def splat(value, element, shape, operations):
    """Append (constant, splat) of one integer literal."""
    constant = arith.ConstantOp(IntegerAttr(value, element))
    filled = tensor.SplatOp(constant.result, [], TensorType(element, shape))
    operations.extend([constant, filled])
    return filled.result


def reshape(value, shape, operations):
    """Append (shape literal, tensor.reshape)."""
    sizes = arith.ConstantOp(DenseIntOrFPElementsAttr.from_list(TensorType(i64, [len(shape)]), list(shape)))
    changed = tensor.ReshapeOp(value, sizes.result, TensorType(value.type.element_type, shape))
    operations.extend([sizes, changed])
    return changed.result
