"""Prospective exact geometry proof for bounded all-parallel affine copies."""

from xdsl.ir.affine import AffineBinaryOpExpr, AffineBinaryOpKind, AffineConstantExpr, AffineDimExpr

PROTOCOL = "merlin.connected_integer_affine_copy.v2"


def _interval(expression, extents):
    if isinstance(expression, AffineDimExpr):
        if not 0 <= expression.position < len(extents):
            raise ValueError("affine copy references an unbounded loop")
        return 0, extents[expression.position] - 1
    if isinstance(expression, AffineConstantExpr) and expression.value >= 0:
        return expression.value, expression.value
    if isinstance(expression, AffineBinaryOpExpr):
        left, right = _interval(expression.lhs, extents), _interval(expression.rhs, extents)
        if expression.kind == AffineBinaryOpKind.Add:
            return left[0] + right[0], left[1] + right[1]
        if expression.kind == AffineBinaryOpKind.Mul and (
            isinstance(expression.lhs, AffineConstantExpr) or isinstance(expression.rhs, AffineConstantExpr)
        ):
            return left[0] * right[0], left[1] * right[1]
    raise ValueError("affine copy requires nonnegative constant linear access expressions")


def geometry(op, *, shape, bits):
    """Every output is written once; every affine input read is in bounds.

    A pure yield of the input element copies its interval without changing any
    integer arithmetic. No reduction, output read, symbol, dynamic extent,
    division, remainder or arbitrary scalar effect is admitted by this protocol.
    """
    if len(op.operands) != 2 or len(op.results) != 1 or len(op.regions) != 1 or len(op.regions[0].blocks) != 1:
        raise ValueError("affine copy requires one input and one complete output")
    block = op.regions[0].block
    body = tuple(block.ops)
    if (len(block.args) != 2 or len(body) != 1 or body[0].name != "linalg.yield"
            or tuple(body[0].operands) != (block.args[0],)
            or op.results[0].type != op.operands[1].type
            or bits(op.operands[0].type) != bits(op.results[0].type)):
        raise ValueError("affine copy scalar body is not a complete same-type pure copy")
    maps = [a.data for a in op.properties["indexing_maps"].data]
    kinds = [a.data.value for a in op.properties["iterator_types"].data]
    extents = shape(op.results[0])
    if (len(maps) != 2 or len(kinds) != len(extents) or any(kind != "parallel" for kind in kinds)
            or any(mapping.num_symbols or mapping.num_dims != len(extents) for mapping in maps)
            or len(maps[1].results) != len(extents)
            or any(not isinstance(axis, AffineDimExpr) or axis.position != i
                   for i, axis in enumerate(maps[1].results))):
        raise ValueError("affine copy requires complete identity output geometry and parallel loops")
    source_shape = shape(op.operands[0])
    if len(maps[0].results) != len(source_shape):
        raise ValueError("affine copy input map differs from its typed rank")
    bounds = [_interval(axis, extents) for axis in maps[0].results]
    if any(low < 0 or high >= extent for (low, high), extent in zip(bounds, source_shape, strict=True)):
        raise ValueError("affine copy input access escapes complete source storage")
    return kinds, dict(enumerate(extents)), maps, [list(bound) for bound in bounds]
