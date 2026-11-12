"""Read closed normalization islands and their exact source row projections.

Discovery is read-only. These source helpers grant no numerical, physical
storage, target capability, execution or profitability permission. Materializing
and independently checking a selected island remain separate responsibilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

from xdsl.dialects import arith, builtin, math, tensor
from xdsl.dialects.linalg import ops as L
from xdsl.ir import Operation, SSAValue
from xdsl.ir.affine import AffineBinaryOpExpr, AffineBinaryOpKind, AffineConstantExpr, AffineDimExpr, AffineMap
from xdsl.traits import get_effects


@dataclass(frozen=True)
class RowNormalizationGroup:
    source: SSAValue
    operations: tuple[Operation, ...]
    shape: tuple[int, ...]

    @property
    def rows(self):
        return prod(self.shape[:-1])

    @property
    def columns(self):
        return self.shape[-1]


def _shape(value):
    typ = value.type
    if not isinstance(typ, builtin.TensorType) or not isinstance(typ.encoding, builtin.NoneAttr):
        raise ValueError("row normalization requires unencoded tensor semantics")
    shape = typ.get_shape()
    if any(n <= 0 for n in shape) or typ.get_element_type() not in (builtin.f32, builtin.f64):
        raise ValueError("row normalization requires positive static floating tensor shapes")
    return shape


def _metadata(operation):
    if any(not key.startswith("prov.") for key in operation.attributes):
        raise ValueError("unknown normalization operation attributes are conservative barriers")


def _view(value):
    seen = 0
    while isinstance(value.owner, (tensor.CollapseShapeOp, tensor.ExpandShapeOp)):
        op = value.owner
        _metadata(op)
        if len(op.operands) != 1 or prod(_shape(op.operands[0])) != prod(_shape(value)):
            raise ValueError("row normalization cannot interpret dynamic/noncontiguous reshape effects")
        value = op.operands[0]
        seen += 1
        if seen > 64:
            raise ValueError("row normalization reshape chain exceeds its source budget")
    return value


def _body(operation, scalar_class, inputs):
    if len(operation.regions) != 1 or len(operation.regions[0].blocks) != 1:
        return False
    block = operation.regions[0].block
    ops = tuple(block.ops)
    if len(ops) != 2 or type(ops[0]) is not scalar_class or type(ops[1]) is not L.YieldOp:
        return False
    scalar = ops[0]
    if get_effects(scalar) != set() or scalar.regions or scalar.successors:
        return False
    if tuple(scalar.operands) != tuple(block.args[:inputs]) or tuple(ops[1].operands) != tuple(scalar.results):
        return False
    return len(block.args) == inputs + (1 if isinstance(operation, L.GenericOp) else 0)


def _generic(operation, scalar_class, inputs):
    return isinstance(operation, L.GenericOp) and len(operation.results) == 1 and _body(operation, scalar_class, inputs)


def _reduce(operation, scalar_class):
    return isinstance(operation, L.ReduceOp) and len(operation.operands) == 2 and _body(operation, scalar_class, 2)


def _identity_copy(operation, value):
    if not isinstance(operation, L.GenericOp) or len(operation.operands) != 2 or len(operation.results) != 1:
        return False
    if (
        operation.operands[0] is not value
        or operation.results[0].type != value.type
        or not isinstance(operation.operands[1].owner, tensor.EmptyOp)
        or len(operation.regions) != 1
        or len(operation.regions[0].blocks) != 1
    ):
        return False
    block = operation.regions[0].block
    ops = tuple(block.ops)
    return (
        len(block.args) == 2
        and len(ops) == 1
        and type(ops[0]) is L.YieldOp
        and tuple(ops[0].operands) == (block.args[0],)
    )


def _linear(expression, rank):
    if isinstance(expression, AffineConstantExpr):
        return (expression.value, *([0] * rank))
    if isinstance(expression, AffineDimExpr):
        return (0, *(int(i == expression.position) for i in range(rank)))
    if isinstance(expression, AffineBinaryOpExpr):
        left, right = _linear(expression.lhs, rank), _linear(expression.rhs, rank)
        if expression.kind == AffineBinaryOpKind.Add:
            return tuple(a + b for a, b in zip(left, right))
        if expression.kind == AffineBinaryOpKind.Mul:
            if not any(left[1:]):
                return tuple(left[0] * x for x in right)
            if not any(right[1:]):
                return tuple(right[0] * x for x in left)
    raise ValueError("row normalization requires checked affine row-fiber indexing")


def _fiber_map(mapping, input_shape, loop_shape, kind):
    rank = len(loop_shape)
    if mapping.num_dims != rank or mapping.num_symbols or len(mapping.results) != len(input_shape):
        raise ValueError("row normalization has incompatible indexing ranks")
    actual = [0] * (rank + 1)
    for axis, expression in enumerate(mapping.results):
        coefficients = _linear(expression, rank)
        low = coefficients[0] + sum(min(0, c * (n - 1)) for c, n in zip(coefficients[1:], loop_shape, strict=True))
        high = coefficients[0] + sum(max(0, c * (n - 1)) for c, n in zip(coefficients[1:], loop_shape, strict=True))
        if low < 0 or high >= input_shape[axis]:
            raise ValueError("normalization indexing leaves a source tensor dimension")
        stride = prod(input_shape[axis + 1 :])
        for i, coefficient in enumerate(coefficients):
            actual[i] += coefficient * stride
    expected = [0] + [prod(loop_shape[i + 1 :]) for i in range(rank)]
    if kind == "scalar":
        expected = [0] * (rank + 1)
    elif kind == "row":
        expected = [0] + [prod(loop_shape[i + 1 : -1]) for i in range(rank - 1)] + [0]
    for i, extent in enumerate(loop_shape, 1):
        if extent == 1:
            actual[i] = expected[i] = 0
    if actual != expected:
        raise ValueError("normalization indexing crosses or permutes source row fibers")


def local_type(value, group):
    shape = _shape(value)
    count = prod(shape)
    if count == group.rows * group.columns:
        shape = (group.columns,)
    elif count in (1, group.rows):
        shape = ()
    else:
        raise ValueError("normalization tensor cannot be restricted to one complete row")
    return builtin.TensorType(value.type.get_element_type(), shape)


def row_input_projection(mapping, shape, loop_shape):
    """Interpret a static axis projection/broadcast; never flatten unknown maps."""
    rank = len(loop_shape)
    if mapping.num_dims != rank or mapping.num_symbols or len(mapping.results) != len(shape):
        raise ValueError("row input projection has incompatible ranks")
    axes = []
    for extent, expression in zip(shape, mapping.results, strict=True):
        if isinstance(expression, AffineConstantExpr) and expression.value == 0 and extent == 1:
            axes.append(None)
        elif (
            isinstance(expression, AffineDimExpr)
            and 0 <= expression.position < rank
            and extent == loop_shape[expression.position]
        ):
            axes.append(expression.position if extent > 1 else None)
        else:
            raise ValueError("row input requires a bounded axis projection or singleton broadcast")
    nonunit = [axis for axis in axes if axis is not None]
    if nonunit != sorted(set(nonunit)) or rank - 1 in axes[:-1]:
        raise ValueError("row input permutes, repeats or mixes the source column domain")
    return tuple(axes)


def _check_operation(operation, group):
    _metadata(operation)
    if len(operation.results) != 1:
        raise ValueError("normalization island requires single-result source operations")
    if isinstance(operation, arith.ConstantOp) and not isinstance(operation.results[0].type, builtin.TensorType):
        return
    local_type(operation.results[0], group)
    if isinstance(operation, L.GenericOp):
        shape = _shape(operation.results[0])
        if shape[-1:] != (group.columns,) or prod(shape[:-1]) != group.rows:
            raise ValueError("normalization pointwise iteration is not over complete source rows")
        if operation.properties.keys() - {"indexing_maps", "iterator_types", "operandSegmentSizes"}:
            raise ValueError("unknown normalization generic lowering properties")
        if len(operation.iterator_types) != len(shape) or any(
            "parallel" not in str(x) for x in operation.iterator_types
        ):
            raise ValueError("normalization pointwise body has an unknown iteration algorithm")
        for value, attr in zip(operation.operands, operation.indexing_maps, strict=True):
            if value.owner not in set(group.operations):
                row_input_projection(attr.data, _shape(value), shape)
                continue
            count = prod(_shape(value))
            kind = "full" if count == group.rows * group.columns else "row" if count == group.rows else "scalar"
            _fiber_map(attr.data, _shape(value), shape, kind)
    elif isinstance(operation, L.ReduceOp):
        shape = _shape(operation.operands[0])
        if shape[-1:] != (group.columns,) or prod(shape[:-1]) != group.rows:
            raise ValueError("normalization reduction does not cover a complete row")
        if tuple(operation.dimensions.get_values()) != (len(shape) - 1,) or operation.properties.keys() != {
            "dimensions"
        }:
            raise ValueError("normalization reduction has a different domain or unknown properties")
        if prod(_shape(operation.operands[1])) != group.rows or prod(_shape(operation.results[0])) != group.rows:
            raise ValueError("normalization reduction initializer/result has incorrect row coverage")
    elif isinstance(operation, (tensor.CollapseShapeOp, tensor.ExpandShapeOp)):
        _view(operation.results[0])
    elif isinstance(operation, tensor.EmptyOp):
        if operation.operands:
            raise ValueError("dynamic empty tensors are normalization barriers")
    elif isinstance(operation, tensor.SplatOp):
        if len(operation.operands) != 1 or not isinstance(operation.operands[0].owner, arith.ConstantOp):
            raise ValueError("normalization initialization requires its original constant scalar splat")
    else:
        raise ValueError("unknown normalization tensor semantics are conservative barriers")


def match_row_normalization(final):
    """Recognize from current typed source dataflow; provenance never selects it."""
    if not _generic(final, arith.DivfOp, 2):
        return None
    try:
        exponential = _view(final.operands[0]).owner
        summation = _view(final.operands[1]).owner
        if not _generic(exponential, math.ExpOp, 1) or not _reduce(summation, arith.AddfOp):
            return None
        if summation.operands[0] is not exponential.results[0]:
            return None
        shift = _view(exponential.operands[0]).owner
        if not _generic(shift, arith.SubfOp, 2):
            return None
        maximum = _view(shift.operands[1]).owner
        if not _reduce(maximum, arith.MaximumfOp) or maximum.operands[0] is not shift.operands[0]:
            return None
        source = maximum.operands[0]
        shape = _shape(source)
        if final.results[0].type != source.type:
            return None
        if len(shape) < 2 or prod(shape[:-1]) <= 1 or shape[-1] <= 1:
            return None
        if prod(shape) * source.type.get_element_type().bitwidth // 8 >= 1 << 63:
            return None
        selected = set()

        def visit(value):
            if value is source:
                return
            if not isinstance(value.type, builtin.TensorType):
                if not isinstance(value.owner, arith.ConstantOp):
                    raise ValueError("unknown normalization scalar initialization")
                return
            op = value.owner
            if not isinstance(op, Operation) or op.parent is not final.parent:
                raise ValueError("normalization tensor has an external observation or producer")
            if op in selected:
                return
            if len(selected) >= 64:
                raise ValueError("normalization source island exceeds its structural budget")
            selected.add(op)
            for operand in op.operands:
                visit(operand)

        visit(final.results[0])

        def include_prefix(value):
            # Keep an existing same-shape pointwise producer inside the row
            # island rather than introducing a materialization boundary.
            op = value.owner
            if not isinstance(op, Operation) or op.parent is not final.parent or value.type != source.type:
                return
            if op in selected:
                return
            if not any(_generic(op, cls, 2) for cls in (arith.AddfOp, arith.SubfOp, arith.MulfOp, arith.DivfOp)):
                return
            if len(op.operands) != 3 or not isinstance(op.operands[-1].owner, tensor.EmptyOp):
                return
            for v, mapping in zip(op.operands[:-1], tuple(op.indexing_maps)[:-1], strict=True):
                row_input_projection(mapping.data, _shape(v), shape)
            if len(selected) + 2 > 64:
                raise ValueError("normalization pointwise prefix exceeds its structural budget")
            selected.update((op, op.operands[-1].owner))
            for operand in op.operands[:-1]:
                include_prefix(operand)

        include_prefix(source)
        published = final
        for _ in range(8):
            uses = tuple(published.results[0].uses)
            if (
                len(uses) != 1
                or uses[0].operation.parent is not final.parent
                or not _identity_copy(uses[0].operation, published.results[0])
            ):
                break
            published = uses[0].operation
            selected.update((published, published.operands[1].owner))
        ordered = tuple(op for op in final.parent.ops if op in selected)
        if ordered[-1] is not published:
            return None
        active = False
        for op in final.parent.ops:
            active |= op is ordered[0]
            if active and isinstance(op, arith.ConstantOp) and not isinstance(op.results[0].type, builtin.TensorType):
                selected.add(op)
            if (
                active
                and isinstance(op, tensor.SplatOp)
                and prod(_shape(op.results[0])) == 1
                and isinstance(op.operands[0].owner, arith.ConstantOp)
                and all(use.operation in selected for use in op.results[0].uses)
            ):
                selected.add(op)
            if op is published:
                break
        ordered = tuple(op for op in final.parent.ops if op in selected)
        if len(ordered) > 64:
            return None
        group = RowNormalizationGroup(source, ordered, shape)
        for op in ordered:
            _check_operation(op, group)
            if op is not published and any(use.operation not in selected for value in op.results for use in value.uses):
                return None
        first = False
        for op in final.parent.ops:
            first |= op is ordered[0]
            if op is published:
                break
            if first and op not in selected:
                return None
        row_input_specs(group)
        return group
    except (ValueError, IndexError):
        return None


def row_input_values(group):
    return tuple(value for value, _ in row_input_specs(group))


def row_input_specs(group):
    members = set(group.operations)
    values = {}
    for op in group.operations:
        for i, value in enumerate(op.operands):
            if isinstance(value.type, builtin.TensorType) and value.owner not in members:
                if isinstance(op, L.GenericOp):
                    mapping = op.indexing_maps.data[i].data
                elif isinstance(op, L.ReduceOp) and i == 0 and value.type == group.source.type:
                    mapping = AffineMap.identity(len(group.shape))
                else:
                    raise ValueError("row input frontier has an unsupported source use")
                projection = row_input_projection(mapping, _shape(value), group.shape)
                if value in values and values[value] != projection:
                    raise ValueError("one row input has different source projections")
                values[value] = projection
    return tuple(values.items())


def ordered_contraction(operation):
    """Recognize a typed ordered multiply/add boundary for the cost policy.

    This classifies unchanged consumers/producers; it neither lowers a
    contraction nor grants a target primitive or reassociation capability.
    """
    if not isinstance(operation, L.GenericOp) or len(operation.regions) != 1 or len(operation.regions[0].blocks) != 1:
        return False
    block = operation.regions[0].block
    ops = tuple(block.ops)
    if len(block.args) != 3 or len(ops) != 3 or len(operation.operands) != 3 or len(operation.results) != 1:
        return False
    product, addition, yielded = ops
    if type(product) is not arith.MulfOp or type(addition) is not arith.AddfOp or type(yielded) is not L.YieldOp:
        return False
    if (
        tuple(product.operands) != tuple(block.args[:2])
        or tuple(addition.operands) not in ((product.result, block.args[2]), (block.args[2], product.result))
        or tuple(yielded.operands) != (addition.result,)
        or any(get_effects(o) != set() for o in ops[:2])
    ):
        return False
    iterators = tuple(operation.iterator_types)
    return (
        len(iterators) >= 3
        and iterators[-1] == L.IteratorTypeAttr(L.IteratorType.REDUCTION)
        and all(i == L.IteratorTypeAttr(L.IteratorType.PARALLEL) for i in iterators[:-1])
    )


# Published record spelling remains readable through the exact legacy alias.
RowNormalizationGroup.__module__ = "merlin.compiler.transforms.row_normalization"
