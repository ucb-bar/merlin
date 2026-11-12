"""Typed immutable tensor publication for one current scalar observation.

No memory operation is moved. A static tensor carrier and its lane coordinates
stay in their original block; only a private integer-producing DAG may change.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class _Coordinate:
    variable: object | None
    offset: int
    lower: int
    upper: int


def _constant_index(value):
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import IndexType, IntegerAttr

    operation = value.owner
    if (
        not isinstance(operation, arith.ConstantOp)
        or not isinstance(value.type, IndexType)
        or not isinstance(operation.value, IntegerAttr)
    ):
        raise ValueError("static typed index bound required")
    return operation.value.value.data, operation


def _loop_bounds(loop):
    from xdsl.dialects.builtin import IndexType, TensorType

    if (
        loop.name != "scf.for"
        or len(loop.operands) != 4
        or len(loop.results) != 1
        or not isinstance(loop.results[0].type, TensorType)
        or len(loop.regions) != 1
        or len(loop.regions[0].blocks) != 1
    ):
        raise ValueError("one immutable tensor SCF carrier required")
    block = loop.regions[0].block
    if (
        len(block.args) != 2
        or not isinstance(block.args[0].type, IndexType)
        or block.args[1].type != loop.results[0].type
        or loop.operands[3].type != loop.results[0].type
    ):
        raise ValueError("typed SCF tensor carrier does not agree")
    lower, lower_op = _constant_index(loop.operands[0])
    limit, limit_op = _constant_index(loop.operands[1])
    step, step_op = _constant_index(loop.operands[2])
    if lower < 0 or limit <= lower or step <= 0:
        raise ValueError("nonempty nonnegative static SCF interval required")
    upper = lower + (limit - lower - 1) // step * step
    return lower, upper, step, (loop, lower_op, limit_op, step_op)


def _coordinate(value):
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import IndexType
    from xdsl.ir import BlockArgument

    if not isinstance(value.type, IndexType):
        raise ValueError("tensor publication coordinate is not index typed")
    if isinstance(value.owner, arith.ConstantOp):
        number, operation = _constant_index(value)
        return _Coordinate(None, number, number, number), (operation,)
    if isinstance(value, BlockArgument):
        loop = value.owner.parent_op()
        if value.index != 0 or loop is None or loop.name != "scf.for":
            raise ValueError("tensor lane coordinate is not a proved SCF induction value")
        lower, upper, _, witnesses = _loop_bounds(loop)
        return _Coordinate(value, 0, lower, upper), witnesses
    operation = value.owner
    if operation.name != "arith.addi" or len(operation.operands) != 2 or len(operation.results) != 1:
        raise ValueError("tensor lane coordinate has unsupported arithmetic")
    left, left_ops = _coordinate(operation.operands[0])
    right, right_ops = _coordinate(operation.operands[1])
    if left.variable is not None and right.variable is not None:
        raise ValueError("tensor lane coordinate combines induction values")
    return _Coordinate(
        left.variable if left.variable is not None else right.variable,
        left.offset + right.offset,
        left.lower + right.lower,
        left.upper + right.upper,
    ), (*left_ops, *right_ops, operation)


def pure_tensor_control(operation):
    """Recognize only pure operations and bounded tensor-only SCF loops."""
    from xdsl.traits import Pure

    if operation.has_trait(Pure):
        return True
    # tensor.empty retains its original allocation/publication. It cannot read
    # the floating environment; unlike a general effectful producer, it is a
    # virtual static tensor construction, never removed by this rewrite.
    if operation.name == "tensor.empty" and not operation.operands and len(operation.results) == 1:
        from xdsl.dialects.builtin import TensorType

        return isinstance(operation.results[0].type, TensorType) and all(
            size > 0 for size in operation.results[0].type.get_shape()
        )
    if operation.name != "scf.for":
        return False
    try:
        _loop_bounds(operation)
    except ValueError:
        return False
    block = operation.regions[0].block
    terminator = block.last_op
    return (
        terminator is not None
        and terminator.name == "scf.yield"
        and len(terminator.operands) == 1
        and terminator.operands[0].type == operation.results[0].type
        and all(child is terminator or pure_tensor_control(child) for child in block.ops)
    )


def prove_closed_tensor_insert(anchor):
    """Authenticate a single-use immutable destination and distinct typed lanes.

    Returns operations whose current operands, attributes and uses must be part
    of the caller's witness. Unsupported loops/index expressions decline.
    """
    from xdsl.dialects.builtin import TensorType, i8
    from xdsl.ir import BlockArgument

    if (
        anchor.name != "tensor.insert"
        or len(anchor.results) != 1
        or len(anchor.operands) < 3
        or anchor.operands[0].type != i8
        or not isinstance(anchor.operands[1].type, TensorType)
        or anchor.results[0].type != anchor.operands[1].type
        or anchor.operands[1].type.element_type != i8
    ):
        raise ValueError("typed scalar i8 immutable tensor insertion required")
    block = anchor.parent
    if block is None:
        raise ValueError("tensor observation has no current owning block")
    shape = anchor.results[0].type.get_shape()
    if not shape or any(size <= 0 for size in shape):
        raise ValueError("static positive tensor observation shape required")
    chain, current = [], anchor
    while True:
        chain.append(current)
        destination = current.operands[1]
        previous = destination.owner
        if getattr(previous, "name", None) == "tensor.insert" and previous.parent is block:
            if destination is not previous.results[0]:
                raise ValueError("tensor publication has an ambiguous result")
            current = previous
        else:
            root = destination
            break
    chain.reverse()
    current = anchor
    while True:
        uses = tuple(current.results[0].uses)
        if len(uses) != 1:
            raise ValueError("tensor publication carrier has a live escape")
        use = uses[0]
        next_operation = use.operation
        if next_operation.name == "tensor.insert" and use.index == 1 and next_operation.parent is block:
            chain.append(next_operation)
            current = next_operation
            continue
        if (
            next_operation is not block.last_op
            or next_operation.name not in ("scf.yield", "func.return")
            or len(next_operation.operands) != 1
            or use.index != 0
        ):
            raise ValueError("tensor publication has no unique original return")
        terminator = next_operation
        break
    if len(tuple(root.uses)) != 1:
        raise ValueError("tensor publication root has a live escape")
    witnesses = [*chain, terminator]
    if isinstance(root, BlockArgument):
        loop = root.owner.parent_op()
        if root.index != 1 or loop is None or root.owner is not loop.regions[0].block:
            raise ValueError("tensor publication root is not the retained SCF carrier")
        witnesses.extend(_loop_bounds(loop)[3])
    elif getattr(root.owner, "name", None) == "scf.for":
        if not pure_tensor_control(root.owner):
            raise ValueError("tensor tail carrier has unknown effects")
        witnesses.extend(root.owner.walk())
    elif getattr(root.owner, "name", None) == "tensor.empty":
        witnesses.append(root.owner)
    else:
        raise ValueError("tensor publication root ownership is unsupported")
    coordinates = []
    for ordinal, operation in enumerate(chain):
        if (
            operation.parent is not block
            or len(operation.operands) != len(shape) + 2
            or len(operation.results) != 1
            or operation.results[0].type != anchor.results[0].type
            or operation.operands[0].type != i8
        ):
            raise ValueError("tensor publication chain changes type or rank")
        uses = tuple(operation.results[0].uses)
        expected = chain[ordinal + 1] if ordinal + 1 < len(chain) else terminator
        expected_index = 1 if ordinal + 1 < len(chain) else 0
        if len(uses) != 1 or uses[0].operation is not expected or uses[0].index != expected_index:
            raise ValueError("tensor publication carrier has a live escape")
        if len(tuple(operation.operands[0].uses)) != 1:
            raise ValueError("tensor integer lane has an additional live observation")
        point = []
        for index, size in zip(operation.operands[2:], shape, strict=True):
            coordinate, index_witnesses = _coordinate(index)
            if not 0 <= coordinate.lower <= coordinate.upper < size:
                raise ValueError("tensor publication coordinate is not proved in bounds")
            point.append(coordinate)
            witnesses.extend(index_witnesses)
        coordinates.append(point)
    for left_index, left in enumerate(coordinates):
        for right in coordinates[left_index + 1 :]:
            if not any(a.variable is b.variable and a.offset != b.offset for a, b in zip(left, right, strict=True)):
                raise ValueError("tensor publication lanes are not proved disjoint")
    return tuple(dict.fromkeys(witnesses))
