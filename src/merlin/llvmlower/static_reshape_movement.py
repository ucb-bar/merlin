"""Expose statically proved rank changes as materialized element selectors."""


def _static_rank_change(user, source, ranks):
    from math import prod

    from xdsl.dialects.builtin import DenseIntOrFPElementsAttr

    if user.name not in {"tensor.collapse_shape", "tensor.expand_shape", "tensor.reshape"} or len(user.results) != 1:
        return False
    if user.operands[0] is not source or user.results[0].type.element_type != source.type.element_type:
        return False
    shape = user.results[0].type.get_shape()
    if len(shape) not in ranks or any(n <= 0 for n in shape) or prod(shape) != prod(source.type.get_shape()):
        return False
    if user.name != "tensor.reshape":
        return len(user.operands) == 1
    owner = user.operands[1].owner
    value = owner.properties.get("value") if getattr(owner, "name", None) == "arith.constant" else None
    return isinstance(value, DenseIntOrFPElementsAttr) and tuple(value.get_values()) == shape


def lift_scalar_producers(module, *, attribute):
    """Present a scalar component as one element, retaining its scalar ABI."""
    from xdsl.rewriter import Rewriter

    from merlin.llvmlower._ir import PARALLEL, generic, identity, reshape

    changed = 0
    for op in list(module.walk()):
        if op.name != "linalg.generic" or attribute not in op.attributes or len(op.results) != 1:
            continue
        if op.results[0].type.get_shape() or any(value.type.get_shape() for value in op.operands):
            continue
        if list(op.get_iterator_types()) or any(child.name == "linalg.index" for child in op.walk()):
            continue
        operations = []
        inputs = [reshape(value, [1], operations) for value in op.inputs]
        output = reshape(op.outputs[0], [1], operations)
        clone = op.clone()
        region = clone.detach_region(0)
        block = region.detach_block(0)
        producer = generic(inputs, output, block, [identity(1)] * (len(inputs) + 1), [PARALLEL])
        producer.attributes.update(op.attributes)
        operations.append(producer)
        result = reshape(producer.results[0], [], operations)
        Rewriter.replace_op(op, operations, [result])
        changed += 1
    if changed:
        module.verify()
    return changed


def flatten_internal_selectors(module, *, ranks):
    """Elide an internal selector's packing axes before its immediate reshapes.

    Only private parallel copies whose every user is a static rank change
    qualify. The source and every observable result keep their original ranks.
    """
    from math import prod

    from xdsl.dialects.builtin import AffineMapAttr, TensorType
    from xdsl.ir.affine import AffineDimExpr, AffineMap
    from xdsl.rewriter import Rewriter

    from merlin.llvmlower._ir import identity, point, reshape

    if not ranks or 1 not in ranks:
        return 0
    changed = 0
    for op in list(module.walk()):
        if op.name != "linalg.generic" or len(op.results) != 1 or len(op.inputs) != 1:
            continue
        shape = op.results[0].type.get_shape()
        if len(shape) in ranks or any(n <= 0 for n in shape):
            continue
        if any(it.data.value != "parallel" for it in op.get_iterator_types()):
            continue
        block = op.body.block
        body = list(block.ops)
        if len(body) != 1 or body[0].name != "linalg.yield" or tuple(body[0].operands) != (block.args[0],):
            continue
        maps = list(op.get_indexing_maps())
        if maps[-1].data != AffineMap.identity(len(shape)) or maps[0].data.num_symbols:
            continue
        if len(op.inputs[0].type.get_shape()) not in ranks:
            continue
        users = [use.operation for use in op.results[0].uses]
        if not users or any(not _static_rank_change(user, op.results[0], ranks) for user in users):
            continue
        # Compose the original selector with the inverse row-major flattening.
        coordinates = tuple((AffineDimExpr(0) // prod(shape[i + 1:])) % n for i, n in enumerate(shape))
        inverse = AffineMap(1, 0, coordinates)
        selected_map = maps[0].data.compose(inverse)
        operations = []
        flat = point(
            list(op.inputs), TensorType(op.results[0].type.element_type, [prod(shape)]),
            lambda args, emit: args[0], operations,
            maps=[AffineMapAttr(selected_map), identity(1)],
        )
        flat.owner.attributes.update(op.attributes)
        op.parent.insert_ops_before(operations, op)
        for user in dict.fromkeys(users):
            operations = []
            result = reshape(flat, user.results[0].type.get_shape(), operations)
            result.owner.attributes.update(user.attributes)
            Rewriter.replace_op(user, operations, [result])
        Rewriter.erase_op(op)
        changed += 1
    if changed:
        module.verify()
    return changed


def expose_static_reshape_movement(module):
    from math import prod

    from xdsl.dialects.builtin import AffineMapAttr, DenseIntOrFPElementsAttr, TensorType
    from xdsl.ir.affine import AffineConstantExpr, AffineDimExpr, AffineMap
    from xdsl.rewriter import Rewriter

    from merlin.llvmlower._ir import identity, point

    changed = 0
    for op in list(module.walk()):
        if op.name not in {"tensor.reshape", "tensor.collapse_shape", "tensor.expand_shape", "tensor.cast"}:
            continue
        if len(op.results) != 1 or not isinstance(op.results[0].type, TensorType):
            continue
        source, result = op.operands[0], op.results[0]
        if not isinstance(source.type, TensorType) or source.type.element_type != result.type.element_type:
            continue
        before, after = source.type.get_shape(), result.type.get_shape()
        if any(d < 0 for d in (*before, *after)) or not prod(before) or prod(before) != prod(after):
            continue
        if op.name == "tensor.reshape":
            shape = op.operands[1].owner
            attr = shape.properties.get("value") if getattr(shape, "name", None) == "arith.constant" else None
            if not isinstance(attr, DenseIntOrFPElementsAttr) or tuple(attr.get_values()) != after:
                continue
        elif len(op.operands) != 1:
            continue
        # Row-major logical flattening is independent of the caller's physical
        # stride. Bufferization carries the real source descriptor into this map.
        flat = AffineConstantExpr(0)
        for axis in range(len(after)):
            flat = flat + AffineDimExpr(axis) * prod(after[axis + 1 :])
        indices = tuple((flat // prod(before[axis + 1 :])) % extent for axis, extent in enumerate(before))
        operations = []
        selected = point(
            [source],
            result.type,
            lambda args, emit: args[0],
            operations,
            maps=[AffineMapAttr(AffineMap(len(after), 0, indices)), identity(len(after))],
        )
        selected.owner.attributes.update(op.attributes)
        Rewriter.replace_op(op, operations, [selected])
        changed += 1
    return changed
