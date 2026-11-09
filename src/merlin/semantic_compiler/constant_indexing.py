"""Recover affine reads from declared constant indices, without sampling data.

The complete finite output domain is checked. Runtime indices and non-affine
reads remain unsupported; this does not grant a target gather instruction.
"""

from itertools import product
from math import prod

from merlin.frontends.linalg_patterns import _checked_static_linalg_shell

from .linalg_recognition import LinalgBridgeError, _linear_affine_expression, _type
from .model import IndexMap, SemanticNode, pointwise_identity_maps


def _flat(indices, shape):
    if len(indices) != len(shape) or any(not 0 <= i < n for i, n in zip(indices, shape)):
        raise LinalgBridgeError("constant index reads outside its declared tensor bounds")
    offset = 0
    for index, extent in zip(indices, shape):
        offset = offset * extent + index
    return offset


def _constant_indices(roots, nodes, constants, budget):
    """Evaluate only constant signed-i64 movement and modular arithmetic."""
    by_id = {node.id: node for node in nodes}
    pending, needed = list(roots), set()
    while pending:
        key = pending.pop()
        if key in needed:
            continue
        node = by_id[key]
        needed.add(key)
        budget -= prod(node.type.shape)
        if budget < 0:
            from .target_binding import NativeCompilationError

            raise NativeCompilationError("resource_limit", "constant indexing exceeds its element work budget")
        if node.type.dtype != "i64" or node.type.numerical_policy != "signed-i64":
            raise LinalgBridgeError("constant indexing requires signed-i64 index tensors")
        if node.op not in {"constant", "identity", "reshape", "gather", "concat", "i64_add", "i64_sub", "i64_mul"}:
            raise LinalgBridgeError("index tensor is not a supported declared compile-time constant")
        pending.extend(node.inputs)
    values = {}
    for node in nodes:
        if node.id not in needed:
            continue
        operands = [values[key] for key in node.inputs]
        if node.op == "constant":
            value = constants.get(node.id)
            if value is None or value.type != node.type:
                raise LinalgBridgeError("index constant has no matching declared bytes")
            result = value.elements
        elif node.op in {"identity", "reshape"}:
            result = operands[0]
        elif node.op == "gather":
            source = by_id[node.inputs[0]]
            result = tuple(
                operands[0][_flat(node.index_maps[0].apply(point), source.type.shape)]
                for point in product(*(range(n) for n in node.type.shape))
            )
        elif node.op == "concat":
            axis = dict(node.attrs)["axis"]
            result = []
            for point in product(*(range(n) for n in node.type.shape)):
                at = list(point)
                for key, operand in zip(node.inputs, operands):
                    shape = by_id[key].type.shape
                    if at[axis] < shape[axis]:
                        result.append(operand[_flat(at, shape)])
                        break
                    at[axis] -= shape[axis]
            result = tuple(result)
        else:
            operation = node.op.removeprefix("i64_")
            if dict(node.attrs) != {"numeric_semantics": f"signed-i64-{operation}-wrap-v1"}:
                raise LinalgBridgeError("constant index arithmetic has a different rounding policy")
            result = []
            for a, b in zip(*operands, strict=True):
                value = {"add": lambda: a + b, "sub": lambda: a - b, "mul": lambda: a * b}[operation]()
                result.append((value + (1 << 63)) % (1 << 64) - (1 << 63))
            result = tuple(result)
        if len(result) != prod(node.type.shape):
            raise LinalgBridgeError("constant index element count differs from its declared shape")
        values[node.id] = result
    return {key: values[key] for key in roots}, by_id


def translate_constant_index_read(op, env, nodes, constants, *, node_id, max_work=1 << 20):
    """Recognize index casts followed by one immutable captured tensor read.

    Return None for other scalar bodies. A recognized read must satisfy every
    check; it cannot fall through to a matcher with weaker index requirements.
    """
    if len(op.regions) != 1 or len(op.regions[0].blocks) != 1:
        return None
    body = tuple(op.regions[0].block.ops)
    if not any(item.name == "tensor.extract" for item in body):
        return None
    if tuple(item.name for item in body) != ("arith.index_cast",) * len(op.inputs) + (
        "tensor.extract",
        "linalg.yield",
    ):
        return None
    try:
        shell = _checked_static_linalg_shell(op, singleton_projection_inputs=True)
        op.verify()
    except Exception as error:  # noqa: BLE001 - fail closed at the parsed source boundary
        raise LinalgBridgeError(f"constant indexed read has an invalid Linalg shell: {error}") from error
    rank = len(shell.shape)
    if prod(shell.shape) * max(1, len(op.inputs)) > max_work:
        from .target_binding import NativeCompilationError

        raise NativeCompilationError("resource_limit", "constant indexing exceeds its finite-domain work budget")
    if (
        not op.inputs
        or len(body) != len(op.inputs) + 2
        or tuple(item.name for item in body)
        != ("arith.index_cast",) * len(op.inputs) + ("tensor.extract", "linalg.yield")
        or any(item.properties or any(not k.startswith("prov.") for k in item.attributes) for item in body)
        or any(item.regions or item.successors for item in body)
        or any(str(arg.type) != "i64" for arg in shell.args[:-1])
        or env.get(op.outputs[0], "undefined") is not None
    ):
        raise LinalgBridgeError("constant indexed read needs only i64 casts, a tensor read, and an empty output")
    casts, read, yielded = body[:-2], body[-2], body[-1]
    if (
        any(
            tuple(cast.operands) != (arg,) or len(cast.results) != 1 or str(cast.results[0].type) != "index"
            for cast, arg in zip(casts, shell.args[:-1])
        )
        or len(read.results) != 1
        or tuple(read.operands[1:]) != tuple(cast.results[0] for cast in casts)
        or tuple(yielded.operands) != (read.results[0],)
        or read.operands[0] not in env
        or env[read.operands[0]] is None
    ):
        raise LinalgBridgeError("constant indexed read has an unknown table, index, or yielded value")
    table_type, result_type = _type(read.operands[0].type), _type(op.results[0].type)
    if len(table_type.shape) != len(casts) or (table_type.dtype, table_type.numerical_policy) != (
        result_type.dtype,
        result_type.numerical_policy,
    ):
        raise LinalgBridgeError("constant indexed read has incompatible table dimensions or element type")
    roots = tuple(env.get(value) for value in op.inputs)
    if any(key is None for key in roots):
        raise LinalgBridgeError("constant indexed read uses an uninitialized index tensor")
    values, by_id = _constant_indices(roots, nodes, constants, max_work)
    maps = []
    for mapping in shell.input_maps:
        rows = tuple(_linear_affine_expression(expr, rank) for expr in mapping.results)
        maps.append(IndexMap(rank, tuple(row for row, _ in rows), tuple(offset for _, offset in rows)))

    def coordinate(point):
        return tuple(
            values[key][_flat(mapping.apply(point), by_id[key].type.shape)] for key, mapping in zip(roots, maps)
        )

    origin = coordinate((0,) * rank)
    differences = []
    for axis, extent in enumerate(shell.shape):
        unit = tuple(int(dim == axis) for dim in range(rank))
        differences.append(tuple(b - a for a, b in zip(origin, coordinate(unit))) if extent > 1 else (0,) * len(origin))
    source_map = IndexMap(rank, tuple(tuple(delta[dim] for delta in differences) for dim in range(len(origin))), origin)
    for point in product(*(range(n) for n in shell.shape)):
        actual = coordinate(point)
        _flat(actual, table_type.shape)
        if actual != source_map.apply(point):
            raise LinalgBridgeError("declared constant indices do not form an affine tensor read")
    return SemanticNode(
        node_id,
        "gather",
        (env[read.operands[0]],),
        result_type,
        index_maps=(source_map, pointwise_identity_maps(rank, 1)[-1]),
    )
