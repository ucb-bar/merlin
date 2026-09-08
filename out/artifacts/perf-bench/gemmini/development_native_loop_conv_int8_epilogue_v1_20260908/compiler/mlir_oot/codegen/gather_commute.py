"""Uniform scalar-map commute across a proven constant-padded readonly gather.

Source geometry and SSA uses establish a no-recomputation profitability proof.
Padding is transformed by the exact scalar program, not an assumed zero identity.
"""
from math import prod
from itertools import product

from xdsl.dialects.builtin import TensorType
from xdsl.ir import Operation
from xdsl.ir.affine import AffineDimExpr

from .gather_view import IntegerRanges, geometry, readonly_gather
from .host_linalg import attr_of, tensor_shape


def complete_source_coverage(host, gather, insertion, max_points=4096, max_steps=200000):
    """Prove all interior points AND some padding are originally read.

    Only constant-derived integer index expressions are evaluated at compile time.
    Never evaluate Q, floating tensors or external input values. A bounded work
    budget refuses large/unknown index programs rather than assuming surjectivity.
    This matters when Q is partial: computing Q on an originally unread point could
    introduce undefined behavior even if the new value is never subsequently used.
    """
    info = readonly_gather(gather, host._static)
    if info is None or prod(info[1]) > max_points:
        return False
    ranges, cache = IntegerRanges(host._static), {}
    remaining = max_steps

    def charge():
        nonlocal remaining
        remaining -= 1
        return remaining >= 0

    def mapped(amap, ivs):
        return tuple(ivs[e.position] if isinstance(e, AffineDimExpr) else e.value for e in amap.results)

    def scalar(operation, values, ivs):
        if not charge():
            return None
        if operation.name == "linalg.index":
            dimension = int(attr_of(operation, "dim").value.data)
            return ivs[dimension] if 0 <= dimension < len(ivs) else None
        intervals = [(value, value) if value is not None else None for value in values]
        result = ranges.scalar(operation, intervals, ())
        return result[0] if result is not None and result[0] == result[1] else None

    def integer_at(value, index):
        key = value, tuple(index)
        if key in cache:
            return cache[key]
        cache[key] = None
        if not charge() or not isinstance(value.owner, Operation):
            return None
        operation = value.owner
        shape = tensor_shape(value.type)
        if len(shape) != len(index) or any(not 0 <= i < n for i, n in zip(index, shape)):
            return None
        result = None
        if operation.name in {"tensor.expand_shape", "tensor.collapse_shape", "tensor.reshape"}:
            source_shape = tensor_shape(operation.operands[0].type)
            if prod(shape) != prod(source_shape):
                return None
            flat = 0
            for coordinate, bound in zip(index, shape):
                flat = flat*bound+coordinate
            source_index = []
            for bound in reversed(source_shape):
                source_index.append(flat % bound)
                flat //= bound
            result = integer_at(operation.operands[0], tuple(reversed(source_index)))
        elif operation.name == "linalg.transpose":
            perm = host._static(operation, "permutation")
            if sorted(perm) != list(range(len(shape))):
                return None
            result = integer_at(operation.operands[0], tuple(index[perm.index(d)] for d in range(len(perm))))
        elif (constant := uniform_constant(value)) is not None:
            result = scalar(constant, [], ())
        elif (layout := geometry(operation, host._static)) is not None:
            count, _, maps, permutation, _, body = layout
            ivs = tuple(index[permutation.index(d)] for d in range(len(shape)))
            env = {arg: integer_at(source, mapped(amap, ivs)) for arg, source, amap
                in zip(body.args, operation.operands[:count], maps[:count])}
            for inner in body.ops:
                if inner.name == "linalg.yield":
                    result = env.get(inner.operands[0])
                    break
                if inner.regions or len(inner.results) != 1:
                    return None
                env[inner.results[0]] = scalar(inner, [env.get(v) for v in inner.operands], ivs)
        cache[key] = result
        return result

    count, shape, maps, permutation, body = info
    offsets = host._static(insertion, "static_offsets")
    sizes = host._static(insertion, "static_sizes")
    interior, saw_padding = set(), False
    for index in product(*(range(size) for size in shape)):
        if not charge():
            return False
        ivs = tuple(index[permutation.index(d)] for d in range(len(shape)))
        env = {arg: integer_at(source, mapped(amap, ivs)) for arg, source, amap
            in zip(body.args, gather.operands[:count], maps[:count])}
        coordinates = None
        for operation in body.ops:
            if operation.name == "tensor.extract":
                coordinates = tuple(env.get(value) for value in operation.operands[1:])
                break
            env[operation.results[0]] = scalar(operation, [env.get(v) for v in operation.operands], ivs)
        if coordinates is None or any(value is None for value in coordinates):
            return False
        point = tuple(value-offset for value, offset in zip(coordinates, offsets))
        if all(0 <= value < size for value, size in zip(point, sizes)):
            interior.add(point)
        else:
            saw_padding = True
    return saw_padding and len(interior) == prod(sizes)


def scalar_constant(value):
    owner = value.owner
    return owner if isinstance(owner, Operation) and owner.name == "arith.constant" and not isinstance(value.type, TensorType) else None


def uniform_constant(value):
    owner = value.owner
    if isinstance(owner, Operation) and owner.name == "tensor.splat":
        return scalar_constant(owner.operands[0])
    return None


def plan_padded_gather(host, insertion):
    """Return a source-only legal rewrite plan, or None without changing emission."""
    source, padding = insertion.operands[:2]
    pad_constant = uniform_constant(padding)
    if pad_constant is None:
        return None
    sizes = host._static(insertion, "static_sizes")
    offsets = host._static(insertion, "static_offsets")
    strides = host._static(insertion, "static_strides")
    source_shape, padded_shape = tensor_shape(source.type), tensor_shape(padding.type)
    if (len(source_shape) != len(padded_shape) or tuple(sizes) != tuple(source_shape)
            or strides != [1] * len(source_shape) or len(offsets) != len(source_shape)
            or any(size <= 0 or offset < 0 or offset+size > bound
                   for size, offset, bound in zip(sizes, offsets, padded_shape))):
        return None
    inserted = insertion.results[0]
    uses = list(inserted.uses)
    if len(uses) != 1 or uses[0].operation.name != "tensor.extract" or uses[0].index != 0:
        return None
    gather = uses[0].operation.parent_op()
    if gather not in host._segment_ops or (info := readonly_gather(gather, host._static)) is None:
        return None
    if prod(source_shape) + 1 >= prod(info[1]):
        return None
    if not complete_source_coverage(host, gather, insertion):
        return None
    steps, pointwise = [], []
    constants = {pad_constant: None}
    value = gather.results[0]
    views = {"tensor.expand_shape", "tensor.collapse_shape", "tensor.reshape", "linalg.transpose"}
    while value not in host._segment_outputs:
        uses = list(value.uses)
        if len(uses) != 1:
            return None
        use, consumer = uses[0], uses[0].operation
        if consumer not in host._segment_ops or len(consumer.results) != 1:
            return None
        shape, next_shape = tensor_shape(value.type), tensor_shape(consumer.results[0].type)
        if prod(shape) != prod(next_shape) or any(size <= 0 for size in next_shape):
            return None
        if consumer.name in views:
            if use.index != 0 or value.type.get_element_type() != consumer.results[0].type.get_element_type():
                return None
            if consumer.name == "linalg.transpose":
                perm = host._static(consumer, "permutation")
                if sorted(perm) != list(range(len(shape))) or tuple(shape[d] for d in perm) != tuple(next_shape):
                    return None
            steps.append(("view", consumer, None))
        else:
            sink = host._pointwise_info(consumer)
            if sink is None or use.index >= sink[0] or len(shape) != len(next_shape):
                return None
            incoming = sink[2][use.index].results
            if (not all(isinstance(expr, AffineDimExpr) for expr in incoming)
                    or sorted(expr.position for expr in incoming) != list(range(len(shape)))):
                return None
            fixed = {}
            for index, operand in enumerate(consumer.operands[:sink[0]]):
                if index == use.index:
                    continue
                constant = uniform_constant(operand)
                if constant is None:
                    return None
                constants.setdefault(constant, None)
                fixed[index] = constant.results[0]
            body = sink[-1]
            internal = set(body.args)
            for inner in body.ops:
                for operand in inner.operands:
                    if operand in internal:
                        continue
                    constant = scalar_constant(operand)
                    if constant is None:
                        return None
                    constants.setdefault(constant, None)
                internal.update(inner.results)
            step = (consumer, sink, use.index, fixed)
            steps.append(("pointwise", consumer, step))
            pointwise.append(step)
        value = consumer.results[0]
    if not pointwise or any(use.operation in host._segment_ops for use in value.uses):
        return None
    return {"insertion": insertion, "source": source, "padding": pad_constant.results[0],
        "padded_shape": padded_shape, "offsets": offsets, "gather": gather, "gather_info": info,
        "steps": steps, "pointwise": pointwise, "constants": list(constants), "output": value}
