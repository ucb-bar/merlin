"""Conservative source-SSA range proof for readonly scalar gather deferral.

This is compile-time interval analysis, not tensor execution or a value-dependent
specialization. Unknown inputs, integer overflow, nonprojected maps and exhausted
analysis budgets refuse the optimization while leaving normal lowering intact.
"""
from math import prod

from xdsl.dialects.builtin import IndexType, IntegerType, TensorType
from xdsl.ir import Operation
from xdsl.ir.affine import AffineConstantExpr, AffineDimExpr

from .host_linalg import attr_of, tensor_shape


def geometry(op, static):
    if op.name != "linalg.generic" or len(op.results) != 1:
        return None
    counts = static(op, "operandSegmentSizes")
    if len(counts) != 2 or counts[1] != 1:
        return None
    shape = tensor_shape(op.results[0].type)
    rank = len(shape)
    maps = [item.data for item in attr_of(op, "indexing_maps")]
    if (any(size <= 0 for size in shape) or len(maps) != len(op.operands)
            or any(m.num_dims != rank or m.num_symbols for m in maps)
            or [getattr(getattr(item, "data", None), "value", None)
                for item in attr_of(op, "iterator_types")] != ["parallel"] * rank):
        return None
    output = maps[-1].results
    if len(output) != rank or not all(isinstance(e, AffineDimExpr) for e in output):
        return None
    permutation = [e.position for e in output]
    if sorted(permutation) != list(range(rank)):
        return None
    bounds = [shape[permutation.index(dim)] for dim in range(rank)]
    for value, amap in zip(op.operands, maps):
        sizes = tensor_shape(value.type)
        if len(amap.results) != len(sizes):
            return None
        for expr, size in zip(amap.results, sizes):
            if isinstance(expr, AffineDimExpr):
                if not 0 <= expr.position < rank or bounds[expr.position] > size:
                    return None
            elif not isinstance(expr, AffineConstantExpr) or not 0 <= expr.value < size:
                return None
    body = op.regions[0].blocks[0]
    if len(body.args) != counts[0] + 1 or list(body.args[-1].uses):
        return None
    return counts[0], shape, maps, permutation, bounds, body


class IntegerRanges:
    def __init__(self, static, budget=256):
        self.static, self.remaining, self.cache = static, budget, {}

    @staticmethod
    def typed(result, ty):
        if result is None:
            return None
        width = 64 if isinstance(ty, IndexType) else ty.width.data if isinstance(ty, IntegerType) else 0
        return result if 1 <= width <= 64 and -(1 << (width-1)) <= result[0] <= result[1] < (1 << (width-1)) else None

    def scalar(self, op, inputs, bounds):
        name = op.name
        if name == "arith.constant":
            value = getattr(getattr(attr_of(op, "value"), "value", None), "data", None)
            result = (value, value) if isinstance(value, int) else None
        elif name == "linalg.index":
            dim = int(attr_of(op, "dim").value.data)
            result = (0, bounds[dim]-1) if 0 <= dim < len(bounds) else None
        elif name in {"arith.index_cast", "arith.extsi", "arith.trunci"}:
            result = inputs[0]
        elif len(inputs) == 2 and all(item is not None for item in inputs):
            a, b = inputs
            if name == "arith.addi":
                result = (a[0]+b[0], a[1]+b[1])
            elif name == "arith.subi":
                result = (a[0]-b[1], a[1]-b[0])
            elif name == "arith.muli" and (a[0] == a[1] or b[0] == b[1]):
                values = [x*y for x in a for y in b]
                result = (min(values), max(values))
            else:
                result = None
        else:
            result = None
        return self.typed(result, op.results[0].type)

    def tensor(self, value):
        if value in self.cache:
            return self.cache[value]
        self.cache[value] = None
        self.remaining -= 1
        op = value.owner
        if self.remaining < 0 or not isinstance(op, Operation) or not isinstance(value.type, TensorType):
            return None
        result = None
        if op.name in {"tensor.expand_shape", "tensor.collapse_shape", "tensor.reshape", "linalg.transpose"}:
            if (prod(tensor_shape(op.operands[0].type)) == prod(tensor_shape(value.type))
                    and op.operands[0].type.get_element_type() == value.type.get_element_type()):
                result = self.tensor(op.operands[0])
        elif (info := geometry(op, self.static)) is not None:
            n_in, _, _, _, bounds, body = info
            env = {arg: self.tensor(source) for arg, source in zip(body.args, op.operands[:n_in])}
            for inner in body.ops:
                if inner.name == "linalg.yield":
                    result = env.get(inner.operands[0])
                    break
                if inner.regions or len(inner.results) != 1:
                    break
                env[inner.results[0]] = self.scalar(inner, [env.get(v) for v in inner.operands], bounds)
        self.cache[value] = result
        return result


def readonly_gather(op, static):
    info = geometry(op, static)
    if info is None:
        return None
    n_in, shape, maps, permutation, bounds, body = info
    operations = list(body.ops)
    if not any(inner.name == "tensor.extract" for inner in operations):
        return None
    ranges = IntegerRanges(static)
    env = {arg: ranges.tensor(value) for arg, value in zip(body.args, op.operands[:n_in])}
    extracted = None
    for inner in operations[:-1]:
        if inner.regions or len(inner.results) != 1:
            return None
        if inner.name == "tensor.extract":
            source = inner.operands[0]
            if (extracted is not None or not isinstance(source.type, TensorType)
                    or source in body.args or source.owner in operations
                    or source.type.get_element_type() != op.results[0].type.get_element_type()):
                return None
            limits = [env.get(v) for v in inner.operands[1:]]
            sizes = tensor_shape(source.type)
            if len(limits) != len(sizes) or any(r is None or r[0] < 0 or r[1] >= size for r, size in zip(limits, sizes)):
                return None
            extracted = inner.results[0]
        else:
            interval = ranges.scalar(inner, [env.get(v) for v in inner.operands], bounds)
            if interval is None:
                return None
            env[inner.results[0]] = interval
    if (extracted is None or not operations or operations[-1].name != "linalg.yield"
            or tuple(operations[-1].operands) != (extracted,)):
        return None
    return n_in, shape, maps, permutation, body
