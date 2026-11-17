"""Reconstruct source index enclosures from compiler-owned parent assertions.

LLVM a47bddcc InferIntRangeInterfaceImpls.cpp and int-range-interface.mlir
define signed/unsigned extension and cast semantics. The native range solver
is not a Python API and does not export parent-call assertion witnesses. This
bounded reader keeps fixed-width word intervals and requires an actual checked
index -> integer -> index round trip before transferring an integer enclosure.
No target index width, assertion message, source tag or example value is used.
"""

from xdsl.dialects import arith, builtin, cf, tensor

from merlin.xdsl_dialects.standard_arith import IndexCastUIOp


def _constant(value, depth=0):
    if depth >= 16:
        return None
    operation = value.owner
    if isinstance(operation, arith.ConstantOp) and isinstance(operation.value, builtin.IntegerAttr):
        number = operation.value.value.data
        if isinstance(value.type, builtin.IntegerType):
            if not 0 < value.type.bitwidth <= 4096:
                return None
            number %= 1 << value.type.bitwidth
        return number
    if isinstance(operation, (arith.ExtSIOp, arith.ExtUIOp)):
        source = operation.operands[0]
        number = _constant(source, depth + 1)
        if (number is None or not isinstance(source.type, builtin.IntegerType)
                or not isinstance(value.type, builtin.IntegerType)
                or not 0 < source.type.bitwidth < value.type.bitwidth <= 4096):
            return None
        if isinstance(operation, arith.ExtSIOp) and number >= 1 << (source.type.bitwidth - 1):
            number -= 1 << source.type.bitwidth
        return number % (1 << value.type.bitwidth)
    return None


def _index_key(value):
    operation = value.owner
    if isinstance(operation, tensor.DimOp):
        source, axis = operation.operands
        number = _constant(axis)
        if (isinstance(source.type, builtin.TensorType) and type(number) is int
                and 0 <= number < len(source.type.get_shape())):
            return (source, number)
    return value


def _integer_key(value):
    operation = value.owner
    if isinstance(operation, IndexCastUIOp) and isinstance(operation.operands[0].type, builtin.IndexType):
        return (_index_key(operation.operands[0]), value.type)
    return value


def _dominating_prefix(point):
    """Only sequential predecessors, including predecessors of enclosing ops.

    No assertion inside a sibling region, a successor block or a later operation
    dominates the point by this rule. CFG/loop-carried facts need another proof.
    """
    groups, count = [], 0
    while point is not None and point.parent is not None:
        if len(groups) >= 64:
            return None
        block = point.parent
        prefix = []
        for operation in block.ops:
            if operation is point:
                break
            count += 1
            if count > 4096:
                return None
            prefix.append(operation)
        groups.append(prefix)
        point = block.parent_op()
    return [operation for group in reversed(groups) for operation in group]


def _comparisons(prefix):
    comparisons, pending, count = [], [], 0
    for operation in prefix:
        if isinstance(operation, cf.AssertOp):
            pending.append(operation.operands[0])
    while pending:
        count += 1
        if count > 4096:
            return None
        value = pending.pop()
        operation = value.owner
        if isinstance(operation, arith.CmpiOp):
            comparisons.append(operation)
        elif isinstance(operation, arith.AndIOp) and value.type == builtin.i1:
            pending.extend(operation.operands)
    return comparisons


def _allowed(predicate, word, width):
    modulus, half = 1 << width, 1 << (width - 1)
    if predicate == "eq":
        return [(word, word)]
    if predicate == "ne":
        return [(0, word - 1), (word + 1, modulus - 1)]
    signed = predicate.startswith("s")
    low, high = (-half, half - 1) if signed else (0, modulus - 1)
    bound = word - modulus if signed and word >= half else word
    relation = predicate[1:]
    if relation == "lt":
        high = bound - 1
    elif relation == "le":
        high = bound
    elif relation == "gt":
        low = bound + 1
    elif relation == "ge":
        low = bound
    if not signed:
        return [(low, high)]
    return [(max(low, 0), high), (low + modulus, min(high, -1) + modulus)]


def unsigned_index_range(value, point):
    """Return a conservative inclusive enclosure, or None when unproved.

    Both the round-trip equality and every used bound must be executable before
    point. Duplicate tensor.dim and unsigned-cast queries share a key only when
    their actual tensor SSA root, constant axis and integer width agree.
    """
    if not isinstance(value.type, builtin.IndexType):
        return None
    prefix = _dominating_prefix(point)
    if prefix is None or (comparisons := _comparisons(prefix)) is None:
        return None
    candidates = []
    for compare in comparisons:
        if compare.predicate.value.data != 0:
            continue
        for original, restored in (compare.operands, tuple(reversed(compare.operands))):
            if not isinstance(original.type, builtin.IndexType) or _index_key(original) != _index_key(value):
                continue
            cast = restored.owner
            if not isinstance(cast, IndexCastUIOp) or not isinstance(restored.type, builtin.IndexType):
                continue
            bridge = cast.operands[0]
            forward = bridge.owner
            if (not isinstance(forward, IndexCastUIOp) or not isinstance(bridge.type, builtin.IntegerType)
                    or not 0 < bridge.type.bitwidth <= 4096
                    or not isinstance(forward.operands[0].type, builtin.IndexType)
                    or _index_key(forward.operands[0]) != _index_key(value)):
                continue
            candidates.append(bridge)
    result = None
    reverse = {"eq": "eq", "ne": "ne", "slt": "sgt", "sle": "sge", "sgt": "slt", "sge": "sle",
               "ult": "ugt", "ule": "uge", "ugt": "ult", "uge": "ule"}
    for bridge in candidates:
        width = bridge.type.bitwidth
        ranges = [(0, (1 << width) - 1)]
        for compare in comparisons:
            predicate_number = compare.predicate.value.data
            if not 0 <= predicate_number < len(arith.CMPI_COMPARISON_OPERATIONS):
                continue
            predicate = arith.CMPI_COMPARISON_OPERATIONS[predicate_number]
            left, right = compare.operands
            if left.type != bridge.type or right.type != bridge.type:
                continue
            if _integer_key(left) == _integer_key(bridge):
                bound = _constant(right)
            elif _integer_key(right) == _integer_key(bridge):
                bound, predicate = _constant(left), reverse[predicate]
            else:
                continue
            if bound is None:
                continue
            allowed = _allowed(predicate, bound, width)
            ranges = [(max(a, c), min(b, d)) for a, b in ranges for c, d in allowed if max(a, c) <= min(b, d)]
            # Intersections of signed/unsigned comparisons remain at most two
            # word intervals. Bound fragmentation and contradictory domains fail.
            if not ranges or len(ranges) > 2:
                return None
        enclosure = min(a for a, _ in ranges), max(b for _, b in ranges)
        result = enclosure if result is None else (max(result[0], enclosure[0]), min(result[1], enclosure[1]))
    return result if result is not None and result[0] <= result[1] else None
