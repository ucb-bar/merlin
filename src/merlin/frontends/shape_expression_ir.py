"""Lift actual standard scalar operations into typed size programs."""

from xdsl.dialects import arith, builtin, tensor

from .shape_expressions import BINARY, CASTS, MAX_SIZE, PREDICATES, signed


def word(facts, value, depth=0):
    facts._visit(depth)
    if value in facts.expression_words:
        return facts.expression_words[value]
    if isinstance(value.type, builtin.IndexType):
        dimension = facts.index(value, depth + 1)
        return None if dimension is None else facts.program.dimension(dimension)
    if not isinstance(value.type, builtin.IntegerType) or not 1 <= value.type.bitwidth <= 4096:
        return None
    width = value.type.bitwidth
    from .shape_input_words import leaf

    supplied = leaf(facts, value)
    if supplied is not None:
        facts.expression_words[value] = supplied
        return supplied
    if value in facts.integers:
        return facts.program.intern(
            "index_castui",
            type_=width,
            args=[facts.program.dimension(facts.integers[value])],
        )
    operation = value.owner
    if isinstance(operation, arith.ConstantOp) and isinstance(operation.value, builtin.IntegerAttr):
        result = facts.program.intern("const", type_=width, value=operation.value.value.data % (1 << width))
    else:
        name = getattr(operation, "name", "").removeprefix("arith.")
        if name not in BINARY | CASTS | {"cmpi", "select"}:
            return None
        flags = operation.properties.get("overflowFlags")
        if flags is not None and getattr(flags, "data", None) != frozenset():
            return None
        if set(operation.properties) - {"overflowFlags", "predicate"}:
            return None
        arguments = [word(facts, operand, depth + 1) for operand in operation.operands]
        if any(argument is None for argument in arguments):
            return None
        predicate = None
        if name == "cmpi":
            number = operation.predicate.value.data
            if not 0 <= number < len(PREDICATES):
                return None
            predicate = PREDICATES[number]
            if facts.program.type(arguments[0]) == "index" and predicate.startswith("s"):
                return None
        if name == "index_cast" and isinstance(operation.operands[0].type, builtin.IndexType):
            return None
        result = facts.program.intern(name, type_=width, args=arguments, predicate=predicate)
    facts.expression_words[value] = result
    return result


def index(facts, value, depth=0):
    operation = value.owner
    name = getattr(operation, "name", "").removeprefix("arith.")
    if name == "select":
        condition = word(facts, operation.operands[0], depth + 1)
        alternatives = [facts.index(item, depth + 1) for item in operation.operands[1:]]
        if condition is None or any(item is None for item in alternatives):
            return None
        return facts.program.intern(
            "select",
            type_="index",
            args=[condition, *[facts.program.dimension(item) for item in alternatives]],
        )
    if name not in ("index_cast", "index_castui"):
        return None
    source = operation.operands[0]
    expression = word(facts, source, depth + 1)
    if expression is None:
        return None
    intervals = facts.program.bounds(expression, lambda axis: input_range(facts, axis, operation))
    if not intervals:
        return None
    if name == "index_cast":
        intervals = [(signed(a, source.type.bitwidth), signed(b, source.type.bitwidth)) for a, b in intervals]
    if any(not 0 <= low <= high <= MAX_SIZE for low, high in intervals):
        return None
    return facts.program.intern(name, type_="index", args=[expression])


def axis_range(facts, axis, point):
    if axis in facts.axis_limits:
        return facts.axis_limits[axis]
    from merlin.xdsl_dialects.standard_arith import IndexCastUIOp

    from .source_index_ranges import (
        _comparisons,
        _constant,
        _dominating_prefix,
        unsigned_index_range,
    )

    prefix = _dominating_prefix(point)
    if prefix is None or axis not in facts.axes:
        return None

    def source_axis(value):
        operation = value.owner
        if not isinstance(operation, tensor.DimOp):
            return None
        for name, binding in facts.axes.items():
            if (
                binding["argument"] < len(facts.arguments)
                and operation.source is facts.arguments[binding["argument"]]
                and _constant(operation.index) == binding["axis"]
            ):
                return name
        return None

    ranges, edges = dict(facts.axis_limits), []
    for operation in prefix:
        if isinstance(operation, tensor.DimOp) and (name := source_axis(operation.result)) is not None:
            interval = unsigned_index_range(operation.result, point)
            if interval is not None:
                previous = ranges.get(name, interval)
                ranges[name] = (
                    max(previous[0], interval[0]),
                    min(previous[1], interval[1]),
                )
    for compare in _comparisons(prefix) or []:
        if compare.predicate.value.data != 0:
            continue
        left, right = compare.operands
        if isinstance(left.type, builtin.IndexType):
            names = source_axis(left), source_axis(right)
        elif (
            isinstance(left.type, builtin.IntegerType)
            and left.type == right.type
            and isinstance(left.owner, IndexCastUIOp)
            and isinstance(right.owner, IndexCastUIOp)
        ):
            originals = left.owner.operands[0], right.owner.operands[0]
            names = tuple(source_axis(value) for value in originals)
            # Word equality implies size equality only when both actual source
            # round trips establish injectivity at this exact program point.
            if any(
                name is None
                or name not in ranges
                or not 0 <= ranges[name][0] <= ranges[name][1] < 1 << left.type.bitwidth
                for name in names
            ):
                continue
        else:
            continue
        if all(name is not None for name in names):
            edges.append(names)
    pending, component = [axis], set()
    while pending:
        name = pending.pop()
        if name in component:
            continue
        component.add(name)
        for left, right in edges:
            if name == left:
                pending.append(right)
            if name == right:
                pending.append(left)
    intervals = [ranges[name] for name in component if name in ranges]
    if not intervals:
        return None
    low, high = max(low for low, _ in intervals), min(high for _, high in intervals)
    return (low, high) if 0 <= low <= high <= MAX_SIZE else None


def range_at(facts, value, point):
    dimension = facts.size(value, point)
    if dimension is None:
        return None
    if type(dimension) is int:
        return dimension, dimension
    if dimension not in facts.program.nodes:
        return input_range(facts, dimension, point)
    intervals = facts.program.bounds(dimension, lambda axis: input_range(facts, axis, point))
    if not intervals or any(not 0 <= low <= high <= MAX_SIZE for low, high in intervals):
        return None
    return min(a for a, _ in intervals), max(b for _, b in intervals)


def input_range(facts, name, point):
    if name in facts.word_bindings:
        from .shape_input_words import range_at

        return range_at(facts, name, point)
    return axis_range(facts, name, point)
