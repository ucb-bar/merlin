"""Source-owned scalar storage leaves and dominating word-range proofs.

Only actual public singleton tensors or native integer arguments supply leaves.
Storage values are runtime data; compilation reads only their types and source
assertions. Truncation and arithmetic never inherit an injective cast proof.
"""

from xdsl.dialects import builtin, tensor

from .shape_expressions import _intersect


def bindings(arguments):
    result, sources = {}, {}
    for ordinal, value in enumerate(arguments):
        type_ = value.type
        if isinstance(type_, builtin.TensorType):
            if not isinstance(type_.encoding, builtin.NoneAttr) or any(size != 1 for size in type_.get_shape()):
                continue
            indices, element = [0] * len(type_.get_shape()), type_.element_type
        else:
            indices, element = None, type_
        if not isinstance(element, builtin.IntegerType) or element.bitwidth not in (8, 16, 32, 64):
            continue
        name = f"w{ordinal}"
        result[name] = {"argument": ordinal, "indices": indices, "dtype": str(element)}
        sources[value] = name
    return result, sources


def leaf(facts, value):
    if not facts.data_words:
        return None
    source = value
    if isinstance(value.owner, tensor.ExtractOp):
        source = value.owner.operands[0]
        name = facts.word_sources.get(source)
        if name is None:
            return None
        from .source_index_ranges import _constant

        if [_constant(index) for index in value.owner.operands[1:]] != facts.word_bindings[name]["indices"]:
            return None
    else:
        name = facts.word_sources.get(source)
        if name is None or facts.word_bindings[name]["indices"] is not None:
            return None
    facts.word_reads = True
    return facts.program.intern("word", type_=value.type.bitwidth, word=name)


def range_at(facts, name, point):
    if name in facts.word_limits:
        return facts.word_limits[name]
    binding = facts.word_bindings.get(name)
    if binding is None:
        return None
    width = int(binding["dtype"][1:])
    full = [(0, (1 << width) - 1)]
    key = name, point
    if key in facts.word_range_queries:
        return full
    facts.word_range_queries.add(key)
    try:
        from .shape_expression_ir import word
        from .source_index_ranges import _comparisons, _dominating_prefix

        prefix = _dominating_prefix(point)
        if prefix is None:
            return None
        intervals = full
        for compare in _comparisons(prefix) or []:
            # A word restriction cannot come from an unknown native index sign.
            if not isinstance(compare.operands[0].type, builtin.IntegerType):
                continue
            condition = word(facts, compare.result)
            if condition is None:
                continue
            restrictions = facts.program._assume(condition, True, {})
            for expression, allowed in restrictions.items():
                row = facts.program.nodes[expression]
                while row["op"] in ("extui", "extsi"):
                    source = row["args"][0]
                    source_width, target_width = facts.program.type(source), facts.program.type(expression)
                    modulus, half = 1 << source_width, 1 << (source_width - 1)
                    if row["op"] == "extui":
                        allowed = _intersect(allowed, [(0, modulus - 1)])
                    else:
                        offset = (1 << target_width) - modulus
                        allowed = _intersect(allowed, [(0, half - 1)]) + [
                            (a - offset, b - offset)
                            for a, b in _intersect(allowed, [(offset + half, (1 << target_width) - 1)])
                        ]
                    expression, row = source, facts.program.nodes[source]
                if row["op"] == "word" and row["word"] == name:
                    intervals = _intersect(intervals, allowed)
        return intervals or None
    finally:
        facts.word_range_queries.remove(key)
