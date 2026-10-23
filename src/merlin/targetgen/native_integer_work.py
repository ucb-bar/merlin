"""A quantized contraction prelude cannot cover retained integer family work."""


def uncovered(module, proof):
    from xdsl.dialects.builtin import IntegerType

    from merlin.llvmlower.uniform_numeric_literals import compile_time_operand

    bindings = {b.get("source_operation_ordinal", b.get("root_ordinal")) for b in (proof or {}).get("bindings", ())}
    missing = []
    for ordinal, op in enumerate(module.walk()):
        if op.name not in {"linalg.generic", "linalg.reduce"} or ordinal in bindings or not op.results:
            continue
        element = getattr(op.results[0].type, "element_type", None)
        if not isinstance(element, IntegerType) or element.width.data < 8:
            continue
        if all(compile_time_operand(v) for v in op.operands):
            continue
        if any(child.name in {"arith.addi", "arith.subi", "arith.muli"} for child in op.walk()):
            missing.append(ordinal)
    return missing
