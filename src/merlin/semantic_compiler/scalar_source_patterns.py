"""Recognize closed source scalar DAGs before specialized tensor families.

These guards preserve authored operation order and data dependencies. The
scalar-region translator separately validates maps, types, effects, and exact
arithmetic policies; recognition does not select a target implementation.
"""

from .linalg_recognition import LinalgBridgeError, _type


def is_strict_source_negation(op) -> bool:
    """Check a unary sign flip without admitting a changed yielded value."""
    from merlin.frontends.linalg_patterns import InvalidLinalgPattern

    if (len(op.inputs) == 1 and _type(op.inputs[0].type).dtype in {"bf16", "f32", "f64"}
            and len(op.regions) == 1 and len(op.regions[0].blocks) == 1
            and tuple(item.name for item in op.regions[0].block.ops) == ("arith.negf", "linalg.yield")):
        scalar, yielded = tuple(op.regions[0].block.ops)
        if tuple(yielded.operands) != tuple(scalar.results):
            raise LinalgBridgeError("strict source negation must yield its result")
        if _type(op.inputs[0].type).dtype == "f32":
            from merlin.frontends.linalg_math_patterns import recognize_static_f32_neg_body
            try:
                recognize_static_f32_neg_body(op)
            except InvalidLinalgPattern as error:
                raise LinalgBridgeError(str(error)) from error
        return True
    return False


def is_direct_division_silu(op) -> bool:
    """Check the exact f32 opmath DAG between explicit source casts."""
    if (len(op.inputs) == 1 and _type(op.inputs[0].type).dtype == "f32"
            and _type(op.results[0].type).dtype == "f32"
            and len(op.regions) == 1 and len(op.regions[0].blocks) == 1
            and tuple(item.name for item in op.regions[0].block.ops) == (
                "arith.constant", "arith.negf", "math.exp", "arith.addf", "arith.divf", "linalg.yield")):
        from xdsl.dialects.builtin import FloatAttr

        literal, neg, exp, add, div, yielded = tuple(op.regions[0].block.ops)
        scalar_source = op.regions[0].block.args[0]
        # A literal numerator belongs to the existing sigmoid recognizer.
        # Only a source-valued numerator identifies the SiLU family here.
        if tuple(div.operands)[:1] != (scalar_source,):
            return False
        value = literal.properties.get("value")
        if (not isinstance(value, FloatAttr) or str(value.type) != "f32" or value.value.data != 1.0
                or tuple(neg.operands) != (scalar_source,)
                or tuple(exp.operands) != tuple(neg.results)
                or tuple(add.operands) != (literal.results[0], exp.results[0])
                or tuple(div.operands) != (scalar_source, add.results[0])
                or tuple(yielded.operands) != tuple(div.results)):
            raise LinalgBridgeError("source SiLU must preserve its exact direct-division scalar DAG")
        return True
    return False
