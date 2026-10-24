"""Prove a finite exact source scalar program to be a zero-threshold unit.

This theorem belongs only to the declared quantized rational model. It must
never replace an IEEE program, where intermediate NaNs or rounding differ.
"""

import math
from fractions import Fraction


def normalize(module):
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import FloatAttr
    from xdsl.dialects.linalg import ops as L
    from xdsl.ir import Block
    from xdsl.rewriter import Rewriter

    from merlin.llvmlower._ir import generic
    from merlin.xdsl_dialects.lowering.contraction_coverage import is_compare_select_relu

    changed = 0
    for op in list(module.walk()):
        if op.name != "linalg.generic" or len(op.inputs) != 1 or len(op.outputs) != 1 or is_compare_select_relu(op):
            continue
        if any(it.data.value != "parallel" for it in op.get_iterator_types()):
            continue
        if op.inputs[0].type != op.results[0].type or str(op.results[0].type.element_type) not in {
            "f16",
            "bf16",
            "f32",
        }:
            continue
        x, finite = object(), object()

        def evaluate(nonnegative):
            env = {op.body.block.args[0]: x}
            for child in op.body.block.ops:
                name = getattr(getattr(child, "op_name", None), "data", child.name)
                flags = child.properties.get("fastmath")
                if flags is not None and flags.data:
                    raise ValueError("relaxed source arithmetic")
                if name == "linalg.yield":
                    return env[child.operands[0]]
                if child.regions or len(child.results) != 1:
                    raise ValueError("unproved scalar effect or region")
                if name == "arith.constant":
                    value = child.value.value.data
                    if not isinstance(value, (int, float)) or not math.isfinite(value):
                        raise ValueError("nonfinite source literal")
                    result = Fraction(value)
                else:
                    args = [env[v] for v in child.operands]
                    if name in {"arith.extf", "arith.truncf"}:
                        result = args[0]
                    elif name in {"arith.addf", "arith.subf", "arith.mulf"}:
                        a, b = args
                        if isinstance(a, Fraction) and isinstance(b, Fraction):
                            result = {
                                "arith.addf": lambda: a + b,
                                "arith.subf": lambda: a - b,
                                "arith.mulf": lambda: a * b,
                            }[name]()
                        elif name == "arith.mulf" and (a == 0 or b == 0):
                            result = Fraction(0)
                        elif (name == "arith.mulf" and b == 1) or (name in {"arith.addf", "arith.subf"} and b == 0):
                            result = a
                        elif (name == "arith.mulf" and a == 1) or (name == "arith.addf" and a == 0):
                            result = b
                        else:
                            result = finite
                    elif name == "math.expm1":
                        # exp(t)-1 is finite as an exact real for finite t.
                        # No approximate value is computed or emitted.
                        result = finite
                    elif name == "arith.cmpf" and args[0] is x and args[1] == 0:
                        if child.predicate.value.data == 3:
                            result = nonnegative
                        elif child.predicate.value.data == 4:
                            result = not nonnegative
                        else:
                            raise ValueError("unproved comparison cell")
                    elif name == "arith.select" and isinstance(args[0], bool):
                        result = args[1] if args[0] else args[2]
                    else:
                        raise ValueError("unproved source primitive")
                env[child.results[0]] = result
            raise ValueError("missing scalar result")

        try:
            if evaluate(True) is not x or evaluate(False) != 0:
                continue
        except (ValueError, KeyError, AttributeError):
            continue
        element = op.results[0].type.element_type
        body = Block(arg_types=[element, element])
        zero = arith.ConstantOp(FloatAttr(0.0, element))
        negative = arith.CmpfOp(body.args[0], zero.result, "olt")
        value = arith.SelectOp(negative.result, zero.result, body.args[0])
        body.add_ops([zero, negative, value, L.YieldOp(value.result)])
        replacement = generic(list(op.inputs), op.outputs[0], body, op.indexing_maps, op.iterator_types)
        replacement.attributes.update(op.attributes)
        Rewriter.replace_op(op, replacement)
        changed += 1
    if changed:
        module.verify()
    return changed
