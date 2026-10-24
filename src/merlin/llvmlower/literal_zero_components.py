"""Remove integer producers proved zero from source literals, never input samples."""

import math


def fold_literal_zero_components(module, limb_attribute):
    """Keep IEEE floating epilogues; fold only the exact integer zero numerator."""
    from xdsl.dialects import arith, tensor
    from xdsl.dialects.builtin import IntegerAttr, IntegerType
    from xdsl.rewriter import Rewriter

    unknown = object()
    memo = {}
    depth = 0

    def scalar(op, values):
        name = op.name
        if name == "arith.constant":
            attr = op.properties.get("value", op.attributes.get("value"))
            if hasattr(attr, "get_values"):
                xs = list(attr.get_values())
                result = xs[0] if xs and all(v == xs[0] for v in xs) else unknown
            else:
                result = attr.value.data
            return result if result is not unknown and (isinstance(result, int) or math.isfinite(result)) else unknown
        integer = isinstance(op.results[0].type, IntegerType) if op.results else False
        # Index arithmetic has a target-dependent width. It can select a
        # literal threshold, but cannot justify a fixed-width zero payload.
        if op.results and not integer and str(op.results[0].type) == "index":
            return unknown
        if name in {"arith.extsi", "arith.extui", "arith.trunci"} and values[0] is not unknown:
            width = op.results[0].type.width.data
            value = int(values[0])
            if name == "arith.extui":
                value &= (1 << op.operands[0].type.width.data) - 1
            value &= (1 << width) - 1
            return value - (1 << width) if value >> (width - 1) else value
        if name in {"arith.extf", "arith.truncf", "arith.fptosi", "arith.fptoui", "arith.sitofp", "arith.uitofp"}:
            # Zero survives these numeric conversions exactly, including -0 -> 0.
            return 0 if values[0] is not unknown and values[0] == 0 else unknown
        if name in {"arith.muli", "arith.andi"} and any(v is not unknown and v == 0 for v in values):
            return 0
        if name in {"arith.addi", "arith.subi", "arith.muli", "arith.andi", "arith.ori", "arith.xori"}:
            if any(v is unknown for v in values):
                return unknown
            a, b = values
            value = {
                "arith.addi": lambda: a + b,
                "arith.subi": lambda: a - b,
                "arith.muli": lambda: a * b,
                "arith.andi": lambda: a & b,
                "arith.ori": lambda: a | b,
                "arith.xori": lambda: a ^ b,
            }[name]()
            width = op.results[0].type.width.data
            value &= (1 << width) - 1
            return value - (1 << width) if value >> (width - 1) else value
        if name == "arith.mulf" and all(v is not unknown for v in values) and 0 in values:
            return 0  # Every other known operand is finite; no inf*0 inference.
        if name == "arith.divf" and all(v is not unknown for v in values) and values[0] == 0 and values[1] != 0:
            return 0
        if name == "arith.negf" and values[0] is not unknown and values[0] == 0:
            return 0
        if name in {"arith.shli", "arith.shrsi", "arith.shrui"} and values[1] is not unknown:
            if 0 <= values[1] < op.results[0].type.width.data and values[0] is not unknown and values[0] == 0:
                return 0
        if name == "arith.select" and values[0] is not unknown:
            return values[1] if values[0] else values[2]
        if integer and name == "arith.select" and values[1] is not unknown and values[1] == values[2]:
            return values[1]
        return unknown

    def uniform(value):
        nonlocal depth
        # This optional source-literal optimization must not exhaust Python's
        # stack on complete full-dtype limb reconstruction chains. Unknown
        # retains the original exact producer; it never changes admission.
        if depth >= 128:
            return unknown
        depth += 1
        try:
            return visit(value)
        finally:
            depth -= 1

    def visit(value):
        if value in memo:
            return memo[value]
        memo[value] = unknown
        op = value.owner
        name = getattr(op, "name", None)
        if name == "arith.constant":
            result = scalar(op, [])
        elif name in {
            "tensor.splat",
            "linalg.fill",
            "tensor.reshape",
            "tensor.expand_shape",
            "tensor.collapse_shape",
            "tensor.extract_slice",
            "linalg.copy",
            "linalg.transpose",
            "linalg.broadcast",
        }:
            result = uniform(op.operands[0])
        elif name == "linalg.generic" and len(op.results) == 1:
            env = dict(zip(op.body.block.args, [uniform(v) for v in op.operands], strict=True))
            for child in op.body.block.ops:
                if child.name == "linalg.yield":
                    result = env.get(child.operands[0], unknown)
                    break
                if (
                    child.regions
                    or len(child.results) != 1
                    or not (child.name.startswith("arith.") or child.name == "linalg.index")
                ):
                    result = unknown
                    break
                env[child.results[0]] = scalar(child, [env.get(v, unknown) for v in child.operands])
            else:
                result = unknown
            if any(it.data.value == "reduction" for it in op.get_iterator_types()):
                if uniform(op.outputs[0]) != 0 or result != 0:
                    result = unknown
        else:
            result = unknown
        memo[value] = result
        return result

    count = 0
    for op in list(module.walk()):
        if limb_attribute not in op.attributes or len(op.results) != 1:
            continue
        typ = op.results[0].type
        if not isinstance(typ.element_type, IntegerType) or uniform(op.results[0]) != 0:
            continue
        zero = arith.ConstantOp(IntegerAttr(0, typ.element_type))
        replacement = tensor.SplatOp(zero.result, [], typ)
        Rewriter.replace_op(op, [zero, replacement], [replacement.result])
        count += 1
    module.verify()
    return count
