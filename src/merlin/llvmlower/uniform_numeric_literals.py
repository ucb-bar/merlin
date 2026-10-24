"""Fold uniform source literals and their explicit typed conversion boundaries."""


def compile_time_operand(value):
    """Prove that a value has no runtime payload dependencies.

    An index-dependent constant tensor is allowed here; the separate native-unit
    construction policy still determines whether its use is admissible.
    """
    from merlin.targetgen.native_construction_patterns import dataflow

    def dependencies(current):
        owner = current.owner
        name = getattr(owner, "name", None)
        if name in {"arith.constant", "linalg.index"}:
            return True
        if name == "linalg.generic":
            return (owner.body.block.last_op.operands[current.index],)
        if name is None and hasattr(owner, "parent_op"):
            parent = owner.parent_op()
            if getattr(parent, "name", None) == "linalg.generic" and current.index < len(parent.inputs):
                return (parent.inputs[current.index],)
        if name in {"tensor.splat", "tensor.cast", "tensor.reshape"}:
            return (owner.operands[0],)
        if name is not None and name.startswith("arith."):
            return tuple(owner.operands) if owner.operands else False
        return False

    return dataflow(value, {}, dependencies, all)


def fold_uniform_numeric_literals(module, *, close_guards=False):
    """Close literal conversion/guard chains to a fixed point, without data."""
    if not close_guards:
        return _fold_uniform_numeric_literals_once(module, close_guards=False)
    total = 0
    while count := _fold_uniform_numeric_literals_once(module, close_guards=True):
        total += count
    return total


def _fold_uniform_numeric_literals_once(module, *, close_guards):
    import math
    from fractions import Fraction

    from xdsl.dialects import arith, tensor
    from xdsl.dialects.builtin import FloatAttr, IntegerAttr
    from xdsl.rewriter import Rewriter

    from merlin.llvmlower.fixed_point_ir import FORMAT
    from merlin.targetgen.closed_scalar import predicate_holds
    from merlin.targetgen.exact_arithmetic import round_binary
    from merlin.xdsl_dialects.lowering.group_numerics import _constant_of

    def evaluate(primitive, operands):
        name = primitive.name
        if name == "arith.select" and operands[0] is not None:
            return operands[1 if operands[0] else 2]
        if any(v is None or not math.isfinite(v) for v in operands):
            return None
        if name == "arith.cmpf":
            a, b = operands
            return int(predicate_holds(primitive.predicate.value.data, (a > b) - (a < b), floating=True))
        rational = list(map(Fraction, operands))
        negative = [math.copysign(1.0, v) < 0 for v in operands]
        negative_zero = False
        if name in {"arith.extf", "arith.truncf", "arith.sitofp", "arith.uitofp"}:
            if str(primitive.operands[0].type) == "f64":
                return None
            result = rational[0]
            negative_zero = negative[0] and not result
            if name == "arith.uitofp":
                result = Fraction(int(result) % (1 << primitive.operands[0].type.width.data))
        elif name in {"arith.addf", "arith.subf"}:
            subtract = name == "arith.subf"
            result = rational[0] + (-rational[1] if subtract else rational[1])
            negative_zero = not any(rational) and negative[0] and (negative[1] != subtract)
        elif name == "arith.mulf":
            result = rational[0] * rational[1]
            negative_zero = negative[0] != negative[1]
        elif name == "arith.negf":
            result = -rational[0]
            negative_zero = not negative[0]
        elif name == "math.fma":
            product = rational[0] * rational[1]
            result = product + rational[2]
            negative_zero = not product and not rational[2] and (negative[0] != negative[1]) and negative[2]
        else:
            return None
        token = str(primitive.results[0].type)
        if token not in {"f16", "bf16", "f32"}:
            return None
        try:
            rounded = round_binary(
                result, 16 if token in {"f16", "bf16"} else 32 if token == "f32" else 64, bfloat16=token == "bf16"
            )
        except (ValueError, OverflowError):
            return None
        return -0.0 if not result and negative_zero else rounded

    count = 0
    for op in list(module.walk()):
        if len(op.results) != 1 or not op.operands or str(op.results[0].type) not in {*FORMAT, "i1"}:
            continue
        values = [_constant_of(v) for v in op.operands]
        if close_guards and op.name == "arith.select" and values[0] is not None:
            chosen = op.operands[1 if values[0] else 2]
            Rewriter.replace_op(op, [], [chosen])
            count += 1
            continue
        value = evaluate(op, values)
        if value is None or not math.isfinite(value):
            continue
        typ = op.results[0].type
        attr = IntegerAttr(value, typ) if str(typ) == "i1" else FloatAttr(value, typ)
        Rewriter.replace_op(op, arith.ConstantOp(attr))
        count += 1
    for op in list(module.walk()):
        if op.name != "linalg.generic" or len(op.results) != 1:
            continue
        if any(it.data.value != "parallel" for it in op.get_iterator_types()):
            continue
        body = list(op.body.block.ops)
        if len(body) == 2 and body[1].name == "linalg.yield" and list(body[1].operands) == list(body[0].results):
            primitive = body[0]
            if (
                str(primitive.results[0].type) in {*FORMAT, "i1"}
                and primitive.operands
                and all(v in op.body.block.args[: len(op.inputs)] for v in primitive.operands)
            ):
                values = [_constant_of(op.inputs[op.body.block.args.index(v)]) for v in primitive.operands]
                value = evaluate(primitive, values)
                if value is not None and math.isfinite(value):
                    typ = primitive.results[0].type
                    attr = IntegerAttr(value, typ) if str(typ) == "i1" else FloatAttr(value, typ)
                    literal = arith.ConstantOp(attr)
                    filled = tensor.SplatOp(literal.results[0], [], op.results[0].type)
                    Rewriter.replace_op(op, [literal, filled], [filled.result])
                    count += 1
                    continue
        if (
            len(body) == 2
            and body[0].name == "arith.constant"
            and body[1].name == "linalg.yield"
            and list(body[1].operands) == list(body[0].results)
        ):
            attr = body[0].properties.get("value")
            if not isinstance(attr, (FloatAttr, IntegerAttr)) or not math.isfinite(attr.value.data):
                continue
            literal = body[0].clone()
            filled = tensor.SplatOp(literal.results[0], [], op.results[0].type)
            Rewriter.replace_op(op, [literal, filled], [filled.result])
            count += 1
    module.verify()
    return count
