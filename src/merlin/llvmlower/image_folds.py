"""Closed identity folds proved over the complete image of a conversion.

A Boolean word, the image of uitofp/sitofp of an integer, an integral IEEE
value or a NaN-word guard determines each rewrite from the IR alone. Every
fold refuses relaxed (fastmath/overflow) arithmetic and payload-UNKNOWN
values; arbitrary floating inputs retain their operations.
"""

import math
from fractions import Fraction

from xdsl.dialects import arith, tensor
from xdsl.dialects.builtin import AffineMapAttr, FloatAttr, IntegerAttr, IntegerType, i1
from xdsl.ir.affine import AffineMap
from xdsl.rewriter import Rewriter
from xdsl.transforms.dead_code_elimination import region_dce

from merlin.llvmlower._ir import generic_input, identity, input_literal, point, splat
from merlin.llvmlower.fixed_point_ir import FORMAT
from merlin.targetgen.closed_scalar import predicate_holds
from merlin.targetgen.exact_arithmetic import round_binary
from merlin.targetgen.native_payload_carriers import UNKNOWN, LiteralPayloads
from merlin.xdsl_dialects.lowering.group_numerics import _constant_of


def _parallel(op):
    return all(it.data.value == "parallel" for it in op.get_iterator_types())


def _relaxed(op):
    flags = op.properties.get("fastmath")
    return flags is not None and flags.data


def _replace_by_constant(op, attr):
    constant = arith.ConstantOp(attr)
    Rewriter.replace_op(op, [constant], [constant.result])


def _yielded(owner):
    """The scalar a one-result, all-parallel linalg.generic yields, else None."""
    if getattr(owner, "name", None) == "linalg.generic" and len(owner.results) == 1 and _parallel(owner):
        return owner.body.block.last_op.operands[0]
    return None


def fold_boolean_domain(module):
    """Fold closed one-Boolean integer programs only to identity or uniform fill.

    Both possible storage words are evaluated exactly, with every original integer
    width boundary. An arbitrary two-entry result table is never emitted.
    """
    payloads = LiteralPayloads()
    count = 0
    for op in list(module.walk()):
        if op.name != "linalg.generic" or len(op.results) != 1 or not _parallel(op):
            continue
        element = op.results[0].type.element_type
        if not isinstance(element, IntegerType):
            continue
        runtime = [i for i, value in enumerate(op.inputs) if _constant_of(value) is None]
        if len(runtime) != 1 or str(op.inputs[runtime[0]].type.element_type) != "i1":
            continue
        maps = list(op.get_indexing_maps())
        rank = len(op.results[0].type.get_shape())
        if maps[-1] != identity(rank):
            continue
        if payloads.classify(op.inputs[runtime[0]]) & UNKNOWN:
            continue
        body = op.body.block
        operations = list(body.ops)
        if len(op.inputs) == 1 and (
            (
                len(operations) == 1
                and operations[0].name == "linalg.yield"
                and operations[0].operands[0] is body.args[0]
            )
            or (
                len(operations) == 2
                and operations[0].name == "arith.extui"
                and operations[0].operands[0] is body.args[0]
                and list(operations[1].operands) == list(operations[0].results)
            )
        ):
            continue
        answers = []
        evaluations = []
        valid = True
        for bit in (0, 1):
            env = {body.args[runtime[0]]: bit}
            for i, value in enumerate(op.inputs):
                if i != runtime[0]:
                    literal = _constant_of(value)
                    if not isinstance(literal, int) or not isinstance(value.type.element_type, IntegerType):
                        valid = False
                        break
                    env[body.args[i]] = literal % (1 << value.type.element_type.width.data)
            if not valid:
                break
            for child in body.ops:
                if child.name == "linalg.yield":
                    if child.operands[0] not in env:
                        valid = False
                        break
                    answers.append(env[child.operands[0]])
                    break
                if child.name == "cf.assert":
                    if len(child.operands) != 1 or child.operands[0] not in env or not env[child.operands[0]]:
                        valid = False
                        break
                    continue
                if len(child.results) != 1 or not isinstance(child.results[0].type, IntegerType) or child.regions:
                    valid = False
                    break
                flags = child.properties.get("overflowFlags")
                if flags is not None and str(flags).split("<")[-1] != "none>":
                    valid = False
                    break
                width = child.results[0].type.width.data
                if child.name == "arith.constant" and isinstance(child.value, IntegerAttr):
                    value = child.value.value.data
                elif any(v not in env for v in child.operands):
                    valid = False
                    break
                else:
                    xs = [env[v] for v in child.operands]
                    if child.name == "arith.select":
                        value = xs[1 if xs[0] else 2]
                    elif child.name == "arith.cmpi":
                        a, b = xs
                        predicate = child.predicate.value.data
                        if predicate in {2, 3, 4, 5}:
                            bits = child.operands[0].type.width.data
                            a = a - (1 << bits) if a >= 1 << (bits - 1) else a
                            b = b - (1 << bits) if b >= 1 << (bits - 1) else b
                        value = predicate_holds(predicate, (a > b) - (a < b))
                    elif child.name in {"arith.divsi", "arith.floordivsi", "arith.remsi", "arith.divui", "arith.remui"}:
                        a, b = xs
                        bits = child.operands[0].type.width.data
                        signed = child.name not in {"arith.divui", "arith.remui"}
                        if signed:
                            a = a - (1 << bits) if a >= 1 << (bits - 1) else a
                            b = b - (1 << bits) if b >= 1 << (bits - 1) else b
                        if not b or (signed and a == -(1 << (bits - 1)) and b == -1):
                            valid = False
                            break
                        quotient = abs(a) // abs(b)
                        if (a < 0) != (b < 0):
                            quotient = -quotient
                        value = (
                            a // b
                            if child.name == "arith.floordivsi"
                            else a - quotient * b
                            if child.name in {"arith.remsi", "arith.remui"}
                            else quotient
                        )
                    elif child.name in {"arith.extui", "arith.trunci"}:
                        value = xs[0]
                    elif child.name == "arith.extsi":
                        bits = child.operands[0].type.width.data
                        value = xs[0] - (1 << bits) if xs[0] >= 1 << (bits - 1) else xs[0]
                    elif child.name in {
                        "arith.andi",
                        "arith.ori",
                        "arith.xori",
                        "arith.addi",
                        "arith.subi",
                        "arith.muli",
                    }:
                        a, b = xs
                        value = {
                            "arith.andi": lambda: a & b,
                            "arith.ori": lambda: a | b,
                            "arith.xori": lambda: a ^ b,
                            "arith.addi": lambda: a + b,
                            "arith.subi": lambda: a - b,
                            "arith.muli": lambda: a * b,
                        }[child.name]()
                    else:
                        valid = False
                        break
                env[child.results[0]] = value % (1 << width)
            if not valid:
                break
            evaluations.append(env)
        if valid and len(evaluations) == 2:
            # A two-word evaluation must not erase a runtime product/OR or
            # runtime quotient construction merely because its outputs coincide.
            def static(value):
                return (
                    value in evaluations[0]
                    and value in evaluations[1]
                    and evaluations[0][value] == evaluations[1][value]
                )

            for child in body.ops:
                if child.name in {"arith.andi", "arith.ori", "arith.xori", "arith.addi", "arith.subi", "arith.muli"}:
                    if not any(static(v) for v in child.operands):
                        valid = False
                        break
                elif child.name in {"arith.divsi", "arith.floordivsi", "arith.remsi", "arith.divui", "arith.remui"}:
                    if not static(child.operands[1]):
                        valid = False
                        break
        if not valid or len(answers) != 2 or not (answers[0] == answers[1] or answers == [0, 1]):
            continue
        replacement = []
        if answers[0] == answers[1]:
            result = splat(answers[0], element, op.results[0].type.get_shape(), replacement)
        else:
            # Keep the original selector map; only the proved scalar body changes.
            result = point(
                [op.inputs[runtime[0]]],
                op.results[0].type,
                lambda args, emit: args[0] if element.width.data == 1 else emit(arith.ExtUIOp(args[0], element)),
                replacement,
                maps=[maps[runtime[0]], maps[-1]],
                iterators=op.get_iterator_types(),
            )
        Rewriter.replace_op(op, replacement, [result])
        count += 1
    return count


def fold_boolean_image_predicates(module):
    """Prove storage selectors over the complete image of one Boolean conversion."""
    payloads = LiteralPayloads()
    changed = 0
    for op in list(module.walk()):
        if op.name != "linalg.generic" or len(op.inputs) != 2 or len(op.results) != 1:
            continue
        if str(op.results[0].type.element_type) != "i1":
            continue
        body = list(op.body.block.ops)
        if len(body) != 2 or body[0].name != "arith.cmpf" or body[1].name != "linalg.yield" or body[0].fastmath.data:
            continue
        if list(body[0].operands) != list(op.body.block.args[:2]) or list(body[1].operands) != list(body[0].results):
            continue
        if not _parallel(op):
            continue
        for slot in (0, 1):
            conversion = op.inputs[slot].owner
            constant = _constant_of(op.inputs[1 - slot])
            if not isinstance(constant, (int, float)) or getattr(conversion, "name", None) != "linalg.generic":
                continue
            if len(conversion.inputs) != 1 or len(conversion.results) != 1:
                continue
            source = conversion.inputs[0]
            if str(source.type.element_type) != "i1" or payloads.classify(source) & UNKNOWN:
                continue
            scalar = list(conversion.body.block.ops)
            if (
                len(scalar) != 2
                or scalar[0].name not in {"arith.uitofp", "arith.sitofp"}
                or scalar[1].name != "linalg.yield"
            ):
                continue
            if list(scalar[0].operands) != [conversion.body.block.args[0]]:
                continue
            if list(scalar[1].operands) != list(scalar[0].results):
                continue
            rank = len(conversion.results[0].type.get_shape())
            maps = list(conversion.get_indexing_maps())
            if not _parallel(conversion):
                continue
            if maps[-1].data != AffineMap.identity(rank):
                continue
            answers = []
            for bit in (0, 1):
                value = -bit if scalar[0].name == "arith.sitofp" else bit
                a, b = (value, constant) if slot == 0 else (constant, value)
                answers.append(
                    predicate_holds(
                        body[0].predicate.value.data, (a > b) - (a < b), math.isnan(constant), floating=True
                    )
                )
            replacement = []
            if answers[0] == answers[1]:
                result = splat(int(answers[0]), i1, op.results[0].type.get_shape(), replacement)
            elif answers == [False, True]:
                indexing = maps[0].data.compose(list(op.get_indexing_maps())[slot].data)
                result = point(
                    [source],
                    op.results[0].type,
                    lambda args, emit: args[0],
                    replacement,
                    maps=[AffineMapAttr(indexing), list(op.get_indexing_maps())[-1]],
                    iterators=op.get_iterator_types(),
                )
            else:
                continue
            Rewriter.replace_op(op, replacement, [result])
            changed += 1
            break
    if changed:
        module.verify()
    return changed


def fold_boolean_unit_image(module):
    """Erase arithmetic identities proved over the complete unsigned Boolean image."""
    payloads = LiteralPayloads()
    memo = {}

    def literal(value):
        source = generic_input(value)
        value = value if source is None else source
        owner = value.owner
        if getattr(owner, "name", None) == "arith.bitcast":
            bits = _constant_of(owner.operands[0])
            width = getattr(getattr(owner.operands[0].type, "width", None), "data", None)
            if isinstance(bits, int) and width in {16, 32, 64} and str(value.type) in {"f16", "bf16", "f32", "f64"}:
                raw = (bits % (1 << width)).to_bytes(width // 8, "little")
                if len(value.type.pack([0.0])) == len(raw):
                    return value.type.unpack(raw, 1)[0]
        return _constant_of(value)

    def unit(value):
        if value in memo:
            return memo[value]
        memo[value] = False
        owner = value.owner
        name = getattr(owner, "name", None)
        source = generic_input(value)
        answer = False
        if name == "arith.uitofp" and str(owner.operands[0].type) == "i1":
            answer = True
        elif name in {"arith.extf", "arith.truncf"}:
            answer = unit(owner.operands[0])
        elif source is not None:
            answer = unit(source)
        elif _yielded(owner) is not None:
            answer = unit(_yielded(owner))
        if answer and payloads.classify(value) & UNKNOWN:
            answer = False
        memo[value] = answer
        return answer

    changed = 0
    for op in list(module.walk()):
        if any(payloads.classify(value) & UNKNOWN for value in op.operands):
            continue
        if op.name == "cf.assert" and literal(op.operands[0]):
            Rewriter.erase_op(op)
            changed += 1
            continue
        if len(op.results) != 1 or _relaxed(op):
            continue
        name = getattr(getattr(op, "op_name", None), "data", op.name)
        replacement = None
        zero = False
        integer_zero = False
        predicate_literal = None
        if name == "arith.cmpf":
            a, b = [literal(v) for v in op.operands]
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                predicate_literal = predicate_holds(
                    op.predicate.value.data, (a > b) - (a < b), math.isnan(a) or math.isnan(b), floating=True
                )
            else:
                for slot in (0, 1):
                    threshold = literal(op.operands[1 - slot])
                    if not unit(op.operands[slot]) or not isinstance(threshold, (int, float)):
                        continue
                    answers = []
                    for bit in (0.0, 1.0):
                        left, right = (bit, threshold) if slot == 0 else (threshold, bit)
                        answers.append(
                            predicate_holds(
                                op.predicate.value.data,
                                (left > right) - (left < right),
                                math.isnan(threshold),
                                floating=True,
                            )
                        )
                    if answers[0] == answers[1]:
                        predicate_literal = answers[0]
                        break
        elif name in {"arith.andi", "arith.ori", "arith.xori"} and str(op.results[0].type) == "i1":
            a, b = [literal(v) for v in op.operands]
            if a is not None and b is not None:
                predicate_literal = {
                    "arith.andi": bool(a) and bool(b),
                    "arith.ori": bool(a) or bool(b),
                    "arith.xori": bool(a) != bool(b),
                }[name]
            else:
                for slot, bound in enumerate((a, b)):
                    if bound is None:
                        continue
                    if name == "arith.andi" and not bound:
                        predicate_literal = False
                    elif name == "arith.ori" and bound:
                        predicate_literal = True
                    elif (name == "arith.andi" and bound) or (name in {"arith.ori", "arith.xori"} and not bound):
                        replacement = op.operands[1 - slot]
                    break
        elif name == "arith.select":
            condition = literal(op.operands[0])
            if condition is not None:
                replacement = op.operands[1 if condition else 2]
        if name == "math.copysign" and len(op.operands) == 2:
            magnitude = literal(op.operands[0])
            if isinstance(magnitude, (int, float)) and math.isfinite(magnitude) and unit(op.operands[1]):
                _replace_by_constant(op, FloatAttr(abs(magnitude), op.results[0].type))
                changed += 1
                memo.clear()
                continue
        if name == "arith.subf" and op.operands[0] is op.operands[1] and unit(op.operands[0]):
            zero = True  # Both Boolean floating words are finite and nonnegative.
        if name == "arith.andi":
            for slot in (0, 1):
                cast = op.operands[slot].owner
                mask = literal(op.operands[1 - slot])
                if getattr(cast, "name", None) == "arith.bitcast" and isinstance(mask, int) and unit(cast.operands[0]):
                    typ = cast.operands[0].type
                    one = int.from_bytes(typ.pack([1.0]), "little")
                    if not mask & one:
                        integer_zero = True
        if name in {"math.sqrt", "math.floor", "math.ceil", "math.trunc", "math.round", "math.roundeven"}:
            if len(op.operands) == 1 and unit(op.operands[0]):
                replacement = op.operands[0]
        elif name in {"arith.divf", "arith.remf", "arith.subf", "arith.addf", "math.powf"} and len(op.operands) == 2:
            data, constant = op.operands
            bound = literal(constant)
            if isinstance(bound, (int, float)) and math.isfinite(bound) and unit(data):
                if name == "arith.divf" and bound == 1:
                    replacement = data
                elif name == "math.powf" and bound > 0:
                    replacement = data
                elif name in {"arith.addf", "arith.subf"} and bound == 0 and math.copysign(1.0, bound) > 0:
                    replacement = data
                elif name == "arith.remf" and abs(bound) > 1:
                    replacement = data
                elif name == "arith.remf" and bound and (1 / abs(Fraction(bound))).denominator == 1:
                    zero = True
        if name == "arith.select" and replacement is None:
            predicate = op.operands[0].owner
            if getattr(predicate, "name", None) == "arith.cmpf" and not predicate.fastmath.data:
                for reverse in (False, True):
                    data = predicate.operands[int(reverse)]
                    threshold = literal(predicate.operands[int(not reverse)])
                    if not unit(data) or not isinstance(threshold, (int, float)):
                        continue
                    answers = []
                    for bit in (0.0, 1.0):
                        a, b = (threshold, bit) if reverse else (bit, threshold)
                        yes = predicate_holds(
                            predicate.predicate.value.data, (a > b) - (a < b), math.isnan(threshold), floating=True
                        )
                        chosen = op.operands[1 if yes else 2]
                        value = bit if chosen is data else literal(chosen)
                        if not isinstance(value, (int, float)):
                            break
                        answers.append(value)
                    if len(answers) == 2 and data.type == op.results[0].type:
                        if all(
                            data.type.pack([value]) == data.type.pack([bit])
                            for value, bit in zip(answers, (0.0, 1.0), strict=True)
                        ):
                            replacement = data
                            break
        if integer_zero:
            _replace_by_constant(op, IntegerAttr(0, op.results[0].type))
            changed += 1
        elif predicate_literal is not None:
            _replace_by_constant(op, IntegerAttr(int(predicate_literal), i1))
            changed += 1
        elif replacement is not None and replacement.type == op.results[0].type:
            Rewriter.replace_op(op, [], [replacement])
            changed += 1
        elif zero:
            _replace_by_constant(op, FloatAttr(0.0, op.results[0].type))
            changed += 1
        if replacement is not None or zero:
            memo.clear()
    if changed:
        region_dce(module.body)
    return changed


def fold_integer_image_predicates(module):
    """Fold comparisons proved uniform over an integer-to-float image.

    IEEE RNE of an integer is an integral finite value or infinity, never NaN.
    Monotonic rounded dtype endpoints and that discrete-image fact are independent
    of tensor payloads; arbitrary floating inputs retain their predicates.
    """
    memo = {}
    payloads = LiteralPayloads()

    def rounded(value, typ):
        if not math.isfinite(value):
            return value
        token = str(typ)
        if token not in {"f16", "bf16", "f32", "f64"}:
            raise ValueError("unproved floating format")
        try:
            return round_binary(
                Fraction(value),
                16 if token in {"f16", "bf16"} else 64 if token == "f64" else 32,
                bfloat16=token == "bf16",
                overflow="infinity",
            )
        except OverflowError:
            return math.copysign(math.inf, value)

    def domain(value):
        if value in memo:
            return memo[value]
        memo[value] = None
        owner = value.owner
        name = getattr(owner, "name", None)
        source = generic_input(value)
        result = None
        if name in {"arith.sitofp", "arith.uitofp"}:
            typ = owner.operands[0].type
            if isinstance(typ, IntegerType):
                width = typ.width.data
                lo, hi = (
                    (-(1 << (width - 1)), (1 << (width - 1)) - 1) if name == "arith.sitofp" else (0, (1 << width) - 1)
                )
                result = rounded(lo, value.type), rounded(hi, value.type)
        elif name in {"arith.extf", "arith.truncf"}:
            original = domain(owner.operands[0])
            if original is not None:
                result = tuple(rounded(v, value.type) for v in original)
        elif source is not None:
            result = domain(source)
        elif name == "linalg.generic":
            ops = list(owner.body.block.ops)
            if (
                len(owner.results) == 1
                and len(ops) == 2
                and ops[-1].name == "linalg.yield"
                and list(ops[-1].operands) == list(ops[0].results)
            ):
                result = domain(ops[0].results[0])
        elif name in {
            "tensor.reshape",
            "tensor.cast",
            "tensor.expand_shape",
            "tensor.collapse_shape",
            "tensor.extract_slice",
            "tensor.extract",
            "linalg.transpose",
            "linalg.copy",
            "linalg.broadcast",
        }:
            result = domain(owner.operands[0])
        if result is not None and payloads.classify(value) & UNKNOWN:
            result = None
        memo[value] = result
        return result

    count = 0
    for op in list(module.walk()):
        if op.name != "arith.cmpf" or op.fastmath.data:
            continue
        for reverse in (False, True):
            data = op.operands[int(reverse)]
            constant = input_literal(op.operands[int(not reverse)])
            if not isinstance(constant, (int, float)):
                continue
            try:
                bounds = domain(data)
            except (TypeError, ValueError, OverflowError):
                bounds = None
            if bounds is None:
                continue
            pred = op.predicate.value.data
            lo, hi = bounds
            answer = None
            if math.isnan(constant):
                answer = predicate_holds(pred, 0, True, floating=True)
            elif pred in {1, 6, 8, 13}:
                unattainable = (
                    (math.isfinite(constant) and not float(constant).is_integer()) or constant < lo or constant > hi
                )
                if unattainable:
                    answer = pred in {6, 13}
            elif pred in {2, 3, 4, 5, 9, 10, 11, 12}:
                relations = [
                    ((constant > x) - (constant < x)) if reverse else ((x > constant) - (x < constant)) for x in bounds
                ]
                answers = {predicate_holds(pred, relation, floating=True) for relation in relations}
                if len(answers) == 1:
                    answer = answers.pop()
            if answer is not None:
                _replace_by_constant(op, IntegerAttr(int(answer), i1))
                count += 1
                break
    if count:
        for op in list(module.walk()):
            if op.name != "linalg.generic" or len(op.results) != 1:
                continue
            rank = len(op.results[0].type.get_shape())
            if not _parallel(op) or list(op.get_indexing_maps())[-1] != AffineMapAttr(AffineMap.identity(rank)):
                continue
            body = list(op.body.block.ops)
            if (
                len(body) == 2
                and body[0].name == "arith.constant"
                and body[1].name == "linalg.yield"
                and list(body[1].operands) == list(body[0].results)
            ):
                constant = body[0].clone()
                filled = tensor.SplatOp(constant.results[0], [], op.results[0].type)
                Rewriter.replace_op(op, [constant, filled], [filled.result])
    return count


def fold_integral_float_rounding(module):
    """Integral IEEE images are fixed points of floor/ceil/trunc/RNE."""
    payloads = LiteralPayloads()
    memo = {}

    def integral(value):
        if value in memo:
            return memo[value]
        memo[value] = False
        if payloads.classify(value) & UNKNOWN:
            return False
        owner = value.owner
        name = getattr(owner, "name", None)
        source = generic_input(value)
        result = False
        if source is not None:
            result = integral(source)
        elif name in {"arith.sitofp", "arith.uitofp"}:
            typ = owner.operands[0].type
            result = isinstance(typ, IntegerType) and typ.width.data <= 64
        elif name in {"arith.extf", "arith.truncf"}:
            result = integral(owner.operands[0])
        elif name == "arith.divf":
            divisor = input_literal(owner.operands[1])
            if isinstance(divisor, (int, float)) and divisor and math.isfinite(divisor):
                reciprocal = 1 / abs(Fraction(divisor))
                # Power-of-two reciprocal has no fractional image or
                # intermediate rounding other than IEEE overflow to inf.
                n = reciprocal.numerator
                result = reciprocal.denominator == 1 and n & (n - 1) == 0 and integral(owner.operands[0])
        elif _yielded(owner) is not None:
            result = integral(_yielded(owner))
        memo[value] = result
        return result

    def finite_integer_cast(value):
        """A <=64-bit integer conversion to binary32/64 cannot overflow."""
        owner = value.owner
        name = getattr(owner, "name", None)
        if name in {"arith.sitofp", "arith.uitofp"} and str(value.type) in {"f32", "f64", "bf16"}:
            typ = owner.operands[0].type
            return isinstance(typ, IntegerType) and typ.width.data <= 64
        if generic_input(value) is not None:
            return finite_integer_cast(generic_input(value))
        return _yielded(owner) is not None and finite_integer_cast(_yielded(owner))

    count = 0
    # Only finite integer images license these identities. Infinities from an
    # overflowing narrow cast would make remainder NaN and must stay intact.
    for op in list(module.walk()):
        name = getattr(getattr(op, "op_name", None), "data", op.name)
        if name not in {"arith.divf", "arith.remf"} or input_literal(op.operands[1]) != 1 or _relaxed(op):
            continue
        data = op.operands[0]
        if not finite_integer_cast(data) or payloads.classify(data) & UNKNOWN:
            continue
        if name == "arith.divf":
            Rewriter.replace_op(op, [], [data])
            count += 1
            continue
        result = op.results[0]
        safe = True
        subtractions = []
        for use in list(result.uses):
            child = use.operation
            if child.name == "arith.subf" and use.index == 1 and child.operands[0] is data and not child.fastmath.data:
                subtractions.append(child)
            elif (
                child.name == "arith.cmpf"
                and input_literal(child.operands[1 - use.index]) == 0
                and not child.fastmath.data
            ):
                continue
            else:
                safe = False
                break
        if not safe:
            continue
        for child in subtractions:
            Rewriter.replace_op(child, [], [data])
        # Remaining comparisons cannot observe the sign of zero.
        _replace_by_constant(op, FloatAttr(0.0, result.type))
        count += 1
    for op in list(module.walk()):
        name = getattr(getattr(op, "op_name", None), "data", op.name)
        if name not in {"math.floor", "math.ceil", "math.trunc", "math.round", "math.roundeven"} or _relaxed(op):
            continue
        if (
            len(op.operands) == len(op.results) == 1
            and op.operands[0].type == op.results[0].type
            and str(op.results[0].type) in {"f16", "bf16", "f32", "f64"}
            and integral(op.operands[0])
        ):
            Rewriter.replace_op(op, [], [op.operands[0]])
            count += 1
    for op in list(module.walk()):
        if (
            op.name == "arith.subf"
            and not op.fastmath.data
            and op.operands[0] is op.operands[1]
            and finite_integer_cast(op.operands[0])
        ):
            _replace_by_constant(op, FloatAttr(0.0, op.results[0].type))
            count += 1
    return count


def fold_integral_image_identities(module):
    """Close nested guards after each preceding identity exposes their image."""
    total = 0
    while True:
        changed = (
            fold_integral_float_rounding(module)
            + fold_boolean_unit_image(module)
            + fold_integer_image_predicates(module)
            + fold_zero_sign_selectors(module)
        )
        total += changed
        if not changed:
            return total


def fold_ieee_identity_selectors(module):
    """Keep the quieting arithmetic branch when an ordered selector is redundant."""
    payloads = LiteralPayloads()

    def parallel(op):
        if getattr(op, "name", None) != "linalg.generic" or len(op.results) != 1:
            return False
        identity = AffineMap.identity(len(op.results[0].type.get_shape()))
        return _parallel(op) and all(m.data == identity for m in op.get_indexing_maps())

    def product_is_identity(value, data):
        product = value.owner
        if not parallel(product) or value.type != data.type:
            return False
        body = list(product.body.block.ops)
        if len(body) != 2 or body[0].name != "arith.mulf" or body[1].name != "linalg.yield":
            return False
        if body[0].fastmath.data or list(body[1].operands) != list(body[0].results):
            return False
        operands = []
        for operand in body[0].operands:
            if operand not in product.body.block.args[: len(product.inputs)]:
                return False
            operands.append(product.inputs[operand.index])
        return any(operands[i] is data and _constant_of(operands[1 - i]) == 1.0 for i in (0, 1))

    count = 0
    for op in list(module.walk()):
        if not parallel(op) or len(op.inputs) != 3 or any(payloads.classify(v) & UNKNOWN for v in op.inputs):
            continue
        body = list(op.body.block.ops)
        if len(body) != 2 or body[0].name != "arith.select" or body[1].name != "linalg.yield":
            continue
        if list(body[0].operands) != list(op.body.block.args[:3]) or list(body[1].operands) != list(body[0].results):
            continue
        predicate = op.inputs[0].owner
        if not parallel(predicate):
            continue
        comparison = list(predicate.body.block.ops)
        if len(comparison) != 2 or comparison[0].name != "arith.cmpf" or comparison[0].fastmath.data:
            continue
        if list(comparison[1].operands) != list(comparison[0].results):
            continue
        if list(comparison[0].operands) != list(predicate.body.block.args[:2]):
            continue
        if len(predicate.inputs) != 2:
            continue
        p = comparison[0].predicate.value.data
        # On NaN retain the original multiplication, never replace quieting
        # arithmetic by the source word. On all non-NaNs x*1 is exactly x.
        for data_slot in (1, 2):
            data, product = op.inputs[data_slot], op.inputs[3 - data_slot]
            unordered_selects_product = p in (range(1, 8) if data_slot == 1 else range(8, 15))
            if not unordered_selects_product or not product_is_identity(product, data):
                continue
            if not any(
                predicate.inputs[i] is data and _constant_of(predicate.inputs[1 - i]) is not None for i in (0, 1)
            ):
                continue
            Rewriter.replace_op(op, [], [product])
            count += 1
            break
    if count:
        from merlin.llvmlower.entry_weight_projection import _pure_dce

        for function in module.walk():
            if function.name == "func.func":
                _pure_dce(function)
        module.verify()
    return count


def fold_zero_sign_selectors(module):
    """select(x == 0, copysign(0,x), x) preserves every IEEE storage word."""
    count = 0
    for op in list(module.walk()):
        if op.name != "arith.select" or str(op.results[0].type) not in FORMAT:
            continue
        predicate = op.operands[0].owner
        if getattr(predicate, "name", None) != "arith.cmpf" or predicate.fastmath.data:
            continue
        if predicate.predicate.value.data not in {1, 13}:
            continue
        zero_slot = 1 if predicate.predicate.value.data == 1 else 2
        data = op.operands[3 - zero_slot]
        if not any(predicate.operands[i] is data and input_literal(predicate.operands[1 - i]) == 0 for i in (0, 1)):
            continue
        cast = op.operands[zero_slot].owner
        if getattr(cast, "name", None) != "arith.bitcast":
            continue
        mask = cast.operands[0].owner
        if getattr(mask, "name", None) != "arith.andi":
            continue
        width = sum(FORMAT[str(data.type)][:2]) + 1
        for slot in (0, 1):
            raw = mask.operands[slot].owner
            bits = input_literal(mask.operands[1 - slot])
            if (
                getattr(raw, "name", None) == "arith.bitcast"
                and raw.operands[0] is data
                and raw.results[0].type == mask.results[0].type
                and getattr(getattr(raw.results[0].type, "width", None), "data", None) == width
                and isinstance(bits, int)
                and bits % (1 << width) == 1 << (width - 1)
            ):
                block = op.parent_block()
                Rewriter.replace_op(op, [], [data])
                # Only remove the proved, now-dead scalar wiring. Unknown
                # effects and other region operations retain their identity.
                for dead in (cast, mask, raw, predicate):
                    if all(not value.uses for value in dead.results):
                        Rewriter.erase_op(dead)
                for dead in list(block.ops):
                    if isinstance(dead, arith.ConstantOp) and all(not value.uses for value in dead.results):
                        Rewriter.erase_op(dead)
                count += 1
                break
    return count


def fold_nan_guarded_neutral(module):
    """Erase neutral IEEE operations only behind a complete NaN-word guard.

    The primitive and its input agree bitwise outside NaNs and have the same NaN
    class. Every use must inspect only that class or select a separate NaN word.
    An escaping raw result, payload bitcast or relaxed operation refuses folding.
    """

    def classification(op, value):
        return (
            getattr(op, "name", None) == "arith.cmpf"
            and op.operands[0] is value
            and op.operands[1] is value
            and op.predicate.value.data in {7, 14}
            and not op.fastmath.data
        )

    changed = 0
    for op in list(module.walk()):
        if op.name not in {"arith.maximumf", "arith.minimumf", "arith.mulf", "arith.divf"}:
            continue
        if op.fastmath.data:
            continue
        replacement = None
        for slot in (0,) if op.name == "arith.divf" else (0, 1):
            value = input_literal(op.operands[1 - slot])
            if (
                op.name in {"arith.mulf", "arith.divf"}
                and value == 1
                or op.name == "arith.maximumf"
                and value == -math.inf
                or op.name == "arith.minimumf"
                and value == math.inf
            ):
                replacement = op.operands[slot]
                break
        if replacement is None:
            continue
        result = op.results[0]
        for use in result.uses:
            consumer = use.operation
            if classification(consumer, result):
                continue
            if consumer.name != "arith.select" or use.index == 0:
                break
            condition = consumer.operands[0].owner
            if not classification(condition, result):
                break
            nonnan_slot = 2 if condition.predicate.value.data == 14 else 1
            if use.index != nonnan_slot or consumer.operands[3 - nonnan_slot] is result:
                break
        else:
            Rewriter.replace_op(op, [], [replacement])
            changed += 1
    return changed
