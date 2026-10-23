"""Exact evaluation of scalar IR on closed inputs, and closed value intervals.

Used by source specializations that must take their meaning from the IR
itself: a literal exponent, a literal divisor, a mask compared against a
literal over a provably bounded value. Every unsupported operation, runtime
leaf or non-finite intermediate is UNKNOWN, so callers fail closed.
"""

import math
from fractions import Fraction

from merlin.targetgen.native_payload_carriers import op_name as _name

UNKNOWN = object()
FLOAT_BITS = {"f16": (16, False), "bf16": (16, True), "f32": (32, False), "f64": (64, False)}


def _element_type(value):
    typ = value.type
    return getattr(typ, "element_type", typ)


def literal(value):
    """A closed uniform literal behind generic block arguments and splats, else None."""
    from xdsl.dialects.builtin import DenseIntOrFPElementsAttr

    seen = set()
    while value not in seen:
        seen.add(value)
        owner = value.owner
        name = _name(owner)
        if name == "arith.constant":
            attr = owner.properties.get("value", owner.attributes.get("value"))
            if hasattr(attr, "value") and hasattr(attr.value, "data"):
                data = attr.value.data
                return bool(data) if str(_element_type(value)) == "i1" and isinstance(data, int) else data
            if isinstance(attr, DenseIntOrFPElementsAttr):
                values = list(attr.get_values())
                if values and all(str(v) == str(values[0]) for v in values):
                    return values[0]
            return None
        if name == "tensor.splat":
            value = owner.operands[0]
            continue
        if name is None and hasattr(owner, "parent_op"):
            parent = owner.parent_op()
            if getattr(parent, "name", None) == "linalg.generic" and value.index < len(parent.inputs):
                value = parent.inputs[value.index]
                continue
        return None
    return None


def round_to(value, typ):
    """One IEEE RNE of a real value to a floating type; integers wrap to their width."""
    from xdsl.dialects.builtin import IntegerType

    from merlin.targetgen.exact_arithmetic import round_binary

    if isinstance(typ, IntegerType):
        width = typ.width.data
        integer = int(value) % (1 << width)
        if width == 1:
            return bool(integer)
        return integer - (1 << width) if integer >= 1 << (width - 1) else integer
    bits = FLOAT_BITS.get(str(typ))
    if bits is None:
        return UNKNOWN
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value) or value == 0.0):
        return value
    if value == 0:
        return 0.0
    try:
        return round_binary(Fraction(value), bits[0], bfloat16=bits[1], overflow="infinity")
    except (ValueError, OverflowError):
        return UNKNOWN


def _finite(*values):
    return all(isinstance(v, (int, float, bool)) and not (isinstance(v, float) and not math.isfinite(v)) for v in values)


def _zero_sign(value):
    return math.copysign(1.0, value) < 0 if isinstance(value, float) else False


def predicate_holds(predicate, relation, unordered=False, *, floating=False):
    """arith.cmpf (floating) or arith.cmpi predicate of a three-way order; unordered is a NaN operand."""
    eq, gt, lt = relation == 0, relation > 0, relation < 0
    if floating:
        ordered = not unordered
        return (
            False,
            ordered and eq,
            ordered and gt,
            ordered and not lt,
            ordered and lt,
            ordered and not gt,
            ordered and not eq,
            ordered,
            unordered or eq,
            unordered or gt,
            unordered or not lt,
            unordered or lt,
            unordered or not gt,
            unordered or not eq,
            unordered,
            True,
        )[predicate]
    return (eq, not eq, lt, not gt, gt, not lt, lt, not gt, gt, not lt)[predicate]


def evaluate(value, environment, memo=None):
    """Evaluate one scalar SSA value. environment maps values to Python numbers.

    A dependence chain deeper than the interpreter's recursion limit is
    UNKNOWN (closed proofs fail closed)."""
    try:
        return _evaluate(value, environment, memo)
    except RecursionError:
        return UNKNOWN


def _evaluate(value, environment, memo=None):
    from xdsl.dialects.builtin import IntegerType

    memo = {} if memo is None else memo
    if value in environment:
        return environment[value]
    if value in memo:
        return memo[value]
    known = literal(value)
    if known is not None:
        memo[value] = known
        return known
    owner = value.owner
    name = _name(owner)
    result_type = _element_type(value)

    def get(v):
        return _evaluate(v, environment, memo)

    out = UNKNOWN
    if name in {"arith.sitofp", "arith.uitofp", "arith.extui", "arith.extsi", "arith.trunci"}:
        x = get(owner.operands[0])
        if x is not UNKNOWN:
            source = _element_type(owner.operands[0])
            integer = int(x)
            if isinstance(source, IntegerType) and name != "arith.trunci":
                # Reinterpret the source word: unsigned, or two's complement (i1 true is -1).
                width = source.width.data
                integer %= 1 << width
                if name in {"arith.sitofp", "arith.extsi"} and integer >= 1 << (width - 1):
                    integer -= 1 << width
            out = round_to(integer, result_type)
    elif name in {"arith.extf", "arith.truncf"}:
        x = get(owner.operands[0])
        if x is not UNKNOWN:
            out = x if name == "arith.extf" else round_to(x, result_type)
    elif name == "math.absf":
        x = get(owner.operands[0])
        if x is not UNKNOWN and isinstance(x, float):
            out = abs(x)
    elif name == "arith.negf":
        x = get(owner.operands[0])
        if x is not UNKNOWN and isinstance(x, float):
            out = -x
    elif name in {"arith.mulf", "arith.addf", "arith.subf", "arith.divf"}:
        a, b = get(owner.operands[0]), get(owner.operands[1])
        if a is not UNKNOWN and b is not UNKNOWN and _finite(a, b):
            a, b = float(a), float(b)
            if name == "arith.mulf":
                exact = Fraction(a) * Fraction(b)
                sign = _zero_sign(a) != _zero_sign(b) if a == 0 or b == 0 else exact < 0
            elif name in {"arith.addf", "arith.subf"}:
                b = -b if name == "arith.subf" else b
                exact = Fraction(a) + Fraction(b)
                sign = _zero_sign(a) and _zero_sign(b)
            else:
                if b == 0:
                    exact = None
                else:
                    exact = Fraction(a) / Fraction(b)
                    sign = _zero_sign(a) != _zero_sign(b)
            if exact is not None:
                rounded = round_to(exact, result_type)
                if rounded is not UNKNOWN and rounded == 0:
                    rounded = -0.0 if sign else 0.0
                out = rounded
    elif name == "arith.remf":
        a, b = get(owner.operands[0]), get(owner.operands[1])
        if a is not UNKNOWN and b is not UNKNOWN and _finite(a, b) and float(b) != 0:
            out = math.fmod(float(a), float(b))  # IEEE fmod is exact; zero takes the dividend's sign.
    elif name in {"math.powf", "math.fpowi", "math.ipowi"}:
        base, exponent = get(owner.operands[0]), get(owner.operands[1])
        if exponent is not UNKNOWN and exponent == 0:
            out = round_to(1, result_type)
        elif base is not UNKNOWN and exponent is not UNKNOWN and _finite(base, exponent):
            if exponent > 0 and float(base) in (0.0, 1.0) and math.copysign(1.0, float(base)) > 0:
                out = round_to(float(base), result_type)  # 0^e = +0 and 1^e = 1 for every e > 0.
            elif float(exponent).is_integer() and exponent > 0:
                out = round_to(Fraction(base) ** int(exponent), result_type)
                if out == 0 and _zero_sign(float(base)) and int(exponent) % 2:
                    out = -0.0
    elif name in {"arith.cmpf", "arith.cmpi"}:
        a, b = get(owner.operands[0]), get(owner.operands[1])
        if a is not UNKNOWN and b is not UNKNOWN:
            predicate = owner.predicate.value.data
            if name == "arith.cmpf":
                unordered = (isinstance(a, float) and math.isnan(a)) or (isinstance(b, float) and math.isnan(b))
                relation = 0 if unordered else (a > b) - (a < b)
                out = predicate_holds(predicate, relation, unordered, floating=True)
            else:
                typ = _element_type(owner.operands[0])
                if isinstance(typ, IntegerType):
                    width = typ.width.data
                    ua, ub = int(a) % (1 << width), int(b) % (1 << width)
                    if predicate in {2, 3, 4, 5}:
                        ua = ua - (1 << width) if ua >= 1 << (width - 1) else ua
                        ub = ub - (1 << width) if ub >= 1 << (width - 1) else ub
                    out = predicate_holds(predicate, (ua > ub) - (ua < ub))
    elif name in {"arith.andi", "arith.ori", "arith.xori"}:
        a, b = get(owner.operands[0]), get(owner.operands[1])
        boolean = str(result_type) == "i1"
        if boolean and name == "arith.andi" and (a is False or b is False):
            out = False
        elif boolean and name == "arith.ori" and (a is True or b is True):
            out = True
        elif a is not UNKNOWN and b is not UNKNOWN:
            x, y = int(a), int(b)
            out = round_to(x & y if name == "arith.andi" else x | y if name == "arith.ori" else x ^ y, result_type)
    elif name == "arith.select":
        condition = get(owner.operands[0])
        if condition is not UNKNOWN:
            out = get(owner.operands[1] if condition else owner.operands[2])
    memo[value] = out
    return out


class Interval:
    """A closed real interval of the values a tensor or scalar can hold; None bounds mean empty."""

    def __init__(self, lo=None, hi=None):
        self.lo, self.hi = lo, hi

    @property
    def empty(self):
        return self.lo is None

    def union(self, other):
        if other is None:
            return None
        if self.empty:
            return other
        if other.empty:
            return self
        return Interval(min(self.lo, other.lo), max(self.hi, other.hi))

    def __eq__(self, other):
        return isinstance(other, Interval) and (self.lo, self.hi) == (other.lo, other.hi)


def interval(value, memo=None, active=None):
    """Closed interval of value, or None (also when the dependence chain is too deep)."""
    try:
        return _interval(value, memo, active)
    except RecursionError:
        return None


def _interval(value, memo=None, active=None):
    """Closed interval of every element value can take, or None when unbounded/unknown.

    tensor.empty contributes no value: its contents are unspecified until written,
    so a read of an unwritten element may be refined to any written value.
    """
    from xdsl.dialects.builtin import IntegerType

    memo = {} if memo is None else memo
    active = set() if active is None else active
    if value in memo:
        return memo[value]
    if value in active:
        return Interval()  # Loop-carried: the fixpoint below re-checks the assumption.
    known = literal(value)
    if known is not None and _finite(known):
        return Interval(Fraction(known), Fraction(known))
    owner = value.owner
    name = _name(owner)
    active.add(value)

    def get(v):
        return _interval(v, memo, active)

    out = None
    try:
        if name == "tensor.empty":
            out = Interval()
        elif name == "arith.andi":
            typ = _element_type(value)
            for mask_operand in owner.operands:
                mask = literal(mask_operand)
                if isinstance(typ, IntegerType) and isinstance(mask, int) and 0 <= mask < 1 << (typ.width.data - 1):
                    out = Interval(Fraction(0), Fraction(mask))
                    break
        elif name in {"arith.uitofp", "arith.sitofp"}:
            source = get(owner.operands[0])
            typ = _element_type(owner.operands[0])
            if source is not None and not source.empty and isinstance(typ, IntegerType):
                # The interval must lie inside the range of the interpreted source word (i1 true is -1 signed).
                width = typ.width.data
                lo, hi = (0, (1 << width) - 1) if name == "arith.uitofp" else (-(1 << (width - 1)), (1 << (width - 1)) - 1)
                source = source if lo <= source.lo and source.hi <= hi else None
            if source is not None:
                if source.empty:
                    out = source
                else:
                    lo, hi = round_to(source.lo, _element_type(value)), round_to(source.hi, _element_type(value))
                    if _finite(lo, hi):
                        out = Interval(Fraction(lo), Fraction(hi))
        elif name in {"arith.mulf", "arith.addf"}:
            for index in (0, 1):
                scale = literal(owner.operands[1 - index])
                source = get(owner.operands[index]) if scale is not None and _finite(scale) else None
                if source is None:
                    continue
                if source.empty:
                    out = source
                    break
                ends = (
                    [source.lo * Fraction(scale), source.hi * Fraction(scale)]
                    if name == "arith.mulf"
                    else [source.lo + Fraction(scale), source.hi + Fraction(scale)]
                )
                rounded = [round_to(v, _element_type(value)) for v in ends]
                if _finite(*rounded):
                    out = Interval(Fraction(min(rounded)), Fraction(max(rounded)))
                break
        elif name == "tensor.insert":
            scalar, destination = get(owner.operands[0]), get(owner.operands[1])
            out = None if scalar is None or destination is None else scalar.union(destination)
        elif name == "tensor.extract":
            out = get(owner.operands[0])
        elif name == "scf.for":
            index = value.index
            block = owner.body.block
            initial = get(owner.iter_args[index])
            carried = block.args[index + 1]
            out = initial
            for _ in range(8):
                if out is None:
                    break
                memo[carried] = out
                local = dict(memo)
                yielded = _interval(block.last_op.operands[index], local, set())
                widened = None if yielded is None else out.union(yielded)
                if widened == out:
                    break
                out = widened
            else:
                out = None
            memo.pop(carried, None)
        elif name == "scf.if":
            out = Interval()
            for region in owner.regions:
                branch = get(region.block.last_op.operands[value.index])
                out = None if branch is None or out is None else out.union(branch)
        elif name is None and hasattr(owner, "parent_op"):
            parent = owner.parent_op()
            if getattr(parent, "name", None) == "linalg.generic" and value.index < len(parent.inputs):
                out = get(parent.inputs[value.index])
        elif name == "linalg.generic":
            out = get(owner.body.block.last_op.operands[value.index])
    finally:
        active.discard(value)
    memo[value] = out
    return out


def fold_closed_generics(module):
    """Replace an elementwise generic whose inputs are all closed literals by a splat fill.

    The yielded scalar is evaluated exactly (typed IEEE rounding); only a finite
    closed result is folded, and index-dependent bodies are never folded, so no
    non-uniform table is ever produced. Returns the number of folds.
    """
    import math as _math

    from xdsl.dialects import arith, tensor
    from xdsl.dialects.builtin import FloatAttr, IntegerAttr, IntegerType

    count = 0
    for op in list(module.walk()):
        if _name(op) != "linalg.generic" or len(op.results) != 1:
            continue
        if any(it.data.value != "parallel" for it in op.get_iterator_types()):
            continue
        body = op.body.block
        if any(_name(child) == "linalg.index" for child in body.walk()):
            continue
        environment = {}
        closed = True
        for index, _ in enumerate(op.inputs):
            value = literal(body.args[index])
            if value is None:
                closed = False
                break
            environment[body.args[index]] = value
        if not closed:
            continue
        result = evaluate(body.last_op.operands[0], environment)
        typ = op.results[0].type
        element = typ.element_type
        if result is UNKNOWN or result is None:
            continue
        if isinstance(element, IntegerType):
            attr = IntegerAttr(int(result), element)
        elif str(element) in FLOAT_BITS and isinstance(result, (int, float)) and _math.isfinite(float(result)):
            attr = FloatAttr(float(result), element)
        else:
            continue
        constant = arith.ConstantOp(attr)
        filled = tensor.SplatOp(constant.result, [], typ)
        parent = op.parent
        parent.insert_ops_before([constant, filled], op)
        op.results[0].replace_all_uses_with(filled.result)
        parent.erase_op(op)
        count += 1
    return count


def _elementwise_map(op):
    """(input, yielded) of a one-input, one-result, identity-map, all-parallel generic, else None."""
    from xdsl.dialects.builtin import AffineMapAttr
    from xdsl.ir.affine import AffineMap

    if _name(op) != "linalg.generic" or len(op.inputs) != 1 or len(op.results) != 1:
        return None
    if any(it.data.value != "parallel" for it in op.get_iterator_types()):
        return None
    rank = len(op.results[0].type.get_shape())
    if any(m != AffineMapAttr(AffineMap.identity(rank)) for m in op.get_indexing_maps()):
        return None
    return op.inputs[0], op.body.block.last_op.operands[0]


def _elementwise_round_trip(op):
    """A narrowing generic whose input is the widening of a same-format tensor
    (through elementwise copies): the tensor it reproduces, else None."""
    found = _elementwise_map(op)
    if found is None:
        return None
    tensor, yielded = found
    body = op.body.block
    if _name(yielded.owner) != "arith.truncf" or yielded.owner.operands[0] is not body.args[0]:
        return None
    while True:
        producer = _elementwise_map(tensor.owner) if hasattr(tensor.owner, "results") else None
        if producer is None:
            return None
        inner, value = producer
        block = tensor.owner.body.block
        if value is block.args[0]:
            tensor = inner  # an elementwise copy
            continue
        if _name(value.owner) == "arith.extf" and value.owner.operands[0] is block.args[0]:
            return inner if inner.type == op.results[0].type else None
        return None


def fold_unit_scales(module):
    """Replace exact identities by their operand: x * 1, x / 1 (a closed literal
    exactly +1.0) and the widen-then-narrow round trip truncf(extf(x)) back to
    x's own format.

    Each returns its operand's bits for every finite value, infinity, signed
    zero and quiet NaN; the result is movement. Returns the number of folds.
    """
    count = 0
    for op in list(module.walk()):
        name = _name(op)
        if name == "arith.truncf" and _name(op.operands[0].owner) == "arith.extf":
            source = op.operands[0].owner.operands[0]
            if source.type == op.results[0].type:
                op.results[0].replace_all_uses_with(source)
                op.parent.erase_op(op)
                count += 1
            continue
        if name == "linalg.generic":
            source = _elementwise_round_trip(op)
            if source is not None:
                op.results[0].replace_all_uses_with(source)
                count += 1
            continue
        if name not in {"arith.mulf", "arith.divf"} or len(op.results) != 1:
            continue
        candidates = [(1, 0)] if name == "arith.divf" else [(1, 0), (0, 1)]
        for scale, data in candidates:
            value = literal(op.operands[scale])
            if isinstance(value, float) and value == 1.0 and op.operands[data].type == op.results[0].type:
                op.results[0].replace_all_uses_with(op.operands[data])
                op.parent.erase_op(op)
                count += 1
                break
    return count

