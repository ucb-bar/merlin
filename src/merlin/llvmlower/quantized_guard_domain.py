"""Close guarded affine sources over declared quantized code domains.

This is exact-rational quantized semantics, not a native IEEE reassociation
permission. Every floating escape must be a quantizer. Finite bounds discharge
IEEE classification guards; unresolved guards require equal rational branches.
"""

import hashlib
import math
from fractions import Fraction


class Unproved(ValueError):
    pass


def _literal(value):
    from xdsl.dialects.builtin import FloatAttr, IntegerAttr

    owner = value.owner
    if getattr(owner, "name", None) in {"tensor.splat", "linalg.fill"}:
        return _literal(owner.operands[0])
    if getattr(owner, "name", None) != "arith.constant":
        raise Unproved("not a uniform source literal")
    attr = owner.properties.get("value", owner.attributes.get("value"))
    if isinstance(attr, (FloatAttr, IntegerAttr)):
        result = attr.value.data
    elif hasattr(attr, "get_values"):
        values = list(attr.get_values())
        if not values or any(v != values[0] for v in values):
            raise Unproved("nonuniform literal")
        result = values[0]
    else:
        raise Unproved("unknown literal")
    if not math.isfinite(result):
        raise Unproved("nonfinite literal")
    return Fraction(result)


def _domain(value):
    from merlin.common.mlir_query import op_name

    owner = value.owner
    try:
        literal = _literal(value)
        return (literal, literal), dict(literal=str(literal))
    except Unproved:
        pass
    # Propagate a finite enclosure through immutable carriers and closed sums.
    # No tensor payload, eager result or source operator name participates.
    name = op_name(owner)
    if name in {"tensor.expand_shape", "tensor.collapse_shape", "tensor.cast"}:
        interval, record = _domain(owner.operands[0])
        return interval, dict(carrier=name, source=record)
    if name == "linalg.generic":
        proof = prove_affine(owner)
        return tuple(Fraction(v) for v in proof["finite_interval"]), dict(affine=proof)
    if name == "linalg.reduce" and len(owner.operands) == 2 and len(owner.results) == 1:
        block = owner.regions[0].block
        body = list(block.ops)
        if (
            len(body) != 2
            or body[0].name != "arith.addf"
            or body[1].name != "linalg.yield"
            or tuple(body[0].operands) not in (tuple(block.args), tuple(reversed(block.args)))
            or tuple(body[1].operands) != tuple(body[0].results)
            or str(body[0].results[0].type) != "f32"
        ):
            raise Unproved("not a closed f32 additive reduction")
        flag = body[0].properties.get("fastmath", body[0].attributes.get("fastmath"))
        if flag is not None and str(flag).split("<")[-1] != "none>":
            raise Unproved("relaxed reduction arithmetic")
        shape = owner.operands[0].type.get_shape()
        dims = tuple(owner.properties["dimensions"].get_values())
        if len(set(dims)) != len(dims) or any(d < 0 or d >= len(shape) for d in dims) or any(d < 0 for d in shape):
            raise Unproved("reduction has no closed static extent")
        count = math.prod(shape[d] for d in dims)
        interval, source = _domain(owner.operands[0])
        seed, initial = _domain(owner.operands[1])
        # Standard summation error enclosure, using a conservative relative
        # error 2**-23 plus one minimum subnormal per addition. This bounds
        # every partial sum for every reduction order, not just the final sum.
        relative = Fraction(count, 2**23)
        if relative >= 1:
            raise Unproved("reduction rounding enclosure unavailable")
        radius = (max(map(abs, seed)) + count * max(map(abs, interval)) + count * _TINY) / (1 - relative)
        if radius > _MAX / 4:
            raise Unproved("finite reduction intermediate overflow proof unavailable")
        return (-radius, radius), dict(reduction="sum", count=count, source=source, seed=initial)
    if op_name(owner) != "quant_ext.dequantize_per_tensor":
        raise Unproved("operand has no declared quantized domain")

    def integer(key):
        attr = owner.attributes.get("prov.quant." + key)
        if attr is None:
            raise Unproved("missing quantized " + key)
        return int(attr.value.data)

    def rational(key):
        attr = owner.attributes.get("prov.quant." + key)
        if attr is None or len(attr) != 1:
            raise Unproved("requires scalar exact qparams")
        return int(attr.data[0].data)

    low, high, zero = (integer(k) for k in ("quant_min", "quant_max", "zero_point"))
    scale = Fraction(rational("scale_numerator"), rational("scale_denominator"))
    if _literal(owner.operands[1]) != scale or _literal(owner.operands[2]) != zero:
        raise Unproved("qparam metadata differs from executable operands")
    storage = getattr(owner.attributes.get("prov.quant.storage_dtype"), "data", None)
    limits = {"int8": (-128, 127), "uint8": (0, 255)}.get(storage)
    if limits is None or not limits[0] <= low < high <= limits[1] or not low <= zero <= high or scale <= 0:
        raise Unproved("invalid declared quantized domain")
    if (low, high) != limits:
        producer = owner.operands[0].owner
        if op_name(producer) != "quant_ext.quantize_per_tensor":
            raise Unproved("narrow code domain has no clamping producer")
        qlow = producer.properties.get("quant_min")
        qhigh = producer.properties.get("quant_max")
        if qlow is None or qhigh is None or not low <= qlow.value.data < qhigh.value.data <= high:
            raise Unproved("producer does not establish the declared code domain")
    # Include the frontend's DQ rounding as well as the exact rational endpoints.
    interval = _rounded((scale * (low - zero), scale * (high - zero)))
    return interval, dict(storage=storage, quant_min=low, quant_max=high, zero_point=zero, scale=str(scale))


_MAX = Fraction((2**24 - 1) * 2**104)
_TINY = Fraction(1, 2**149)


def _rounded(interval):
    """Conservative outward f32 enclosure, including subnormals and cancellation."""
    low, high = interval
    radius = max(abs(low), abs(high))
    if radius > _MAX / 4:
        raise Unproved("finite intermediate overflow proof unavailable")
    # RNE absolute error <= max(|x|*2**-23, minimum subnormal).
    error = max(radius / 2**23, _TINY)
    return low - error, high + error


def prove_affine(op, *, conditional=False):
    """A source-bound polynomial and finite-domain guard certificate.

    Conditional mode only requests a quantized recapture. It cannot authorize a
    rewrite or admission: all finite intermediate obligations remain pending.
    """
    from xdsl.dialects.builtin import FloatAttr, IntegerAttr

    if op.name != "linalg.generic" or len(op.results) != 1 or str(op.results[0].type.element_type) != "f32":
        raise Unproved("not a single f32 structured map")
    if any(it.data.value != "parallel" for it in op.get_iterator_types()):
        raise Unproved("not an independent affine map")
    block = op.body.block
    if len(block.args) != len(op.inputs) + 1 or block.args[-1].uses:
        raise Unproved("reads destination initialization")
    # Pure scalar closure, including dead classification branches. No effect is dropped.
    permitted = {
        "arith.constant",
        "arith.sitofp",
        "arith.uitofp",
        "arith.extf",
        "arith.truncf",
        "arith.addf",
        "arith.subf",
        "arith.mulf",
        "arith.divf",
        "arith.negf",
        "math.fma",
        "arith.cmpf",
        "arith.cmpi",
        "arith.select",
        "arith.bitcast",
        "arith.ori",
        "arith.andi",
        "arith.addi",
        "arith.subi",
        "arith.muli",
        "linalg.index",
        "linalg.yield",
    }
    if any(child.name not in permitted for child in block.ops):
        raise Unproved("unknown scalar effects or primitive")
    for child in block.ops:
        for key in ("fastmath", "overflowFlags"):
            flag = child.properties.get(key, child.attributes.get(key))
            if flag is not None and str(flag).split("<")[-1] != "none>":
                raise Unproved("relaxed or promised arithmetic")
    n = len(op.inputs)
    values = {}
    domains = []
    for i, (arg, source) in enumerate(zip(block.args, op.inputs)):
        if str(arg.type) != "f32":
            raise Unproved("not a homogeneous f32 map")
        try:
            literal = _literal(source)
            poly = (Fraction(0),) * n + (literal,)
            interval, record = (literal, literal), dict(literal=str(literal))
        except Unproved:
            poly = tuple(Fraction(int(j == i)) for j in range(n)) + (Fraction(0),)
            interval, record = (
                (None, dict(obligation="finite declared qparams and all intermediates"))
                if conditional
                else _domain(source)
            )
        values[arg] = (poly, interval)
        domains.append(record)
    removed = [0, 0]

    def bound(interval):
        return None if interval is None and conditional else _rounded(interval)

    def add(a, b, sign=1):
        p, ia = a
        q, ib = b
        polynomial = tuple(x + sign * y for x, y in zip(p, q))
        interval = (
            None
            if ia is None or ib is None
            else (ia[0] + (ib[0] if sign == 1 else -ib[1]), ia[1] + (ib[1] if sign == 1 else -ib[0]))
        )
        return polynomial, bound(interval)

    def mul(a, b):
        p, ia = a
        q, ib = b
        if any(p[:-1]) and any(q[:-1]):
            raise Unproved("runtime product is not affine")
        polynomial = tuple(x * q[-1] for x in p) if not any(q[:-1]) else tuple(x * p[-1] for x in q)
        if ia is None or ib is None:
            interval = None
        else:
            products = [x * y for x in ia for y in ib]
            interval = (min(products), max(products))
        return polynomial, bound(interval)

    def condition(value):
        owner = value.owner
        if owner.name == "arith.constant":
            return bool(owner.value.value.data)
        if owner.name == "arith.cmpf":
            predicate = owner.predicate.value.data
            if predicate in {7, 14}:  # ordered/unordered; every operand must be finite
                try:
                    for arg in owner.operands:
                        evaluate(arg)
                except Unproved:
                    return None
                return predicate == 7
        return None

    def evaluate(value):
        if value in values:
            return values[value]
        child = value.owner
        name = child.name
        if name == "arith.constant":
            attr = child.properties.get("value")
            if not isinstance(attr, (FloatAttr, IntegerAttr)) or not math.isfinite(attr.value.data):
                raise Unproved("nonfinite or unknown literal")
            number = Fraction(attr.value.data)
            result = ((Fraction(0),) * n + (number,), (number, number))
        elif name in {"arith.sitofp", "arith.uitofp", "arith.extf", "arith.truncf"}:
            original = evaluate(child.operands[0])
            if any(original[0][:-1]):
                raise Unproved("runtime precision conversion")
            from merlin.targetgen.exact_arithmetic import round_binary

            x = original[0][-1]
            if name == "arith.uitofp":
                x = int(x) % (1 << child.operands[0].type.width.data)
            if str(value.type) != "f32":
                raise Unproved("unsupported conversion boundary")
            x = Fraction(round_binary(x, 32))
            result = ((Fraction(0),) * n + (x,), (x, x))
        elif name == "arith.select":
            test = condition(child.operands[0])
            if test is not None:
                removed[0] += 1
                result = evaluate(child.operands[1 if test else 2])
            else:
                a, b = (evaluate(v) for v in child.operands[1:])
                if a[0] != b[0]:
                    raise Unproved("live guard chooses different rational results")
                removed[1] += 1
                interval = None if a[1] is None or b[1] is None else (min(a[1][0], b[1][0]), max(a[1][1], b[1][1]))
                result = (a[0], interval)
        elif name in {"arith.addf", "arith.subf"}:
            result = add(*(evaluate(v) for v in child.operands), sign=-1 if name == "arith.subf" else 1)
        elif name == "arith.mulf":
            result = mul(*(evaluate(v) for v in child.operands))
        elif name == "arith.divf":
            numerator = evaluate(child.operands[0])
            denominator, divisor_interval = evaluate(child.operands[1])
            if any(denominator[:-1]) or not denominator[-1] or divisor_interval != (denominator[-1], denominator[-1]):
                raise Unproved("division requires an exact nonzero literal denominator")
            divisor = denominator[-1]
            interval = None if numerator[1] is None else tuple(sorted(x / divisor for x in numerator[1]))
            result = (tuple(x / divisor for x in numerator[0]), bound(interval))
        elif name == "math.fma":
            a, b, c = (evaluate(v) for v in child.operands)
            result = add(mul(a, b), c)  # Conservative two-round interval encloses fused arithmetic too.
        elif name == "arith.negf":
            p, i = evaluate(child.operands[0])
            result = (tuple(-x for x in p), None if i is None else (-i[1], -i[0]))
        else:
            raise Unproved("unproved live scalar primitive: " + name)
        values[value] = result
        return result

    yielded = block.last_op
    if yielded.name != "linalg.yield" or len(yielded.operands) != 1:
        raise Unproved("unknown result frontier")
    polynomial, interval = evaluate(yielded.operands[0])
    if not conditional and interval is None:
        raise Unproved("missing finite intermediate proof")
    return dict(
        schema="merlin.quantized_affine_domain.v1",
        status="conditional" if conditional else "proven",
        source_sha256=hashlib.sha256(str(op).encode()).hexdigest(),
        domains=domains,
        finite_interval=None if interval is None else [str(v) for v in interval],
        coefficients=[str(v) for v in polynomial[:-1]],
        constant=str(polynomial[-1]),
        dead_classification_selects=removed[0],
        equal_rational_branch_selects=removed[1],
        arithmetic="exact_rational_then_single_final_rne_and_clamp",
        signed_zero="rational_zero_quantizes_to_zero_point_no_floating_escape",
    )


def canonicalize_quantized_affine(module, *, exact_rational=False):
    """Rewrite only proved closed Q/DQ maps, retaining source-bound certificates."""
    import json

    from xdsl.dialects import arith
    from xdsl.dialects.builtin import FloatAttr, StringAttr, f32
    from xdsl.ir import Block

    from merlin.common.mlir_query import op_name
    from merlin.llvmlower._ir import emitter

    if exact_rational is not True:
        return []
    proofs = []
    for ordinal, op in enumerate(list(module.walk())):
        if op.name != "linalg.generic" or not op.results:
            continue
        if not op.results[0].uses or any(
            op_name(use.operation) != "quant_ext.quantize_per_tensor" for use in op.results[0].uses
        ):
            continue
        # Leave already simple families alone; this pass closes guarded sources.
        if not any(child.name == "arith.select" for child in op.body.block.ops):
            continue
        try:
            proof = prove_affine(op)
        except (Unproved, ValueError, OverflowError, ZeroDivisionError):
            continue
        numbers = [Fraction(s) for s in [*proof["coefficients"], proof["constant"]]]
        if any(
            not math.isfinite(float(v))
            or Fraction(float(v)) != v
            or Fraction(f32.unpack(f32.pack([float(v)]), 1)[0]) != v
            for v in numbers
        ):
            continue
        body = Block(arg_types=[v.type for v in op.body.block.args])

        emit = emitter(body)

        result = None
        for i, coefficient in enumerate(numbers[:-1]):
            if not coefficient:
                continue
            term = body.args[i]
            if coefficient != 1:
                term = emit(arith.MulfOp(term, emit(arith.ConstantOp(FloatAttr(float(coefficient), f32)))))
            result = term if result is None else emit(arith.AddfOp(result, term))
        if numbers[-1] or result is None:
            literal = emit(arith.ConstantOp(FloatAttr(float(numbers[-1]), f32)))
            result = literal if result is None else emit(arith.AddfOp(result, literal))
        from xdsl.dialects.linalg.ops import YieldOp

        body.add_op(YieldOp(result))
        old = op.body.block
        op.body.erase_block(old, safe_erase=False)
        op.body.add_block(body)
        op.attributes["merlin.quantized_domain_proof"] = StringAttr(json.dumps(proof, sort_keys=True))
        proofs.append(dict(source_operation_ordinal=ordinal, **proof))
    module.verify()
    return proofs
