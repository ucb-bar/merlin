"""Pure matcher and bound helpers of the ``roundeven -> clamp -> fptosi`` quantize chain.

Split out of :mod:`merlin.llvmlower.quant_round` so independent checks (``quant_round_check``) can
read the chain without importing the rewrite's feature registry. Behaviour is unchanged: the
rewrite re-exports these names.
"""

from __future__ import annotations

import math


#: Largest magnitude a clamp bound may have for the inline round to be exact. At or above 2^23 an f32
#: has no fractional bits left, so `c - trunc(c)` stops being exact and `t` stops being representable
#: in a narrow destination type. NOT a tuning number and NOT a target fact -- it is where f32's
#: mantissa ends, and it is derived from the float type of the value being rewritten rather than
#: written down, so an f16 or f64 chain gets its own bound.
def _integral_limit(float_type) -> float:
    """The magnitude at which ``float_type`` runs out of fractional bits (2 ** mantissa bits)."""
    bits = {16: 11, 32: 24, 64: 53}.get(_float_width(float_type))
    if bits is None:
        return 0.0
    return float(2 ** (bits - 1))


def _float_width(t) -> int | None:
    from xdsl.dialects.builtin import Float16Type, Float32Type, Float64Type

    if isinstance(t, Float16Type):
        return 16
    if isinstance(t, Float32Type):
        return 32
    if isinstance(t, Float64Type):
        return 64
    return None


def _const_float(value, remaining: int = 5) -> float | None:
    """Read a float constant through uniform splats and their generic input arguments."""
    from xdsl.dialects.builtin import FloatAttr

    if remaining <= 0:
        return None
    owner = getattr(value, "owner", None)
    name = getattr(owner, "name", None)
    if name == "arith.constant":
        attr = owner.properties.get("value", owner.attributes.get("value"))
        return float(attr.value.data) if isinstance(attr, FloatAttr) else None
    if name == "tensor.splat":
        return _const_float(owner.operands[0], remaining - 1)
    region = getattr(owner, "parent", None)
    consumer = getattr(region, "parent", None)
    index = getattr(value, "index", None)
    if index is not None and getattr(consumer, "name", None) == "linalg.generic" and index < len(consumer.inputs):
        source = consumer.inputs[index]
        if getattr(getattr(source, "owner", None), "name", None) == "tensor.splat":
            return _const_float(source, remaining - 1)
    return None


def _single_use(value) -> bool:
    """``value`` feeds exactly one op. Rewriting a chain whose intermediate is read elsewhere would
    leave the original `math.roundeven` alive AND add the inline round, which is strictly worse than
    doing nothing -- so a shared intermediate is refused rather than duplicated."""
    return len(list(value.uses)) == 1


def _constant_zero(value, remaining: int = 5) -> bool:
    """Prove a zero through casts and uniform splats, including generic input arguments."""
    from xdsl.dialects.builtin import FloatAttr, IntegerAttr

    if remaining <= 0:
        return False
    owner = getattr(value, "owner", None)
    name = getattr(owner, "name", None)
    if name == "arith.constant":
        attr = owner.properties.get("value", owner.attributes.get("value"))
        return isinstance(attr, (FloatAttr, IntegerAttr)) and attr.value.data == 0
    if name in ("arith.sitofp", "tensor.splat"):
        return len(owner.operands) == 1 and _constant_zero(owner.operands[0], remaining - 1)
    region = getattr(owner, "parent", None)
    consumer = getattr(region, "parent", None)
    index = getattr(value, "index", None)
    if index is not None and getattr(consumer, "name", None) == "linalg.generic" and index < len(consumer.inputs):
        source = consumer.inputs[index]
        if getattr(getattr(source, "owner", None), "name", None) == "tensor.splat":
            return _constant_zero(source, remaining - 1)
    return False


def _guarded_scale_value(value):
    """Return the scalar yielded by a scale tensor feeding this generic block argument.

    Dynamic quantization computes the scale in one ``linalg.generic`` and consumes that tensor in
    another.  The divisor visible beside ``roundeven`` is consequently a block argument, not the
    guarding ``arith.select`` itself.  Follow exactly that one producer edge; do not guess through
    arbitrary tensor computations.
    """
    block = getattr(value, "owner", None)
    index = getattr(value, "index", None)
    region = getattr(block, "parent", None)
    consumer = getattr(region, "parent", None)
    if index is None or getattr(consumer, "name", None) != "linalg.generic" or index >= len(consumer.inputs):
        return None
    producer = getattr(consumer.inputs[index], "owner", None)
    if getattr(producer, "name", None) != "linalg.generic" or not producer.body.blocks:
        return None
    body = producer.body.blocks[0]
    terminator = body.last_op
    if getattr(terminator, "name", None) != "linalg.yield" or len(terminator.operands) != 1:
        return None
    return terminator.operands[0]


def _is_zero_guarded_scale(value) -> bool:
    """Whether ``value`` comes from a proved-positive symmetric scale constructor."""
    selected = _guarded_scale_value(value)
    select = getattr(selected, "owner", None)
    if getattr(select, "name", None) != "arith.select":
        return False
    cond, when_zero, otherwise = select.operands
    fallback = _const_float(when_zero)
    false_owner = getattr(otherwise, "owner", None)
    cmp = getattr(cond, "owner", None)
    if fallback is None or math.isnan(fallback) or fallback <= 0.0 or getattr(cmp, "name", None) != "arith.cmpf":
        return False
    # Most constructors select the raw division. i-GELU retains its pre-existing positive epsilon
    # floor for tiny nonzero rows and selects one only at exactly zero. Accept that exact spelling
    # too; a zero/negative/nonconstant floor proves nothing.
    if getattr(false_owner, "name", None) == "arith.maximumf":
        a, b = false_owner.operands
        floor, otherwise = _const_float(b), a
        if floor is None:
            floor, otherwise = _const_float(a), b
        if floor is None or math.isnan(floor) or floor <= 0.0:
            return False
        false_owner = getattr(otherwise, "owner", None)
    div = false_owner
    if getattr(div, "name", None) != "arith.divf":
        return False
    # xDSL encodes ordered-equal (`oeq`) as predicate 1.  Requiring it matters: unordered equality
    # would select the fallback for NaN too and no longer describe this constructor.
    predicate = cmp.properties.get("predicate")
    if getattr(getattr(predicate, "value", None), "data", None) != 1:
        return False
    numerator, denominator = div.operands
    denom = _const_float(denominator)
    if denom is None or math.isnan(denom) or denom == 0.0:
        return False
    lhs, rhs = cmp.operands
    return (lhs is numerator and _const_float(rhs) == 0.0) or (rhs is numerator and _const_float(lhs) == 0.0)


def _dynamic_scale_of_quant_input(value):
    """Return the scale in ``x / scale`` or TorchAO's ``x * (1 / scale)`` spelling."""
    owner = getattr(value, "owner", None)
    if getattr(owner, "name", None) == "arith.divf":
        return owner.operands[1]
    if getattr(owner, "name", None) != "arith.mulf":
        return None
    for candidate in owner.operands:
        reciprocal = getattr(candidate, "owner", None)
        if getattr(reciprocal, "name", None) != "arith.divf":
            # The efficient TorchAO form computes the reciprocal in a separate linalg.generic.
            # Follow that one producer edge, then inspect its yielded scalar expression.
            yielded = _guarded_scale_value(candidate)
            reciprocal = getattr(yielded, "owner", None)
        if getattr(reciprocal, "name", None) != "arith.divf":
            continue
        numerator, scale = reciprocal.operands
        if _const_float(numerator) == 1.0:
            return scale
    return None


def _int_range(int_type) -> tuple[int, int]:
    w = int_type.width.data
    return (-(2 ** (w - 1)), 2 ** (w - 1) - 1)


def _match_chain(fptosi_op):
    """``(round_op, clamp_ops, lo, hi)`` for a fusable chain ending at ``fptosi_op``, else a reason.

    Walks BACKWARDS from the convert through a run of `arith.minimumf` / `arith.maximumf` whose other
    operand is an integral float constant, and requires the run to bottom out at `math.roundeven`.
    A single-use addition of a constant signed zero is also traversed. It preserves all defined
    integer results of this chain; the possible zero-sign change is erased by `fptosi`.
    The interval is accumulated in the order the ops apply, so a chain that never bounds one side is
    caught rather than assumed.

    Returns either a 4-tuple or a string naming the refusal.
    """
    from xdsl.dialects.builtin import IntegerType

    res_t = fptosi_op.results[0].type
    if not isinstance(res_t, IntegerType) or res_t.width.data < 2:
        return "dest_not_multibit_int"
    imin, imax = _int_range(res_t)

    cur = fptosi_op.operands[0]
    ftype = cur.type
    if _float_width(ftype) is None:
        return "src_not_float"
    limit = _integral_limit(ftype)
    if limit <= 0.0:
        return "float_type_unhandled"

    lo, hi = float("-inf"), float("inf")
    clamps: list[tuple[str, float, object]] = []
    # Bounded walk: the chain is a handful of ops, and an unbounded loop over a malformed IR graph
    # would hang the build rather than fail it.
    for _ in range(8):
        owner = getattr(cur, "owner", None)
        name = getattr(owner, "name", None)
        if name == "math.roundeven":
            if not _single_use(cur):
                return "round_result_shared"
            # A dynamic quantization scale may be zero.  The real whole-model form is
            # ``roundeven(x / (amax / 127))``; for an all-zero activation ``amax == 0`` and the
            # zero lanes evaluate ``0 / 0``.  Both the old and new LLVM IR then contain
            # ``fptosi(NaN)``, which is poison, but the old scalar libm call happens to inhibit
            # vectorization while the inline form lets LLVM exploit that poison.  On K1 the latter
            # made TinyLlama's complete output canonical NaNs.  Preserve the observed program until
            # Scale construction defines the zero-amax case with a positive floor. Only rewrite
            # when the divisor is a finite nonzero constant or comes from that exact producer.
            rounded = owner.operands[0]
            divisor_value = _dynamic_scale_of_quant_input(rounded)
            if divisor_value is not None:
                divisor = _const_float(divisor_value)
                if (divisor is None or math.isnan(divisor) or divisor == 0.0) and not _is_zero_guarded_scale(
                    divisor_value
                ):
                    return "round_divisor_may_be_zero"
            if lo == float("-inf") or hi == float("inf"):
                # A one-sided clamp leaves the rounded value unbounded on the other side, so the
                # inline round has no range to be exact in. Refuse; do not guess a bound.
                return "clamp_not_two_sided"
            if lo > hi:
                return "clamp_empty"
            if not (abs(lo) < limit and abs(hi) < limit):
                return "clamp_bound_above_mantissa"
            if not (imin <= lo <= imax and imin <= hi <= imax):
                return "clamp_bound_outside_dest_int"
            return (owner, clamps, lo, hi)
        if name not in ("arith.minimumf", "arith.maximumf", "arith.addf"):
            return "chain_root_not_roundeven"
        if not _single_use(cur):
            return "clamp_result_shared"
        a, b = owner.operands
        if name == "arith.addf":
            nxt = b if _constant_zero(a) else a if _constant_zero(b) else None
            if nxt is None:
                return "chain_addend_not_zero"
            clamps.append((name, 0.0, owner))
            cur = nxt
            continue
        cval, nxt = _const_float(b), a
        if cval is None:
            cval, nxt = _const_float(a), b
        if cval is None:
            return "clamp_bound_not_constant"
        if cval != int(cval) if cval == cval and abs(cval) != float("inf") else True:
            # NaN / inf / non-integral bound: Lemma 1 does not hold, so the clamp cannot be moved
            # in front of the rounding. (`cval != cval` is the NaN test; no `math` import needed.)
            return "clamp_bound_not_integral"
        if name == "arith.minimumf":
            hi = min(hi, cval)
        else:
            lo = max(lo, cval)
        clamps.append((name, cval, owner))
        cur = nxt
    return "chain_too_long"
