"""A two-tensor sum spelled bit-faithfully, recognized as the sum it is when that is provable.

A frontend that reproduces a CPU framework's add bit for bit does not emit one ``arith.addf``. It
emits ``math.fma(b, alpha, a)`` (``a + alpha*b`` with one rounding) for its vectorized lanes and
``arith.addf(arith.mulf(b, alpha), a)`` for its scalar tail, and then chains of ``arith.select`` that
pick a NaN payload: which operand's quieted NaN the result carries depends on which operand was NaN
and on the element's lane or tail position. Only the payload of a NaN result changes; each such
select's replacement is taken only when a value it tests is NaN.

So the body is the plain sum ``a + b`` when, with both inputs finite, its yielded value can be shown
to be that sum value by value:

* ``fma(x, 1, y)`` and ``addf(x, y)`` over the two inputs are the sum (one rounding of ``a + b``),
  and ``mulf(x, 1)`` is ``x`` exactly -- so ``alpha`` must be the constant one;
* ``select(cmpf uno(...), t, f)`` is ``f`` when every tested value is NaN-free (an input, a finite
  constant, or their sum -- finite, or an overflowed infinity, never NaN); any other select is known
  only when both its arms are the same known value (a lane/tail choice between two equal sums);
* the inputs are finite. That is not a property of the body: :func:`provable_sum` proves it from
  where the inputs come from (a dequantized integer at a finite scale is finite), and without it
  nothing here is claimed.

Anything else -- another alpha, a body op with an effect, a value this cannot follow -- is not
recognized, and the body stays what it was (fail closed).
"""

from __future__ import annotations

import math
from typing import Any

#: ``arith.cmpf`` predicate ``uno``: true iff either operand is NaN (the dialect's own enum).
_CMPF_UNORDERED = 14
#: Body ops with no effect: the payload selection is integer/bit arithmetic over the same values.
_PURE_PREFIXES = ("arith.",)
_PURE = frozenset({"math.fma", "linalg.index", "linalg.yield"})


def _float_constant(value: Any) -> float | None:
    """The compile-time float ``value`` is, through ``sitofp``/``uitofp`` of an integer constant."""
    owner = getattr(value, "owner", None)
    name = getattr(owner, "name", None)
    if name in ("arith.sitofp", "arith.uitofp"):
        owner = getattr(owner.operands[0], "owner", None)
        name = getattr(owner, "name", None)
    if name != "arith.constant":
        return None
    attr = owner.properties.get("value")
    data = getattr(getattr(attr, "value", None), "data", None)
    return float(data) if isinstance(data, (int, float)) and not isinstance(data, bool) else None


def is_finite_input_unit_sum(op) -> bool:
    """Whether ``op`` (a two-input ``linalg.generic``) computes ``in0 + in1`` for FINITE inputs.

    The finiteness of the inputs is the caller's to prove; see the module docstring.
    """
    if getattr(op, "name", None) != "linalg.generic" or not op.regions or len(op.regions[0].blocks) != 1:
        return False
    inputs = list(getattr(op, "inputs", ()))
    block = op.regions[0].blocks[0]
    if len(inputs) != 2 or len(block.args) < 2:
        return False
    operations = list(block.ops)
    if not operations or any(
        inner.name not in _PURE and not inner.name.startswith(_PURE_PREFIXES) for inner in operations
    ):
        return False
    yielded = operations[-1]
    if yielded.name != "linalg.yield" or len(yielded.operands) != 1:
        return False
    a, b = block.args[0], block.args[1]
    one, args, total = ("const", 1.0), {("arg", 0), ("arg", 1)}, ("sum",)
    known: dict[int, tuple | None] = {}

    def value_of(value) -> tuple | None:
        """What ``value`` is when both inputs are finite: an input, a finite constant, their sum, or
        ``None`` (not known). Every known value is NaN-free: inputs and constants are finite, and a
        sum or product of finite values is finite or an overflowed infinity."""
        if value is a or value is b:
            return ("arg", 0 if value is a else 1)
        if id(value) in known:
            return known[id(value)]
        owner = getattr(value, "owner", None)
        name = getattr(owner, "name", None)
        answer = None
        constant = _float_constant(value)
        if constant is not None:
            answer = ("const", constant) if math.isfinite(constant) else None
        elif name == "arith.select":
            condition, when_true, when_false = owner.operands
            test = getattr(condition, "owner", None)
            if (
                getattr(test, "name", None) == "arith.cmpf"
                and int(test.properties["predicate"].value.data) == _CMPF_UNORDERED
                and all(value_of(operand) is not None for operand in test.operands)
            ):
                answer = value_of(when_false)  # a NaN test of NaN-free values is false
            else:
                picked = value_of(when_false)
                answer = picked if picked is not None and value_of(when_true) == picked else None
        elif name == "arith.mulf":
            # Multiplying by exactly one is exact (for an infinity too).
            x, y = (value_of(operand) for operand in owner.operands)
            answer = x if y == one and x is not None else y if x == one and y is not None else None
        elif name == "arith.addf":
            answer = total if {value_of(operand) for operand in owner.operands} == args else None
        elif name == "math.fma":
            x, y, z = (value_of(operand) for operand in owner.operands)  # x * y + z, rounded once
            product = x if y == one else y if x == one else None
            answer = total if {product, z} == args else None
        known[id(value)] = answer
        return answer

    return value_of(yielded.operands[0]) == total


def provable_sum(op) -> bool:
    """Whether ``op`` is a bit-faithful two-tensor sum whose inputs are PROVABLY finite.

    Each input must be (through shape-only views) a dequantize of integers by a compile-time,
    finite scale: such a value is finite, so the body is :func:`is_finite_input_unit_sum`'s sum.
    """
    if not is_finite_input_unit_sum(op):
        return False
    from . import compute_groups as CG  # noqa: PLC0415 -- it imports this module
    from . import group_numerics as GN  # noqa: PLC0415 -- it imports compute_groups

    for value in list(op.inputs):
        owner = getattr(value, "owner", None)
        while getattr(owner, "name", None) in CG._VIEW_OPS and getattr(owner, "operands", None):
            owner = getattr(owner.operands[0], "owner", None)
        stage = CG.classify(owner) if hasattr(owner, "operands") else None
        if stage is None or stage.kind != CG.DEQUANTIZE:
            return False
        try:
            scale = GN._scale_source(owner).value
        except GN.GroupNumericsError:
            return False
        if scale is None or not math.isfinite(scale):
            return False
    return True
