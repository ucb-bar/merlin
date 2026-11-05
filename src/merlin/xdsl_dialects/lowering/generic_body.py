"""Read what a ``linalg.generic`` body computes: its op names, its constants, a rectifier.

Structural recognizers over the single block of an operation's region, used by
:func:`.compute_groups.classify` to name a stage. Split out of :mod:`.compute_groups` to keep that
module under the size gate; the names stay importable from there. Nothing here names a target.
"""

from __future__ import annotations

from merlin.common import mlir_query as mq


def _body_ops(op) -> list[str]:
    if not op.regions or not op.regions[0].blocks:
        return []
    return [mq.op_name(inner) for inner in op.regions[0].blocks[0].ops]


#: MLIR ``arith`` comparison predicates (the dialect's own enum, not a target fact), as
#: ``predicate -> True`` when it holds for ``lhs > rhs`` and ``False`` when it holds for ``lhs < rhs``.
#: Only ordered float and signed integer predicates: an unordered one decides NaN differently.
_CMPF_ORDERED = {2: True, 3: True, 4: False, 5: False}  # ogt, oge, olt, ole
_CMPI_SIGNED = {4: True, 5: True, 2: False, 3: False}  # sgt, sge, slt, sle
_ZERO_CASTS = frozenset({"arith.sitofp", "arith.uitofp", "arith.extf", "arith.truncf", "arith.extsi", "arith.trunci"})


def _is_zero(value) -> bool:
    """A value that is the constant zero, possibly converted from one."""
    for _ in range(4):
        owner = getattr(value, "owner", None)
        if owner is None or not hasattr(owner, "operands"):
            return False
        name = mq.op_name(owner)
        if name == "arith.constant":
            attribute = owner.properties.get("value")
            if attribute is None:
                attribute = owner.attributes.get("value")
            data = getattr(getattr(attribute, "value", None), "data", None)
            return isinstance(data, (int, float)) and data == 0
        if name not in _ZERO_CASTS or len(owner.operands) != 1:
            return False
        value = owner.operands[0]
    return False


def _is_rectifier(op) -> bool:
    """A one-input body that computes ``max(x, 0)`` as a compare against zero and a select.

    Recognized by exact structure: the select's condition compares the block argument with a zero,
    and its arms are that argument and that zero in the order that makes the result the larger one.
    """
    if not op.regions or not op.regions[0].blocks:
        return False
    block = op.regions[0].blocks[0]
    if not block.args:
        return False
    x = block.args[0]
    ops = list(block.ops)
    if not ops or mq.op_name(ops[-1]) != "linalg.yield" or len(ops[-1].operands) != 1:
        return False
    pick = getattr(ops[-1].operands[0], "owner", None)
    if pick is None or mq.op_name(pick) != "arith.select" or len(pick.operands) != 3:
        return False
    condition, when_true, when_false = pick.operands
    compare = getattr(condition, "owner", None)
    if compare is None or not hasattr(compare, "operands") or len(compare.operands) != 2:
        return False
    table = {"arith.cmpf": _CMPF_ORDERED, "arith.cmpi": _CMPI_SIGNED}.get(mq.op_name(compare))
    if table is None:
        return False
    raw = compare.properties.get("predicate") or compare.attributes.get("predicate")
    code = getattr(getattr(raw, "value", None), "data", None)
    if code not in table:
        return False
    lhs, rhs = compare.operands
    if lhs is x and _is_zero(rhs):
        x_greater = table[code]  # cmp(x, 0) holds when x is the greater side
    elif rhs is x and _is_zero(lhs):
        x_greater = not table[code]
    else:
        return False
    kept, cleared = (when_true, when_false) if x_greater else (when_false, when_true)
    return kept is x and _is_zero(cleared)


def _body_constants(op) -> list[float]:
    values: list[float] = []
    if not op.regions or not op.regions[0].blocks:
        return values
    for inner in op.regions[0].blocks[0].ops:
        if mq.op_name(inner) != "arith.constant":
            continue
        attribute = inner.properties.get("value")
        if attribute is None:  # not `or`: a zero-valued attribute is falsy
            attribute = inner.attributes.get("value")
        raw = getattr(attribute, "value", None)
        data = getattr(raw, "data", None)
        if isinstance(data, (int, float)):
            values.append(float(data))
    return values
