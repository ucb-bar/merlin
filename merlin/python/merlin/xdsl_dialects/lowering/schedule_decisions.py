"""``contract`` facts -> ``schedule`` decisions (merlin-apply-kernel-policies stage).

Reads the contract IR (reusable-weight facts whose required feature is declared by the
attached capability and whose checks carry proofs) and records the residency decisions:
hoist the pack, preserve the packed layout, place the weight resident against the
capability, keep accumulators live until commit, group everything into one command
buffer, and select the resident_packed_tensor interface (software_visible variant).
"""
from __future__ import annotations

from .._common import HAS_XDSL, Visibility
from .input_workload import find_matmuls, matmul_lhs_rhs


def _facts(mod):
    """{weight SSA value: (FactOp, CapabilityOp)} for proven reusable weights."""
    from .. import contract as c

    caps = [op for op in mod.walk() if isinstance(op, c.CapabilityOp)]
    cap = caps[0] if caps else None
    # A value counts as proven only when EVERY contract.check on it carries a proof.
    checks: dict = {}
    for op in mod.walk():
        if isinstance(op, c.CheckOp):
            checks.setdefault(op.value, []).append(bool(op.proofs))
    proven = {v for v, oks in checks.items() if all(oks)}
    out = {}
    for op in mod.walk():
        if (isinstance(op, c.FactOp) and op.role is not None
                and op.role.data == c.MemoryRole.REUSABLE_WEIGHT
                and op.value in proven):
            out[op.value] = (op, cap)
    return out


def lower_to_schedule(module):
    """Insert schedule decisions into a clone of the contract module."""
    if not HAS_XDSL:
        return module
    from xdsl.dialects.builtin import StringAttr

    from .. import contract as c
    from .. import schedule as s

    mod = module.clone()
    weights = _facts(mod)
    if not weights:
        return mod
    matmuls = find_matmuls(mod)

    for w, (fact, cap) in weights.items():
        block = fact.parent_block()
        # Insert decisions right after the last contract op in the block.
        anchor = fact
        for op in block.ops:
            if op.dialect_name() == c.DIALECT_NAME:
                anchor = op
        users = [mm for mm in matmuls if matmul_lhs_rhs(mm)[1] is w]
        cap_operands = [cap.cap] if cap is not None else []
        decisions = [
            s.HoistPackOp(operands=[w], properties={
                "outside": StringAttr("@loop_i"),
                "layout": StringAttr("packed_rhs")}),
            s.PreserveLayoutOp(operands=[w], properties={
                "layout": StringAttr("packed_rhs"),
                "scope": StringAttr("region")}),
            s.PlaceOp(operands=[w, cap_operands], properties={
                "state": s.MemoryStateAttr(s.MemoryState.RESIDENT),
                "lifetime": StringAttr("region")}),
            s.SelectInterfaceOp(operands=[w], properties={
                "interface": StringAttr("resident_packed_tensor"),
                "reason": StringAttr("reuse_count >= 2 and capacity_fit proven"),
                "visibility": s.VisibilityAttr(Visibility.SOFTWARE_VISIBLE)}),
        ]
        for op in reversed(decisions):
            block.insert_op_after(op, anchor)
        for mm in users:
            block.insert_op_after(
                s.KeepAccumulatorLiveOp(operands=[mm.results[0]], properties={
                    "until": StringAttr("epilogue_commit")}),
                mm)
        if users:
            # References matmul results, so it must come after them (dominance).
            block.insert_op_before(
                s.GroupDispatchOp(
                    operands=[[mm.results[0] for mm in users]],
                    properties={"granularity": s.DispatchGranularityAttr(
                        s.DispatchGranularity.COMMAND_BUFFER)}),
                block.last_op)
    return mod
