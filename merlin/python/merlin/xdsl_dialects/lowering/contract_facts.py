"""linalg input -> ``contract`` facts (the merlin-infer-contract-facts +
merlin-attach-target-capabilities stages).

Analyzes the payload: finds matmul-family ops, detects a shared immutable RHS (a block
argument used by >= 2 matmuls), and inserts contract IR at the top of the function:
capability (from the target contract), assume/fact on the reused weight, the
resident_packed_tensor requirement, and proofs/checks for what is actually provable
(immutability from block-arg-ness; capacity from the contract's resident storage).
"""
from __future__ import annotations

from typing import Any

from .._common import HAS_XDSL
from .input_workload import find_matmuls, matmul_lhs_rhs

# Capability surface used when no explicit target contract is supplied; matches
# merlin/targets/toy_npu/contracts/target_contract.yaml.
DEFAULT_TARGET_CONTRACT: dict[str, Any] = {
    "name": "toy_npu",
    "features": ["resident_packed_tensor", "accumulator_commit", "command_buffer",
                 "metrics"],
    "runtime": {"backends": ["simulator", "zephyr"]},
    "capabilities": {"resident_storage_bytes": 131072},
}


def _trace_to_block_arg(val, block):
    """Walk up pure layout/view ops (transpose, reshape, collapse/expand, cast, copy) from
    ``val`` to its source block argument, if any. Real-model matmuls feed the weight in as
    ``A @ Wᵀ`` — the RHS owner is a ``linalg.transpose`` of a block-arg weight, not the
    block arg directly — so the immutable-weight detection must trace through it."""
    from xdsl.ir import BlockArgument

    VIEW = ("linalg.transpose", "tensor.collapse_shape", "tensor.expand_shape",
            "tensor.reshape", "tensor.cast", "linalg.copy")
    cur = val
    for _ in range(12):
        if isinstance(cur, BlockArgument):
            return cur if cur.owner is block else None
        owner = getattr(cur, "owner", None)
        if owner is not None and getattr(owner, "name", None) in VIEW and owner.operands:
            cur = owner.operands[0]
            continue
        return None
    return None


def _tensor_nbytes(t) -> int:
    from xdsl.dialects.builtin import IntegerType, TensorType

    if not isinstance(t, TensorType):
        return 0
    elems = 1
    for d in t.get_shape():
        elems *= d
    elem = t.element_type
    width = elem.width.data if isinstance(elem, IntegerType) else 32
    return elems * ((width + 7) // 8)


def lower_to_contract(module, target_contract: dict[str, Any] | None = None):
    """Insert contract facts/requirements/proofs into a clone of ``module``."""
    if not HAS_XDSL:
        return module
    from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, StringAttr

    from .. import contract as c

    tc = target_contract or DEFAULT_TARGET_CONTRACT
    mod = module.clone()
    matmuls = find_matmuls(mod)
    if not matmuls:
        return mod

    # Group matmuls by the underlying block-arg *weight* their RHS traces back to (through
    # the A@Wᵀ transpose / any layout view). The weight block arg is what we pack + prove
    # immutable; the transpose is a layout concern handled downstream.
    block = matmuls[0].parent_block()
    rhs_uses: dict = {}
    for mm in matmuls:
        _, rhs = matmul_lhs_rhs(mm)
        w = _trace_to_block_arg(rhs, block)
        if w is not None:
            rhs_uses.setdefault(w, []).append(mm)
    first_op = block.first_op

    cap = c.CapabilityOp(
        result_types=[c.CapabilityType(StringAttr(tc["name"]))],
        properties={
            "sym_name": StringAttr(tc["name"]),
            "features": ArrayAttr([StringAttr(f) for f in tc.get("features", [])]),
            "runtime": ArrayAttr([StringAttr(b) for b in
                                  tc.get("runtime", {}).get("backends", [])]),
        })
    block.insert_op_before(cap, first_op)

    storage = int(tc.get("capabilities", {}).get("resident_storage_bytes", 0))
    for rhs, users in sorted(rhs_uses.items(), key=lambda kv: len(kv[1]), reverse=True):
        # Resident-pack benefit comes from CROSS-DISPATCH reuse: the same immutable weight
        # used by >= 2 matmuls (so packing once amortizes over many uses). A single matmul
        # — even with many rows — is NOT a candidate (design_pressure: no_reuse -> none).
        # The block-arg test holds because `rhs` was traced back to its weight block arg.
        if len(users) < 2 or rhs.owner is not block:
            continue
        reuse = len(users)
        ops = [
            # Block arguments are not written inside the region: assumed immutable.
            c.AssumeOp(operands=[rhs], properties={
                "kind": StringAttr("immutable"),
                "lifetime": c.LifetimeAttr(c.Lifetime.WITHIN_REGION)}),
            c.FactOp(operands=[rhs], properties={
                "role": c.MemoryRoleAttr(c.MemoryRole.REUSABLE_WEIGHT),
                "reuse_count": IntegerAttr(reuse, 64),
                "layout": c.LayoutRoleAttr(c.LayoutRole.CANONICAL)}),
            c.RequireOp(properties={
                "feature": StringAttr("resident_packed_tensor"),
                "requires": ArrayAttr([StringAttr("rhs_immutable"),
                                       StringAttr("capacity_fit")])}),
        ]
        prove_imm = c.ProveOp(
            operands=[rhs],
            result_types=[c.ProofType(StringAttr("rhs_immutable"))],
            properties={"requirement": StringAttr("rhs_immutable"),
                        "producer_pass": StringAttr("merlin-infer-contract-facts")})
        ops.append(prove_imm)
        check_imm = c.CheckOp(operands=[rhs, [prove_imm.proof]],
                              properties={"requirement": StringAttr("rhs_immutable")})
        ops.append(check_imm)
        # The capacity check is always asserted; the proof exists only when the weight
        # actually fits — an unproven check blocks interface lowering downstream.
        cap_proofs = []
        if storage and _tensor_nbytes(rhs.type) <= storage:
            prove_cap = c.ProveOp(
                operands=[rhs],
                result_types=[c.ProofType(StringAttr("capacity_fit"))],
                properties={"requirement": StringAttr("capacity_fit"),
                            "producer_pass": StringAttr("merlin-infer-contract-facts")})
            ops.append(prove_cap)
            cap_proofs = [prove_cap.proof]
        ops.append(c.CheckOp(operands=[rhs, cap_proofs],
                             properties={"requirement": StringAttr("capacity_fit")}))
        for op in reversed(ops):
            block.insert_op_after(op, cap)
    return mod
