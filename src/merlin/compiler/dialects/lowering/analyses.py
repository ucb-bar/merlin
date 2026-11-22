"""Cross-op analyses for the core dialects.

Local single-op checks live in each op's ``verify_``; the properties below need
program order or SSA def-use chains across ops, so they are explicit analyses run by
the lowering pipeline (and directly by tests). Each returns a list of problem strings
(empty == clean).
"""

from __future__ import annotations

from typing import Any

from xdsl.dialects.builtin import IntegerAttr, IntegerType
from xdsl.ir import Operation, SSAValue

from merlin.compiler.dialects.lowering.queries import (
    ELEMENTWISE_COMBINES, elementwise_operands, find_elementwise, find_matmuls, matmul_lhs_rhs,
)
from merlin.compiler.dialects.xdsl_context import HAS_XDSL

# ---- analyses ----------------------------------------------------------------------------------

if HAS_XDSL:
    from merlin.compiler.dialects import contract as c
    from merlin.compiler.dialects import interface as i
    from merlin.compiler.dialects import runtime as r
    from merlin.compiler.dialects import schedule as s


def _walk_blocks(module):
    for op in module.walk():
        for region in op.regions:
            for block in region.blocks:
                yield block


def check_no_use_after_evict(module) -> list[str]:
    """No op may consume a resident handle after `interface.resident_evict` of it."""
    if not HAS_XDSL:
        return []
    problems: list[str] = []
    for block in _walk_blocks(module):
        evicted = {}
        for op in block.ops:
            for operand in op.operands:
                if operand in evicted and not isinstance(op, i.ResidentEvictOp):
                    problems.append(
                        "use of resident handle after evict: %s uses a handle evicted "
                        "by an earlier %s" % (op.name, evicted[operand].name)
                    )
            if isinstance(op, i.ResidentEvictOp):
                evicted[op.handle] = op
    return problems


def check_place_legality(module) -> list[str]:
    """`schedule.place` into resident state needs a capability allowing residency, and
    `schedule.select_interface` must be backed by a declared capability feature."""
    if not HAS_XDSL:
        return []
    problems: list[str] = []
    cap_features: dict = {}  # capability SSA value -> set of feature strings
    all_features: set[str] = set()
    for op in module.walk():
        if isinstance(op, c.CapabilityOp):
            feats = {e.data for e in op.features}
            cap_features[op.cap] = feats
            all_features |= feats
    for op in module.walk():
        if isinstance(op, s.PlaceOp):
            if op.state.data == s.MemoryState.RESIDENT:
                if op.capability is None:
                    problems.append("schedule.place to resident state without a capability operand")
                elif "resident_packed_tensor" not in cap_features.get(op.capability, set()):
                    problems.append(
                        "schedule.place to resident state but the capability does not declare resident_packed_tensor"
                    )
        elif isinstance(op, s.SelectInterfaceOp):
            if op.interface.data not in all_features:
                problems.append(
                    "schedule.select_interface %r not declared by any "
                    "contract.capability in the module" % op.interface.data
                )
    return problems


def check_contract_discharged(module) -> list[str]:
    """Every `contract.check` must hold a proof token or be covered by a capability."""
    if not HAS_XDSL:
        return []
    problems: list[str] = []
    declared: set[str] = set()
    for op in module.walk():
        if isinstance(op, c.CapabilityOp):
            declared |= {e.data for e in op.features}
    for op in module.walk():
        if isinstance(op, c.CheckOp):
            if not op.proofs and op.requirement.data not in declared:
                problems.append(
                    "contract.check %r has no proof token and no capability covers it" % op.requirement.data
                )
    return problems


# Queues each backend supports (cross-op half of command_buffer.append verification).
BACKEND_QUEUES = {
    "simulator": {"compute", "dma", "unified"},
    "host": {"compute", "host", "unified"},
    "baremetal": {"compute", "dma", "control", "unified"},
    "zephyr": {"compute", "dma", "control", "unified"},
    "firesim": {"compute", "dma", "unified"},
    "external": {"compute", "dma", "copy", "host", "control", "unified"},
}


def check_command_buffer_consistency(module) -> list[str]:
    """Appends/submits must trace back to one create on one device; queue kinds must be
    supported by the device's backend."""
    if not HAS_XDSL:
        return []
    problems: list[str] = []
    for op in module.walk():
        if isinstance(op, r.CommandBufferAppendOp):
            creator = op.cb.owner
            if not isinstance(creator, r.CommandBufferCreateOp):
                problems.append("runtime.command_buffer.append on a value not produced by command_buffer.create")
                continue
            dev_op = creator.dev.owner
            if not isinstance(dev_op, r.DeviceGetOp):
                problems.append("runtime.command_buffer.create device not produced by device.get")
                continue
            backend = dev_op.backend.data.value
            if op.queue is not None:
                queue = op.queue.data.value
                if queue not in BACKEND_QUEUES.get(backend, set()):
                    problems.append("queue kind %r unsupported on backend %r" % (queue, backend))
        elif isinstance(op, r.SubmitOp):
            creator = op.cb.owner
            if isinstance(creator, r.CommandBufferCreateOp) and creator.dev is not op.dev:
                problems.append("runtime.submit device differs from the command buffer's device")
    return problems


# ---- integer_constant_eval ---------------------------------------------------------------------
# Conservative integer SSA folding for statically indexed tensor lowering.
#
# Only pure, explicitly supported integer instructions are evaluated. Loads,
# block arguments and unknown instructions stay unknown. Integer widths and
# signedness come from the IR, never from a target or a model's identity.


def constant_integer(value: SSAValue) -> int | None:
    """Return the signed integer constant carried by ``value``, if provable."""
    cache: dict[SSAValue, int | None] = {}

    def visit(v: SSAValue) -> int | None:
        if v in cache:
            return cache[v]
        cache[v] = None
        if not isinstance(v.type, IntegerType) or not isinstance(v.owner, Operation):
            return None
        width = int(v.type.width.data)
        mask = (1 << width) - 1
        op = v.owner
        overflow = op.properties.get("overflowFlags")
        if isinstance(overflow, IntegerAttr) and int(overflow.value.data):
            # A wrapping evaluation cannot prove an nsw/nuw operation non-poison.
            return None
        attr = op.properties.get("value", op.attributes.get("value"))
        if op.name in ("llvm.mlir.constant", "arith.constant") and isinstance(attr, IntegerAttr):
            result = int(attr.value.data)
        else:
            vals = [visit(operand) for operand in op.operands]
            if not vals or any(item is None for item in vals):
                return None
            a = vals[0]
            assert a is not None
            if op.name in ("llvm.sext", "llvm.trunc", "arith.extsi", "arith.trunci"):
                result = a
            elif op.name in ("llvm.zext", "arith.extui"):
                result = a & ((1 << int(op.operands[0].type.width.data)) - 1)
            elif len(vals) == 2:
                b = vals[1]
                assert b is not None
                if op.name in ("llvm.add", "arith.addi"):
                    result = a + b
                elif op.name in ("llvm.sub", "arith.subi"):
                    result = a - b
                elif op.name in ("llvm.mul", "arith.muli"):
                    result = a * b
                elif op.name in ("llvm.and", "arith.andi"):
                    result = a & b
                elif op.name in ("llvm.or", "arith.ori"):
                    result = a | b
                elif op.name in ("llvm.xor", "arith.xori"):
                    result = a ^ b
                elif op.name in ("llvm.shl", "llvm.ashr", "llvm.lshr") and 0 <= b < width:
                    result = (
                        a << b if op.name == "llvm.shl" else (a >> b if op.name == "llvm.ashr" else (a & mask) >> b)
                    )
                else:
                    return None
            else:
                return None
        result &= mask
        if result >= 1 << (width - 1):
            result -= 1 << width
        cache[v] = result
        return result

    return visit(value)


# ---- contract_facts ----------------------------------------------------------------------------
# linalg input -> ``contract`` facts (the merlin-infer-contract-facts +
# merlin-attach-target-capabilities stages).
#
# Analyzes the payload: finds matmul-family ops, detects a shared immutable RHS (a block
# argument used by >= 2 matmuls), and inserts contract IR at the top of the function:
# capability (from the target contract), assume/fact on the reused weight, the
# resident_packed_tensor requirement, and proofs/checks for what is actually provable
# (immutability from block-arg-ness; capacity from the contract's resident storage).


def _trace_to_block_arg(val, block):
    """Walk up pure layout/view ops (transpose, reshape, collapse/expand, cast, copy) from
    ``val`` to its source block argument, if any. Real-model matmuls feed the weight in as
    ``A @ Wᵀ`` — the RHS owner is a ``linalg.transpose`` of a block-arg weight, not the
    block arg directly — so the immutable-weight detection must trace through it."""
    from xdsl.ir import BlockArgument

    VIEW = (
        "linalg.transpose",
        "tensor.collapse_shape",
        "tensor.expand_shape",
        "tensor.reshape",
        "tensor.cast",
        "linalg.copy",
    )
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


def _resident_storage_bytes(tc: dict[str, Any]) -> int:
    """Resident operand-store capacity for the capacity-fit proof.

    An explicit contract capacity takes precedence. Otherwise resolve the operand
    store structurally from the selected RTL facts: extractor memory names are
    physical labels, not a stable ``scratchpad`` role. An unavailable or ambiguous
    store retains the existing no-proof result of zero.
    """
    v = (tc.get("capabilities", {}) or {}).get("resident_storage_bytes")
    if v is not None:
        return int(v)
    name = tc.get("name")
    if name:
        try:
            from merlin.target.capabilities.address_space import derive_address_space, operand_store
            from merlin.target.rtl.facts import load_facts

            space = derive_address_space(name, facts=load_facts(name))
            store = operand_store(space).store
            if store is not None and store.nbytes is not None and store.nbytes > 0:
                return store.nbytes
        except Exception:  # noqa: BLE001 — facts unavailable ⇒ fall back to the fail-open default
            pass
    return 0


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
    if not isinstance(target_contract, dict) or not target_contract.get("name"):
        raise ValueError("contract lowering requires an explicit named target contract")
    if not HAS_XDSL:
        return module
    from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, StringAttr

    from merlin.compiler.dialects import contract as c

    tc = target_contract
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
            "runtime": ArrayAttr([StringAttr(b) for b in tc.get("runtime", {}).get("backends", [])]),
        },
    )
    block.insert_op_before(cap, first_op)

    storage = _resident_storage_bytes(tc)
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
            c.AssumeOp(
                operands=[rhs],
                properties={"kind": StringAttr("immutable"), "lifetime": c.LifetimeAttr(c.Lifetime.WITHIN_REGION)},
            ),
            c.FactOp(
                operands=[rhs],
                properties={
                    "role": c.MemoryRoleAttr(c.MemoryRole.REUSABLE_WEIGHT),
                    "reuse_count": IntegerAttr(reuse, 64),
                    "layout": c.LayoutRoleAttr(c.LayoutRole.CANONICAL),
                },
            ),
            c.RequireOp(
                properties={
                    "feature": StringAttr("resident_packed_tensor"),
                    "requires": ArrayAttr([StringAttr("rhs_immutable"), StringAttr("capacity_fit")]),
                }
            ),
        ]
        prove_imm = c.ProveOp(
            operands=[rhs],
            result_types=[c.ProofType(StringAttr("rhs_immutable"))],
            properties={
                "requirement": StringAttr("rhs_immutable"),
                "producer_pass": StringAttr("merlin-infer-contract-facts"),
            },
        )
        ops.append(prove_imm)
        check_imm = c.CheckOp(
            operands=[rhs, [prove_imm.proof]], properties={"requirement": StringAttr("rhs_immutable")}
        )
        ops.append(check_imm)
        # The capacity check is always asserted; the proof exists only when the weight
        # actually fits — an unproven check blocks interface lowering downstream.
        cap_proofs = []
        if storage and _tensor_nbytes(rhs.type) <= storage:
            prove_cap = c.ProveOp(
                operands=[rhs],
                result_types=[c.ProofType(StringAttr("capacity_fit"))],
                properties={
                    "requirement": StringAttr("capacity_fit"),
                    "producer_pass": StringAttr("merlin-infer-contract-facts"),
                },
            )
            ops.append(prove_cap)
            cap_proofs = [prove_cap.proof]
        ops.append(c.CheckOp(operands=[rhs, cap_proofs], properties={"requirement": StringAttr("capacity_fit")}))
        for op in reversed(ops):
            block.insert_op_after(op, cap)
    return mod


def find_matmuls(module):
    """All linalg matmul-family ops in the module (quantized + plain)."""
    from xdsl.dialects.linalg import ops as linalg_ops

    found = []
    for op in module.walk():
        if isinstance(op, (linalg_ops.QuantizedMatmulOp, linalg_ops.MatmulOp)):
            found.append(op)
    return found


def matmul_lhs_rhs(mm):
    """(lhs, rhs) SSA values of a linalg matmul-family op (inputs[0], inputs[1])."""
    return mm.inputs[0], mm.inputs[1]


# linalg named ops that map onto a runtime VECTOR_MAP combine. `linalg.sub` is deliberately absent:
# the runtime implements add and mul only, and lowering a subtract as an add would be a miscompile.
ELEMENTWISE_COMBINES = {"linalg.add": "add", "linalg.mul": "mul"}


def find_elementwise(module):
    """All linalg elementwise ops the interface layer can materialize, as (op, combine) pairs."""
    return [(op, ELEMENTWISE_COMBINES[op.name]) for op in module.walk() if op.name in ELEMENTWISE_COMBINES]


def elementwise_operands(op):
    """(lhs, rhs) SSA values of a linalg elementwise op."""
    return op.inputs[0], op.inputs[1]
