"""Cross-op analyses for the core dialects.

Local single-op checks live in each op's ``verify_``; the properties below need
program order or SSA def-use chains across ops, so they are explicit analyses run by
the lowering pipeline (and directly by tests). Each returns a list of problem strings
(empty == clean).
"""
from __future__ import annotations

from .._common import HAS_XDSL

if HAS_XDSL:
    from .. import contract as c
    from .. import interface as i
    from .. import runtime as r
    from .. import schedule as s


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
                        "by an earlier %s" % (op.name, evicted[operand].name))
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
                    problems.append(
                        "schedule.place to resident state without a capability operand")
                elif "resident_packed_tensor" not in cap_features.get(op.capability, set()):
                    problems.append(
                        "schedule.place to resident state but the capability does not "
                        "declare resident_packed_tensor")
        elif isinstance(op, s.SelectInterfaceOp):
            if op.interface.data not in all_features:
                problems.append(
                    "schedule.select_interface %r not declared by any "
                    "contract.capability in the module" % op.interface.data)
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
                    "contract.check %r has no proof token and no capability covers it"
                    % op.requirement.data)
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
                problems.append(
                    "runtime.command_buffer.append on a value not produced by "
                    "command_buffer.create")
                continue
            dev_op = creator.dev.owner
            if not isinstance(dev_op, r.DeviceGetOp):
                problems.append(
                    "runtime.command_buffer.create device not produced by device.get")
                continue
            backend = dev_op.backend.data.value
            if op.queue is not None:
                queue = op.queue.data.value
                if queue not in BACKEND_QUEUES.get(backend, set()):
                    problems.append(
                        "queue kind %r unsupported on backend %r" % (queue, backend))
        elif isinstance(op, r.SubmitOp):
            creator = op.cb.owner
            if isinstance(creator, r.CommandBufferCreateOp) and creator.dev is not op.dev:
                problems.append(
                    "runtime.submit device differs from the command buffer's device")
    return problems
