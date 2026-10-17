"""Bounded exact device spills through the ordinary temporary allocation entry.

Only explicitly selected device register classes participate. Every insertion
reprojects actual instructions and recomputes liveness, physical allocation and
completion checks. Unknown costs use a deterministic longest-unused-gap policy;
no cycle model or host CPU register allocator is supplied.
"""

from __future__ import annotations

from dataclasses import replace
from math import lcm

from merlin.sched.ir import Call

from .program_limits import EncodedAccess
from .spill_conformance import SpillClass, validate_spill_classes, verify_device_spills
from .temporary_storage import (
    StorageEffects,
    StorageOptimizationUnavailable,
    TemporaryObject,
    TemporaryStorageResult,
    _problem,
    allocate_temporary_storage,
)

__all__ = ["SpillClass", "allocate_with_spills", "verify_device_spills"]


def _copy(adapter, capability, source, destination):
    call = capability.emit_transfer(source, destination)
    if type(call) is not Call:
        raise ValueError("device spill emitter must produce an actual typed instruction call")
    actual = adapter.project(call)
    if (
        type(actual) is not StorageEffects
        or not actual.exact_copy
        or actual.completion != "synchronous"
        or actual.accesses != (source, destination)
    ):
        raise ValueError("emitted device spill/reload is not the declared exact synchronous byte transfer")
    return call


def _gap(kernel, objects, banks, adapter, classes, model):
    bank_map, effects, bindings, _intervals, _graph, _order, _ids, _placement = _problem(
        kernel,
        objects,
        banks,
        adapter,
        model["external_windows"],
        model["private_ownership"],
        model["max_calls"],
        model["max_objects"],
    )
    if any(effect.completion != "synchronous" for effect in effects):
        raise StorageOptimizationUnavailable("device spill insertion requires actual synchronous original completion")
    capabilities = {capability.register_bank: capability for capability in classes}
    candidates = []
    for object_index, obj in enumerate(objects):
        capability = capabilities.get(obj.bank)
        if capability is None or obj.size != capability.width_bytes:
            continue
        positions = [
            index
            for index, row in enumerate(bindings)
            if any(binding is not None and binding[0] == obj.name for binding in row)
        ]
        for left, right in zip(positions, positions[1:]):
            if right <= left + 1:
                continue

            # Copies inserted by an earlier round are not another arithmetic
            # consumer to spill around. This keeps nested repair conservative.
            def spill_transfer(effect):
                return effect.exact_copy and {access.space for access in effect.accesses} == {
                    capability.register_bank,
                    capability.spill_bank,
                }

            if spill_transfer(effects[left]) or spill_transfer(effects[right]):
                continue
            if not any(
                binding is not None and binding[0] == obj.name and not access.writes
                for access, binding in zip(effects[right].accesses, bindings[right], strict=True)
            ):
                continue
            candidates.append((-(right - left), left, right, object_index, obj, capability))
    if not candidates:
        return None
    _distance, left, right, _object_index, obj, capability = min(candidates, key=lambda row: row[:4])
    return bank_map, effects, bindings, left, right, obj, capability


def _insert(kernel, objects, banks, adapter, selected, ordinal):
    bank_map, effects, bindings, left, right, obj, capability = selected

    def fresh(name, bank, alignment):
        names = {item.name for item in objects}
        while name in names:
            name += "_"
        backing = bank_map[bank].backing
        end = max(
            (item.source_base + item.size for item in objects if bank_map[item.bank].backing == backing), default=0
        )
        base = (end + alignment - 1) // alignment * alignment
        return TemporaryObject(name, bank, base, obj.size, alignment)

    alignment = lcm(obj.alignment, capability.alignment)
    saved = fresh(f"{obj.name}_spill_{ordinal}", capability.spill_bank, alignment)
    restored = fresh(f"{obj.name}_reload_{ordinal}", obj.bank, alignment)
    store = _copy(
        adapter,
        capability,
        EncodedAccess(capability.register_bank, obj.source_base, obj.size),
        EncodedAccess(capability.spill_bank, saved.source_base, obj.size, True),
    )
    reload = _copy(
        adapter,
        capability,
        EncodedAccess(capability.spill_bank, saved.source_base, obj.size),
        EncodedAccess(capability.register_bank, restored.source_base, obj.size, True),
    )
    body = []
    for index, (call, effect, row) in enumerate(zip(kernel.body, effects, bindings, strict=True)):
        if index == right:
            body.append(reload)
        if index >= right:
            accesses = tuple(
                replace(access, start=restored.source_base + binding[1])
                if binding is not None and binding[0] == obj.name
                else access
                for access, binding in zip(effect.accesses, row, strict=True)
            )
            if accesses != effect.accesses:
                rewritten = adapter.rewrite(call, accesses)
                if type(rewritten) is not Call or adapter.project(rewritten) != replace(effect, accesses=accesses):
                    raise ValueError("spill rebasing changed an original instruction's decoded semantics")
                call = rewritten
        body.append(call)
        if index == left:
            body.append(store)
    return (
        replace(kernel, body=tuple(body)),
        (*objects, saved, restored),
        {
            "object": obj.name,
            "gap_calls": [left, right],
            "saved_object": saved.name,
            "restored_object": restored.name,
            "transfer_qualification_identity": capability.qualification_identity,
        },
    )


def allocate_with_spills(
    kernel,
    objects,
    banks,
    adapter,
    classes,
    *,
    reservations=(),
    external_windows=(),
    private_ownership=False,
    max_calls=4096,
    max_objects=256,
    timeout_ms=5000,
    max_spills=4,
) -> TemporaryStorageResult | None:
    """Try legal placement, then bounded exact spill/reload insertion and replan.

    None preserves the caller's legal fallback when this bounded family cannot
    fit. A failed invariant or unavailable checker is never treated as evidence
    that another unchecked candidate is legal.
    """
    validate_spill_classes(classes, banks)
    if type(max_spills) is not int or max_spills < 0:
        raise ValueError("device spill insertion requires a nonnegative bounded repair count")
    model = dict(
        reservations=reservations,
        external_windows=external_windows,
        private_ownership=private_ownership,
        max_calls=max_calls,
        max_objects=max_objects,
    )
    current, current_objects, repairs = kernel, objects, []
    for attempt in range(max_spills + 1):
        # Copy forwarding is selected by the outer allocator before repair.
        # Keep this loop's transformation family within its insertion proof.
        result = allocate_temporary_storage(
            current, current_objects, banks, adapter, timeout_ms=timeout_ms, eliminate_copies=False, **model
        )
        if result is not None:
            proof = verify_device_spills(
                kernel, result.kernel, objects, banks, adapter, classes, max_inserted_copies=2 * max_spills, **model
            )
            report = {
                **result.report,
                "spills": proof,
                "repair_rounds": repairs,
                "spill_cost_policy": "longest-unused-gap; timing UNKNOWN",
            }
            return TemporaryStorageResult(result.kernel, report)
        if attempt == max_spills:
            return None
        selected = _gap(current, current_objects, banks, adapter, classes, model)
        if selected is None:
            return None
        previous = current.digest()
        current, current_objects, repair = _insert(current, current_objects, banks, adapter, selected, attempt)
        repairs.append(
            {
                **repair,
                "before_sha256": previous,
                "after_sha256": current.digest(),
                "invalidated": ["liveness", "allocation", "completion", "emission", "program_size"],
            }
        )
    raise AssertionError("unreachable bounded spill loop")
