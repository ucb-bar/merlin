"""Exact device-transfer contract and independent interval provenance checking.

The checker reconstructs byte provenance from actual decoded instructions, not
from a spill plan. It operates on written intervals, never tensor elements or
individual bytes. Opaque arithmetic keeps its original operation order and
decoded nonaddress semantics. Only independently qualified exact private copies
may be inserted or, when selected explicitly, omitted between those operations.
Omission propagates original byte definitions and compares every retained
operation's actual inputs and all external state.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from math import lcm

from merlin.compiler.kernel_ir.ir import Call
from merlin.compiler.schedule.pipeline.program_limits import EncodedAccess
from merlin.compiler.schedule.storage.temporaries import (
    StorageEffects,
    StorageOptimizationUnavailable,
    TemporaryObject,
    TemporaryStorageResult,
    _check_reservations,
    _overlap,
    _problem,
    _project,
    allocate_temporary_storage,
)
from merlin.compiler.semantic.search.allocation import _reservation_problem

# ---- spill_conformance -------------------------------------------------------------------------


@dataclass(frozen=True)
class SpillClass:
    """Explicit device register class and exact supported spill/reload pair.

    This is an independently selected target capability, not host CPU register
    allocation. Width and alignment are bytes; both directions must preserve
    every bit and complete synchronously. The emitter belongs to the target.
    """

    register_bank: str
    spill_bank: str
    width_bytes: int
    alignment: int
    emit_transfer: Callable[[EncodedAccess, EncodedAccess], Call]
    qualification_identity: str

    def __post_init__(self):
        if (
            any(
                not isinstance(name, str) or not name
                for name in (self.register_bank, self.spill_bank, self.qualification_identity)
            )
            or self.register_bank == self.spill_bank
            or type(self.width_bytes) is not int
            or self.width_bytes <= 0
            or type(self.alignment) is not int
            or self.alignment <= 0
            or not callable(self.emit_transfer)
        ):
            raise ValueError(
                "device spills require explicit distinct banks, exact width, alignment and qualified transfers"
            )


def validate_spill_classes(classes, banks):
    if type(classes) is not tuple or any(type(capability) is not SpillClass for capability in classes):
        raise ValueError("device spill classes must be explicitly selected immutable capabilities")
    bank_map = {bank.name: bank for bank in banks}
    if len({capability.register_bank for capability in classes}) != len(classes):
        raise ValueError("select one spill capability per device register bank")
    for capability in classes:
        source = bank_map.get(capability.register_bank)
        destination = bank_map.get(capability.spill_bank)
        if (
            source is None
            or destination is None
            or source.unit != "bytes"
            or destination.unit != "bytes"
            or source.backing == destination.backing
        ):
            raise StorageOptimizationUnavailable("device spills require declared distinct addressable byte backings")


@dataclass(frozen=True)
class _Piece:
    identity: tuple
    offset: int
    size: int


def _coalesce(parts):
    result = []
    for part in parts:
        if result and part.identity == result[-1].identity and part.offset == result[-1].offset + result[-1].size:
            previous = result.pop()
            part = _Piece(part.identity, previous.offset, previous.size + part.size)
        result.append(part)
    return tuple(result)


class _Memory:
    def __init__(self):
        self.spaces = {}

    def read(self, space, start, size):
        parts, cursor, end = [], start, start + size
        for low, high, part in self.spaces.get(space, ()):
            if high <= cursor or low >= end:
                continue
            if cursor < low:
                parts.append(_Piece(("initial", space), cursor, low - cursor))
                cursor = low
            stop = min(high, end)
            parts.append(_Piece(part.identity, part.offset + cursor - low, stop - cursor))
            cursor = stop
        if cursor < end:
            parts.append(_Piece(("initial", space), cursor, end - cursor))
        return _coalesce(parts)

    def write(self, space, start, size, parts):
        if sum(part.size for part in parts) != size:
            raise ValueError("spill copy changed its complete byte extent")
        end, retained = start + size, []
        for low, high, part in self.spaces.get(space, ()):
            if high <= start or low >= end:
                retained.append((low, high, part))
                continue
            if low < start:
                retained.append((low, start, _Piece(part.identity, part.offset, start - low)))
            if end < high:
                retained.append((end, high, _Piece(part.identity, part.offset + end - low, high - end)))
        cursor = start
        for part in parts:
            retained.append((cursor, cursor + part.size, part))
            cursor += part.size
        self.spaces[space] = sorted(retained, key=lambda row: row[0])


def verify_storage_dataflow(
    original,
    realized,
    objects,
    banks,
    adapter,
    classes=(),
    *,
    reservations=(),
    external_windows=(),
    private_ownership=False,
    max_calls=4096,
    max_objects=256,
    max_inserted_copies=8,
    eliminate_private_copies=False,
):
    """Compare actual original dataflow and actual final storage byte provenance.

    Target qualification establishes the decoder's operation/copy semantics.
    This checker establishes their preservation, not machine-code conformance
    or target execution qualification. It accepts no selected spill locations.
    """
    if type(eliminate_private_copies) is not bool:
        raise ValueError("private-copy elimination must be explicitly boolean")
    validate_spill_classes(classes, banks)
    _check_reservations(reservations)
    bank_map, source, bindings, _intervals, _graph, _order, _ids, _placement_banks = _problem(
        original, objects, banks, adapter, external_windows, private_ownership, max_calls, max_objects
    )
    reservation_problem = _reservation_problem(reservations, bank_map)
    if reservation_problem:
        raise ValueError("invalid device spill reservations: " + reservation_problem)
    object_map = {obj.name: obj for obj in objects}
    if type(max_inserted_copies) is not int or max_inserted_copies < 0:
        raise ValueError("spill conformance requires an explicit nonnegative copy budget")
    if (realized.name, realized.args, realized.attrs) != (original.name, original.args, original.attrs):
        raise ValueError("device spills changed the original program boundary")
    actual = _project(realized, adapter, max_calls)
    if any(effect.completion != "synchronous" for effect in (*source, *actual)):
        raise StorageOptimizationUnavailable("this spill family requires actual synchronous transfer completion")

    def key(access):
        return ("bank", bank_map[access.space].backing) if access.space in bank_map else ("external", access.space)

    def valid_actual(access):
        if access.space in bank_map:
            return (
                bank_map[access.space].start <= access.start
                and access.start + access.size <= bank_map[access.space].capacity
            ) and not any(
                bank_map[row.storage].backing == bank_map[access.space].backing
                and access.start < row.start + row.extent
                and row.start < access.start + access.size
                for row in reservations
            )
        return any(
            window.space == access.space
            and window.start <= access.start
            and access.start + access.size <= window.start + window.size
            for window in external_windows
        )

    def copy_allowed(effect):
        if not effect.exact_copy:
            return False
        read, write = effect.accesses
        for capability in classes:
            if (
                (read.space, write.space)
                in {
                    (capability.register_bank, capability.spill_bank),
                    (capability.spill_bank, capability.register_bank),
                }
                and read.size == capability.width_bytes
                and read.start % lcm(capability.alignment, bank_map[read.space].alignment) == 0
                and write.start % lcm(capability.alignment, bank_map[write.space].alignment) == 0
            ):
                return True
        return False

    expected_memory, actual_memory = _Memory(), _Memory()
    ordinal, copies, removed = 0, 0, 0

    def removable(effect):
        return (
            eliminate_private_copies
            and effect.exact_copy
            and all(access.space in bank_map for access in effect.accesses)
            and not _overlap(*effect.accesses, bank_map)
        )

    def advance_copy(effect):
        read, write = effect.accesses
        expected_memory.write(
            key(write), write.start, write.size, expected_memory.read(key(read), read.start, read.size)
        )

    for effect in actual:
        if any(not valid_actual(access) for access in effect.accesses):
            raise ValueError("actual spilled access exceeds its original declared address window")

        def matches_source(before):
            return (
                before is not None
                and before.semantic_key == effect.semantic_key
                and before.exact_copy == effect.exact_copy
                and before.elementwise == effect.elementwise
                and len(before.accesses) == len(effect.accesses)
                and all(
                    (a.space, a.size, a.writes) == (b.space, b.size, b.writes)
                    and (a.space in bank_map or a.start == b.start)
                    for a, b in zip(before.accesses, effect.accesses, strict=True)
                )
            ) and all(
                expected_memory.read(key(a), a.start, a.size) == actual_memory.read(key(b), b.start, b.size)
                for a, b in zip(before.accesses, effect.accesses, strict=True)
                if not a.writes
            )

        before = source[ordinal] if ordinal < len(source) else None
        while before is not None and not matches_source(before) and removable(before):
            advance_copy(before)
            ordinal, removed = ordinal + 1, removed + 1
            before = source[ordinal] if ordinal < len(source) else None
        matches = matches_source(before)
        if not matches:
            if not copy_allowed(effect):
                raise ValueError("actual spill program changed ordered original semantics or read overwritten storage")
            read, write = effect.accesses
            if _overlap(read, write, bank_map):
                raise ValueError("spill transfer overlaps its own source without an exact overlap contract")
            parts = actual_memory.read(key(read), read.start, read.size)
            if any(part.identity[0] == "initial" and part.identity[1][0] == "bank" for part in parts):
                raise ValueError("actual spill transfer reads uninitialized private storage")
            actual_memory.write(key(write), write.start, write.size, parts)
            copies += 1
            if copies > max_inserted_copies:
                raise ValueError("actual spill program exceeds its inserted-copy budget")
            continue
        for binding, access in zip(bindings[ordinal], effect.accesses, strict=True):
            if binding is not None:
                obj = object_map[binding[0]]
                base = access.start - binding[1]
                alignment = lcm(obj.alignment, bank_map[obj.bank].alignment, bank_map[access.space].alignment)
                if base < bank_map[obj.bank].start or base % alignment or base + obj.size > bank_map[obj.bank].capacity:
                    raise ValueError("actual spilled object violates its original extent or physical alignment")
        # Preserve within-instruction alias relations too. A streaming target
        # need not read every operand before starting its first output write.
        for index, left in enumerate(before.accesses):
            for other, right in enumerate(before.accesses[index + 1 :], index + 1):
                a, b = effect.accesses[index], effect.accesses[other]
                if left.writes or right.writes:
                    old_overlap, new_overlap = _overlap(left, right, bank_map), _overlap(a, b, bank_map)
                    if old_overlap != new_overlap or old_overlap and left.start - right.start != a.start - b.start:
                        raise ValueError("device spills changed within-instruction physical aliasing")
        copy_parts = None
        if before.exact_copy:
            read = before.accesses[0]
            copy_parts = expected_memory.read(key(read), read.start, read.size)
        for index, (a, b) in enumerate(zip(before.accesses, effect.accesses, strict=True)):
            if a.writes:
                parts = copy_parts or (_Piece(("write", ordinal, index), 0, a.size),)
                expected_memory.write(key(a), a.start, a.size, parts)
                actual_memory.write(key(b), b.start, b.size, parts)
        ordinal += 1
    while ordinal < len(source) and removable(source[ordinal]):
        advance_copy(source[ordinal])
        ordinal, removed = ordinal + 1, removed + 1
    if ordinal != len(source):
        raise ValueError("actual spill program omitted original operations")
    for window in external_windows:
        space = ("external", window.space)
        if expected_memory.read(space, window.start, window.size) != actual_memory.read(
            space, window.start, window.size
        ):
            raise ValueError("actual spill program changed complete external outputs or carried state")
    report = {
        "schema": "merlin.storage_dataflow.v1" if eliminate_private_copies else "merlin.device_spill_conformance.v1",
        "configuration_identity": adapter.configuration_identity,
        "numerical_policy_identity": adapter.numerical_policy_identity,
        "adapter_qualification_identity": adapter.qualification_identity,
        "banks": [bank.record() for bank in banks],
        "objects": [asdict(obj) for obj in objects],
        "reservations": [asdict(row) for row in reservations],
        "external_windows": [asdict(window) for window in external_windows],
        "private_ownership": private_ownership,
        "checking_limits": {
            "max_calls": max_calls,
            "max_objects": max_objects,
            "max_inserted_copies": max_inserted_copies,
        },
        "classes": [
            {
                name: getattr(capability, name)
                for name in ("register_bank", "spill_bank", "width_bytes", "alignment", "qualification_identity")
            }
            for capability in classes
        ],
        "source_sha256": original.digest(),
        "realized_sha256": realized.digest(),
        "original_commands": len(source),
        "actual_commands": len(actual),
        "inserted_exact_copies": copies,
        "checked_original_operations": ordinal,
        "requested_read_bytes": sum(a.size for op in actual for a in op.accesses if not a.writes),
        "requested_write_bytes": sum(a.size for op in actual for a in op.accesses if a.writes),
        "execution_qualification": "not_established",
        "machine_emission": "not_established",
        "timing": "UNKNOWN",
    }
    if eliminate_private_copies:
        report["eliminated_exact_copies"] = removed
        report["original_requested_read_bytes"] = sum(a.size for op in source for a in op.accesses if not a.writes)
        report["original_requested_write_bytes"] = sum(a.size for op in source for a in op.accesses if a.writes)
    return report


def verify_device_spills(original, realized, objects, banks, adapter, classes, **options):
    """Verify exact device spill insertion without permitting operation removal."""
    if options.get("eliminate_private_copies", False) is not False:
        raise ValueError("spill-only verification cannot authorize original operation removal")
    return verify_storage_dataflow(original, realized, objects, banks, adapter, classes, **options)


# ---- device_spills -----------------------------------------------------------------------------
# Bounded exact device spills through the ordinary temporary allocation entry.
#
# Only explicitly selected device register classes participate. Every insertion
# reprojects actual instructions and recomputes liveness, physical allocation and
# completion checks. Unknown costs use a deterministic longest-unused-gap policy;
# no cycle model or host CPU register allocator is supplied.


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
        # The save reads the original register too. Its transfer alignment
        # can be stricter than the original arithmetic/bank requirements.
        # Replan that object's location along with the new save/reload slots.
        (*(replace(item, alignment=alignment) if item.name == obj.name else item for item in objects), saved, restored),
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
    on_repair=None,
) -> TemporaryStorageResult | None:
    """Try legal placement, then bounded exact spill/reload insertion and replan.

    None preserves the caller's legal fallback when this bounded family cannot
    fit. A failed invariant or unavailable checker is never treated as evidence
    that another unchecked candidate is legal.
    """
    validate_spill_classes(classes, banks)
    if type(max_spills) is not int or max_spills < 0:
        raise ValueError("device spill insertion requires a nonnegative bounded repair count")
    if on_repair is not None and not callable(on_repair):
        raise ValueError("device spill repair notification must be an explicit callable")
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
        if on_repair is not None:
            on_repair("inserted exact device spill/reload; recompute liveness, allocation and completion")
        repairs.append(
            {
                **repair,
                "before_sha256": previous,
                "after_sha256": current.digest(),
                "invalidated": ["liveness", "allocation", "completion", "emission", "program_size"],
            }
        )
    raise AssertionError("unreachable bounded spill loop")


__all__ = [
    "SpillClass",
    "allocate_with_spills",
    "verify_device_spills",
]
