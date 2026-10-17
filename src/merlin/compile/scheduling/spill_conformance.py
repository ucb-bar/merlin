"""Exact device-transfer contract and independent interval provenance checking.

The checker reconstructs byte provenance from actual decoded instructions, not
from a spill plan. It operates on written intervals, never tensor elements or
individual bytes. Opaque arithmetic keeps its original operation order and
decoded nonaddress semantics. Only independently qualified exact private copies
may be inserted between those original operations.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from math import lcm

from merlin.sched.ir import Call
from merlin.semantic_compiler.allocate import _reservation_problem

from .program_limits import EncodedAccess
from .temporary_storage import StorageOptimizationUnavailable, _check_reservations, _overlap, _problem, _project


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


def verify_device_spills(
    original,
    realized,
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
    max_inserted_copies=8,
):
    """Compare actual original dataflow and actual final storage byte provenance.

    Target qualification establishes the decoder's operation/copy semantics.
    This checker establishes their preservation, not machine-code conformance
    or target execution qualification. It accepts no selected spill locations.
    """
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
            return access.start + access.size <= bank_map[access.space].capacity and not any(
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
    ordinal, copies = 0, 0
    for effect in actual:
        if any(not valid_actual(access) for access in effect.accesses):
            raise ValueError("actual spilled access exceeds its original declared address window")
        before = source[ordinal] if ordinal < len(source) else None
        matches = (
            before is not None
            and before.semantic_key == effect.semantic_key
            and before.exact_copy == effect.exact_copy
            and len(before.accesses) == len(effect.accesses)
            and all(
                (a.space, a.size, a.writes) == (b.space, b.size, b.writes)
                and (a.space in bank_map or a.start == b.start)
                for a, b in zip(before.accesses, effect.accesses, strict=True)
            )
        )
        if matches:
            # Every input must contain the exact original byte definitions,
            # including partial reads and overwritten aliases.
            matches = all(
                expected_memory.read(key(a), a.start, a.size) == actual_memory.read(key(b), b.start, b.size)
                for a, b in zip(before.accesses, effect.accesses, strict=True)
                if not a.writes
            )
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
                if base < 0 or base % alignment or base + obj.size > bank_map[obj.bank].capacity:
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
    if ordinal != len(source):
        raise ValueError("actual spill program omitted original operations")
    for window in external_windows:
        space = ("external", window.space)
        if expected_memory.read(space, window.start, window.size) != actual_memory.read(
            space, window.start, window.size
        ):
            raise ValueError("actual spill program changed complete external outputs or carried state")
    return {
        "schema": "merlin.device_spill_conformance.v1",
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
