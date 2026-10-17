"""Relocate private temporaries using actual effects and the existing allocator.

This finite straight-line family uses byte-addressed banks and real completion
instructions. Target adapters decode actual operands and rewrite addresses;
Merlin derives lifetimes, allocates and independently checks realized effects.
It never supplies CPU register allocation or infers effects from opcode names.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from math import lcm

from merlin.sched.ir import Call, Kernel
from merlin.sched.ir.kernel import check_structure
from merlin.semantic_compiler.allocate import allocate, check_assignment
from merlin.semantic_compiler.allocation_types import CandidateGraph, Reservation, StorageBank, Value
from merlin.semantic_compiler.rules import AddressConstraint

from .program_limits import AddressWindow, EncodedAccess


class StorageOptimizationUnavailable(ValueError):
    """Retain the original route: this storage family lacks required information."""


@dataclass(frozen=True)
class TemporaryObject:
    """One invocation-private backing allocation, including all of its views.

    Source addresses are virtual before allocation. Multiple typed bank views
    with the same physical backing use a common byte coordinate system. An
    object cannot escape, be a published output or carry invocation state.
    """

    name: str
    bank: str
    source_base: int
    size: int
    alignment: int = 1

    def __post_init__(self):
        AddressWindow(self.bank, self.source_base, self.size)
        if not isinstance(self.name, str) or not self.name or type(self.alignment) is not int or self.alignment <= 0:
            raise ValueError("private storage requires a named object and positive byte alignment")


@dataclass(frozen=True)
class StorageEffects:
    """Decoded effects, including nonaddress semantics and actual completion.

    semantic_key identifies all nonrelocated behavior, including arithmetic,
    conversions, external pointer operands and configuration changes. The
    adapter's independent qualification must establish this interpretation.
    A wait establishes completion and visibility of its named actual tokens.
    """

    semantic_key: str
    accesses: tuple[EncodedAccess, ...] = ()
    completion: str = "unknown"
    token: str | None = None
    waits: tuple[str, ...] = ()

    def __post_init__(self):
        if (
            not isinstance(self.semantic_key, str)
            or not self.semantic_key
            or type(self.accesses) is not tuple
            or any(type(access) is not EncodedAccess for access in self.accesses)
            or self.completion not in {"synchronous", "asynchronous", "wait", "unknown"}
            or type(self.waits) is not tuple
            or any(not isinstance(token, str) or not token for token in self.waits)
            or len(set(self.waits)) != len(self.waits)
            or (self.token is not None and (not isinstance(self.token, str) or not self.token))
        ):
            raise ValueError("storage effects require complete typed access and completion declarations")
        if (
            (self.completion == "asynchronous" and (self.token is None or self.waits))
            or (self.completion != "asynchronous" and self.token is not None)
            or (self.completion == "wait" and (not self.waits or self.accesses))
            or (self.completion != "wait" and self.waits)
        ):
            raise ValueError("storage completion tokens must describe actual async operations or pure completion waits")


@dataclass(frozen=True)
class StorageAdapter:
    project: Callable[[Call], StorageEffects | None]
    rewrite: Callable[[Call, tuple[EncodedAccess, ...]], Call]
    configuration_identity: str
    numerical_policy_identity: str
    qualification_identity: str

    def __post_init__(self):
        if (
            not callable(self.project)
            or not callable(self.rewrite)
            or any(
                not isinstance(identity, str) or not identity
                for identity in (
                    self.configuration_identity,
                    self.numerical_policy_identity,
                    self.qualification_identity,
                )
            )
        ):
            raise ValueError("storage relocation requires explicit selected semantic adapters and identities")


@dataclass(frozen=True)
class TemporaryStorageResult:
    kernel: Kernel
    report: dict


def _check_reservations(reservations):
    if type(reservations) is not tuple or any(type(row) is not Reservation for row in reservations):
        raise ValueError("temporary reservations must be explicit immutable byte intervals")


def _project(kernel, adapter, max_calls):
    if type(kernel) is not Kernel or check_structure(kernel):
        raise ValueError("temporary allocation requires a structurally valid kernel")
    if kernel.attrs or len(kernel.body) > max_calls:
        raise StorageOptimizationUnavailable(
            "temporary allocation has unknown kernel attributes or exceeds its call budget"
        )
    effects = []
    for call in kernel.body:
        if (
            type(call) is not Call
            or call.unit
            or call.produces
            or call.awaits
            or call.stages
            or call.sets
            or call.assumes
        ):
            raise StorageOptimizationUnavailable(
                "temporary allocation requires straight-line calls with decoded execution effects"
            )
        effect = adapter.project(call)
        if type(effect) is not StorageEffects or effect.completion == "unknown":
            raise StorageOptimizationUnavailable("unknown instruction or completion is a conservative storage barrier")
        effects.append(effect)
    return tuple(effects)


def _overlap(left, right, banks):
    def backing(access):
        return ("bank", banks[access.space].backing) if access.space in banks else ("external", access.space)

    return (
        backing(left) == backing(right)
        and left.start < right.start + right.size
        and right.start < left.start + left.size
    )


def _completion(effects, banks):
    """Actual source/realized hazards; an issue-only instruction is not a wait."""
    pending, seen, finished = {}, set(), {}
    for index, effect in enumerate(effects):
        for token in effect.waits:
            if token not in pending:
                raise ValueError("completion wait names a missing or already completed token")
            source, _accesses = pending.pop(token)
            finished[source] = index
        for access in effect.accesses:
            if any(
                (access.writes or earlier.writes) and _overlap(access, earlier, banks)
                for _source, accesses in pending.values()
                for earlier in accesses
            ):
                raise ValueError("storage access conflicts with work whose actual completion was not established")
        if effect.completion == "asynchronous":
            if effect.token in seen:
                raise ValueError("completion token is reused within an invocation")
            seen.add(effect.token)
            pending[effect.token] = (index, effect.accesses)
        else:
            finished[index] = index
    if pending:
        raise ValueError("program returns before actual completion of asynchronous work")
    return finished


def _problem(kernel, objects, banks, adapter, external_windows, private_ownership, max_calls, max_objects):
    if private_ownership is not True:
        raise StorageOptimizationUnavailable(
            "relocation requires invocation-private ownership disjoint from external buffers"
        )
    if type(adapter) is not StorageAdapter:
        raise ValueError("temporary allocation requires a selected typed adapter")
    if any(type(limit) is not int or limit <= 0 for limit in (max_calls, max_objects)):
        raise ValueError("temporary allocation requires positive explicit work budgets")
    if (
        type(objects) is not tuple
        or not objects
        or len(objects) > max_objects
        or any(type(obj) is not TemporaryObject for obj in objects)
        or len({obj.name for obj in objects}) != len(objects)
        or type(banks) is not tuple
        or not banks
        or any(type(bank) is not StorageBank for bank in banks)
        or len({bank.name for bank in banks}) != len(banks)
    ):
        raise ValueError("temporary allocation requires bounded unique objects and explicit banks")
    bank_map = {bank.name: bank for bank in banks}
    if any(
        bank.unit != "bytes"
        or not isinstance(bank.name, str)
        or not isinstance(bank.backing, str)
        or type(bank.capacity) is not int
        or type(bank.alignment) is not int
        or bank.capacity <= 0
        or bank.alignment <= 0
        for bank in banks
    ) or any(obj.bank not in bank_map for obj in objects):
        raise ValueError("temporary allocation requires explicit byte-addressed bank geometry")
    if type(external_windows) is not tuple or any(
        type(window) is not AddressWindow or window.space in bank_map for window in external_windows
    ):
        raise ValueError("external windows must be explicit and separate from relocated banks")
    for index, obj in enumerate(objects):
        if any(
            bank_map[obj.bank].backing == bank_map[other.bank].backing
            and obj.source_base < other.source_base + other.size
            and other.source_base < obj.source_base + obj.size
            for other in objects[:index]
        ):
            raise ValueError("overlapping source views must belong to one backing object")
    effects = _project(kernel, adapter, max_calls)
    finished = _completion(effects, bank_map)
    bindings, intervals = [], {}
    for index, effect in enumerate(effects):
        row = []
        for access in effect.accesses:
            if access.space not in bank_map:
                if not any(
                    window.space == access.space
                    and window.start <= access.start
                    and access.start + access.size <= window.start + window.size
                    for window in external_windows
                ):
                    raise StorageOptimizationUnavailable(
                        "actual external access lacks its declared fixed address window"
                    )
                row.append(None)
                continue
            matches = [
                obj
                for obj in objects
                if bank_map[access.space].backing == bank_map[obj.bank].backing
                and obj.source_base <= access.start
                and access.start + access.size <= obj.source_base + obj.size
            ]
            if len(matches) != 1:
                raise StorageOptimizationUnavailable("actual local access crosses or omits its private backing object")
            obj = matches[0]
            if not access.writes and obj.name in intervals and intervals[obj.name][0] == index:
                raise StorageOptimizationUnavailable(
                    "initializing instruction also reads uninitialized private backing"
                )
            if obj.name not in intervals:
                # Full initialization avoids relying on a fresh allocation's
                # unspecified bytes, including bytes reached through aliases.
                if not access.writes or access.start != obj.source_base or access.size != obj.size:
                    raise StorageOptimizationUnavailable(
                        "private backing needs complete initialization before any read or partial write"
                    )
                intervals[obj.name] = [index, finished[index]]
            intervals[obj.name][1] = max(intervals[obj.name][1], finished[index])
            row.append((obj.name, access.start - obj.source_base))
        bindings.append(tuple(row))
    if set(intervals) != {obj.name for obj in objects}:
        raise ValueError("declared private objects do not match actual used storage")
    # Each actual typed view constrains the shared base, even when the object
    # is assigned through another, larger view. Encode these base upper bounds
    # as derived byte-bank capacities for the existing allocator. Otherwise a
    # legal alternative could be missed after choosing a small address for an
    # unconstrained earlier object.
    placement_banks = list(banks)
    object_banks = {}
    for obj in objects:
        bank = bank_map[obj.bank]
        limit = bank.capacity - obj.size
        alignment = bank.alignment
        for effect, row in zip(effects, bindings, strict=True):
            for access, binding in zip(effect.accesses, row, strict=True):
                if binding is not None and binding[0] == obj.name:
                    limit = min(limit, bank_map[access.space].capacity - binding[1] - access.size)
                    alignment = lcm(alignment, bank_map[access.space].alignment)
        name = f"temporary_{len(placement_banks)}"
        while any(candidate.name == name for candidate in placement_banks):
            name += "_"
        derived = replace(bank, name=name, capacity=max(1, obj.size + limit), alignment=alignment)
        placement_banks.append(derived)
        object_banks[obj.name] = derived.name
    # Use the existing graph allocator. A completion event keeps every object
    # alive through the end of a call, including objects born in that call;
    # no undeclared same-instruction in-place reuse can slip between events.
    values, order, ids = [], [], {}
    for index in range(len(effects)):
        for obj in objects:
            if intervals[obj.name][0] == index:
                node = len(values)
                ids[obj.name] = node
                values.append(
                    Value(
                        node,
                        obj.name,
                        object_banks[obj.name],
                        obj.size,
                        (),
                        obj.name,
                        "instruction",
                        (AddressConstraint("aligned", "out", value=obj.alignment),),
                    )
                )
                order.append(node)
        touched = tuple(ids[name] for name, interval in intervals.items() if interval[1] == index)
        node = len(values)
        values.append(Value(node, "completion", banks[0].name, 0, touched, None, "event"))
        order.append(node)
    return (
        bank_map,
        effects,
        tuple(bindings),
        intervals,
        CandidateGraph(tuple(values), ()),
        tuple(order),
        ids,
        tuple(placement_banks),
    )


def verify_temporary_storage(
    original,
    realized,
    objects,
    banks,
    adapter,
    *,
    reservations=(),
    external_windows=(),
    private_ownership=False,
    max_calls=4096,
    max_objects=256,
):
    """Infer addresses from actual output; never accept the selected plan as proof."""
    _check_reservations(reservations)
    bank_map, effects, bindings, intervals, graph, order, ids, placement_banks = _problem(
        original, objects, banks, adapter, external_windows, private_ownership, max_calls, max_objects
    )
    if (realized.name, realized.args, realized.attrs) != (original.name, original.args, original.attrs):
        raise ValueError("storage relocation changed the original kernel boundary")
    actual = _project(realized, adapter, max_calls)
    if len(actual) != len(effects):
        raise ValueError("storage relocation changed the original instruction count")
    addresses = {value.id: 0 for value in graph.values if value.kind == "event"}
    for source, output, row in zip(effects, actual, bindings, strict=True):
        if replace(output, accesses=source.accesses) != source or len(output.accesses) != len(source.accesses):
            raise ValueError("realized instruction changed nonaddress semantics or completion")
        for before, after, binding in zip(source.accesses, output.accesses, row, strict=True):
            if replace(after, start=before.start) != before:
                raise ValueError("realized storage access changed space, extent or direction")
            if binding is None:
                if after != before:
                    raise ValueError("storage relocation changed a fixed external address")
                continue
            name, offset = binding
            node = ids[name]
            address = after.start - offset
            if addresses.setdefault(node, address) != address:
                raise ValueError("realized aliases disagree on their backing allocation")
            if after.start + after.size > bank_map[after.space].capacity:
                raise ValueError("realized access exceeds its actual typed bank view")
    valid, reason = check_assignment(graph, order, addresses, placement_banks, reservations=reservations)
    if not valid:
        raise ValueError("realized temporary assignment is invalid: " + reason)
    _completion(actual, bank_map)
    objects_by_name = {obj.name: obj for obj in objects}
    high_water, source_high_water = {}, {}
    for name, node in ids.items():
        obj = objects_by_name[name]
        backing = bank_map[obj.bank].backing
        high_water[backing] = max(high_water.get(backing, 0), addresses[node] + obj.size)
        source_high_water[backing] = max(source_high_water.get(backing, 0), obj.source_base + obj.size)
    physical_high_water = dict(high_water)
    for reservation in reservations:
        backing = bank_map[reservation.storage].backing
        physical_high_water[backing] = max(physical_high_water.get(backing, 0), reservation.start + reservation.extent)
    return {
        "schema": "merlin.temporary_storage.v1",
        "source_sha256": original.digest(),
        "realized_sha256": realized.digest(),
        "configuration_identity": adapter.configuration_identity,
        "numerical_policy_identity": adapter.numerical_policy_identity,
        "adapter_qualification_identity": adapter.qualification_identity,
        "banks": [bank.record() for bank in banks],
        "objects": [asdict(obj) for obj in objects],
        "external_windows": [asdict(window) for window in external_windows],
        "private_ownership": private_ownership,
        "checking_limits": {"max_calls": max_calls, "max_objects": max_objects},
        "reservations": [asdict(row) for row in reservations],
        "addresses": {name: addresses[node] for name, node in ids.items()},
        "lifetime_calls": intervals,
        "temporary_high_water_bytes": high_water,
        "physical_high_water_bytes": physical_high_water,
        "source_virtual_high_water_bytes": source_high_water,
        "static_commands": len(actual),
        "requested_read_bytes": sum(a.size for effect in actual for a in effect.accesses if not a.writes),
        "requested_write_bytes": sum(a.size for effect in actual for a in effect.accesses if a.writes),
        "machine_emission": "not_established",
        "execution_qualification": "not_established",
        "timing": "UNKNOWN",
    }


def allocate_temporary_storage(
    kernel,
    objects,
    banks,
    adapter,
    *,
    reservations=(),
    external_windows=(),
    private_ownership=False,
    max_calls=4096,
    max_objects=256,
    timeout_ms=5000,
) -> TemporaryStorageResult | None:
    """Allocate, consume addresses in actual calls, then independently verify.

    Infeasible storage returns None for the caller's existing legal route.
    Unsupported effects/ownership and unavailable checking diagnose precisely.
    This entry neither expands loops nor schedules away required completion.
    """
    model = dict(
        reservations=reservations,
        external_windows=external_windows,
        private_ownership=private_ownership,
        max_calls=max_calls,
        max_objects=max_objects,
    )
    _banks, effects, bindings, _intervals, graph, order, ids, placement_banks = _problem(
        kernel, objects, banks, adapter, external_windows, private_ownership, max_calls, max_objects
    )
    _check_reservations(reservations)
    if type(timeout_ms) is not int or timeout_ms <= 0:
        raise ValueError("temporary allocation requires a positive explicit solver budget")
    result = allocate(graph, order, placement_banks, reservations=reservations, timeout_ms=timeout_ms)
    if result.status == "infeasible_candidate":
        return None
    if result.status != "feasible":
        raise StorageOptimizationUnavailable(f"temporary allocation {result.status}: {result.reason}")
    calls = []
    for call, effect, row in zip(kernel.body, effects, bindings, strict=True):
        accesses = tuple(
            access if binding is None else replace(access, start=result.addresses[ids[binding[0]]] + binding[1])
            for access, binding in zip(effect.accesses, row, strict=True)
        )
        calls.append(call if accesses == effect.accesses else adapter.rewrite(call, accesses))
    realized = replace(kernel, body=tuple(calls))
    report = verify_temporary_storage(kernel, realized, objects, banks, adapter, **model)
    return TemporaryStorageResult(realized, report)
