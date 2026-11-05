"""Allocate temporary and persistent buffers over explicit completion events.

Targets supply storage geometry and access/lifetime events. This module does
not infer DMA completion from issue order. Views share a root allocation and
extend its lifetime; all offsets and extents use the bank's declared unit.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import lcm
from time import monotonic

from .allocate import Reservation, StorageBank


@dataclass(frozen=True)
class Buffer:
    name: str
    storage: str
    extent: int
    first_event: int
    last_event: int
    alignment: int = 1
    fixed_start: int | None = None
    view_of: str | None = None
    view_offset: int = 0
    persistent: bool = False

    def __post_init__(self) -> None:
        integers = (self.extent, self.first_event, self.last_event, self.alignment, self.view_offset)
        if (
            not isinstance(self.name, str)
            or not self.name
            or not isinstance(self.storage, str)
            or not self.storage
            or any(type(n) is not int for n in integers)
            or self.extent <= 0
            or self.first_event < 0
            or self.last_event < self.first_event
            or self.alignment <= 0
            or self.view_offset < 0
            or type(self.persistent) is not bool
            or self.fixed_start is not None
            and (type(self.fixed_start) is not int or self.fixed_start < 0)
            or self.view_of is None
            and self.view_offset != 0
        ):
            raise ValueError("invalid buffer geometry, events or view")


@dataclass(frozen=True)
class BufferAllocation:
    status: str
    addresses: tuple[tuple[str, int], ...] = ()
    reason: str = ""


def _normalize(buffers: tuple[Buffer, ...], banks: tuple[StorageBank, ...]):
    storage = {bank.name: bank for bank in banks}
    items = {buffer.name: buffer for buffer in buffers}
    if len(storage) != len(banks) or len(items) != len(buffers):
        raise ValueError("duplicate storage or buffer names")
    for bank in banks:
        for other in banks:
            if bank.backing == other.backing and (bank.unit != other.unit or bank.capacity != other.capacity):
                raise ValueError("aliased storage views require the same address unit and capacity")
    if any(item.storage not in storage for item in buffers):
        raise ValueError("buffer names an unknown storage")
    roots: dict[str, tuple[str, int]] = {}

    def root(name: str, visiting: frozenset[str] = frozenset()) -> tuple[str, int]:
        if name in visiting or name not in items:
            raise ValueError("cyclic or missing buffer view")
        if name in roots:
            return roots[name]
        item = items[name]
        if item.storage not in storage:
            raise ValueError("buffer names an unknown storage")
        if item.view_of is None:
            result = (name, 0)
        else:
            if item.view_of not in items:
                raise ValueError("missing buffer view parent")
            parent = items[item.view_of]
            if (
                storage[item.storage].backing != storage[parent.storage].backing
                or item.view_offset + item.extent > parent.extent
            ):
                raise ValueError("view escapes its parent allocation")
            base, offset = root(item.view_of, visiting | {name})
            result = (base, offset + item.view_offset)
        roots[name] = result
        return result

    for name in items:
        root(name)
    lifetimes = {
        name: [item.first_event, item.last_event, item.persistent]
        for name, item in items.items()
        if item.view_of is None
    }
    for name, item in items.items():
        life = lifetimes[roots[name][0]]
        life[0] = min(life[0], item.first_event)
        life[1] = max(life[1], item.last_event)
        life[2] |= item.persistent
    return storage, items, roots, lifetimes


def check_buffer_allocation(
    buffers: tuple[Buffer, ...],
    banks: tuple[StorageBank, ...],
    addresses: Mapping[str, int],
    *,
    reservations: tuple[Reservation, ...] = (),
) -> None:
    """Check intervals/aliases directly, without evaluating the solver formula."""
    storage, items, roots, life = _normalize(buffers, banks)
    if set(addresses) != set(items):
        raise ValueError("buffer assignment has missing or extra names")
    for reservation in reservations:
        if reservation.storage not in storage:
            raise ValueError("reservation names an unknown storage")
        if reservation.start + reservation.extent > storage[reservation.storage].capacity:
            raise ValueError("reservation exceeds storage capacity")
    for name, item in items.items():
        address, bank = addresses[name], storage[item.storage]
        if (
            type(address) is not int
            or address < 0
            or address + item.extent > bank.capacity
            or address % lcm(bank.alignment, item.alignment)
            or item.fixed_start is not None
            and item.fixed_start != address
        ):
            raise ValueError("buffer assignment violates bounds, alignment or fixed placement")
        base, offset = roots[name]
        if address != addresses[base] + offset:
            raise ValueError("buffer view disagrees with its root")
    root_items = [item for item in buffers if item.view_of is None]
    for index, item in enumerate(root_items):
        address, bank = addresses[item.name], storage[item.storage]
        for reservation in reservations:
            if (
                bank.backing == storage[reservation.storage].backing
                and address < reservation.start + reservation.extent
                and reservation.start < address + item.extent
            ):
                raise ValueError("buffer overlaps reserved storage")
        for other in root_items[:index]:
            a, b = life[item.name], life[other.name]
            simultaneous = a[2] or b[2] or a[0] <= b[1] and b[0] <= a[1]
            if (
                simultaneous
                and bank.backing == storage[other.storage].backing
                and address < addresses[other.name] + other.extent
                and addresses[other.name] < address + item.extent
            ):
                raise ValueError("live buffer allocations overlap")


def first_fit_buffer_assignment(
    buffers: tuple[Buffer, ...],
    banks: tuple[StorageBank, ...],
    *,
    reservations: tuple[Reservation, ...] = (),
) -> dict[str, int] | None:
    """Propose a deterministic placement; failure is inconclusive.

    Fixed buffers are placed first, including future results. Other roots
    follow decreasing last-use order, avoiding every simultaneously live
    placed interval. This leaves future output slots available to shorter
    temporaries instead of consuming the unrestricted space too early.
    The independent checker validates the proposal. Callers can pin a valid
    proposal in ``allocate_buffers`` to check the full solver formula too.
    """
    storage, items, roots, life = _normalize(buffers, banks)
    root_items = [item for item in buffers if item.view_of is None]
    placed = {item.name: item.fixed_start for item in root_items if item.fixed_start is not None}
    order = {item.name: index for index, item in enumerate(buffers)}
    for item in sorted(root_items, key=lambda value: (-life[value.name][1], life[value.name][0], order[value.name])):
        if item.name in placed:
            continue
        bank = storage[item.storage]
        blocked = [(r.start, r.start + r.extent) for r in reservations
                   if storage[r.storage].backing == bank.backing]
        for other in root_items:
            a, b = life[item.name], life[other.name]
            if other.name in placed and storage[other.storage].backing == bank.backing and (
                a[2] or b[2] or a[0] <= b[1] and b[0] <= a[1]
            ):
                blocked.append((placed[other.name], placed[other.name] + other.extent))
        alignment = lcm(bank.alignment, item.alignment)
        address = 0
        for begin, end in sorted(blocked):
            if address + item.extent <= begin:
                break
            if address < end and begin < address + item.extent:
                address = (end + alignment - 1) // alignment * alignment
        if address + item.extent > bank.capacity:
            return None
        placed[item.name] = address
    addresses = {name: placed[base] + offset for name, (base, offset) in roots.items()}
    try:
        check_buffer_allocation(buffers, banks, addresses, reservations=reservations)
    except ValueError:
        return None
    return addresses


def allocate_buffers(
    buffers: tuple[Buffer, ...],
    banks: tuple[StorageBank, ...],
    *,
    reservations: tuple[Reservation, ...] = (),
    timeout_ms: int = 1000,
) -> BufferAllocation:
    """Solve this finite interval-placement problem; UNKNOWN is not infeasibility.

    Persistent roots reserve storage for the entire invocation. The result is
    a placement component, not executable target lowering or hardware evidence.
    """
    import z3

    if type(timeout_ms) is not int or timeout_ms <= 0:
        raise ValueError("positive solver timeout required")
    deadline = monotonic() + timeout_ms / 1000
    storage, items, roots, life = _normalize(buffers, banks)
    for reservation in reservations:
        if (
            reservation.storage not in storage
            or reservation.start + reservation.extent > storage[reservation.storage].capacity
        ):
            raise ValueError("reservation has invalid storage bounds")
    solver = z3.Solver()
    solver.set(random_seed=0)

    def query():
        remaining_ms = int((deadline - monotonic()) * 1000)
        if remaining_ms <= 0:
            return z3.unknown, "buffer allocation time budget exhausted"
        solver.set(timeout=remaining_ms)
        status = solver.check()
        return status, solver.reason_unknown() if status == z3.unknown else ""

    variables = {name: z3.Int(f"buffer_{index}") for index, name in enumerate(items)}
    for name, item in items.items():
        bank, variable = storage[item.storage], variables[name]
        solver.add(
            variable >= 0, variable + item.extent <= bank.capacity, variable % lcm(bank.alignment, item.alignment) == 0
        )
        if item.fixed_start is not None:
            solver.add(variable == item.fixed_start)
        base, offset = roots[name]
        if name != base:
            solver.add(variable == variables[base] + offset)
    root_items = [item for item in buffers if item.view_of is None]
    for index, item in enumerate(root_items):
        variable, bank = variables[item.name], storage[item.storage]
        for reservation in reservations:
            if bank.backing == storage[reservation.storage].backing:
                solver.add(
                    z3.Or(
                        variable + item.extent <= reservation.start, variable >= reservation.start + reservation.extent
                    )
                )
        for other in root_items[:index]:
            a, b = life[item.name], life[other.name]
            if bank.backing == storage[other.storage].backing and (a[2] or b[2] or a[0] <= b[1] and b[0] <= a[1]):
                solver.add(
                    z3.Or(
                        variable + item.extent <= variables[other.name],
                        variables[other.name] + other.extent <= variable,
                    )
                )
    result, reason = query()
    if result == z3.unknown:
        return BufferAllocation("unknown", reason=reason)
    if result == z3.unsat:
        return BufferAllocation("unsat", reason="bounded buffer-placement constraints are unsatisfiable")
    # Lexicographic minimization with explicit bounded queries avoids relying on
    # solver model order. A timeout in any query remains inconclusive.
    for name in items:
        if items[name].fixed_start is not None:
            continue  # Already constrained to its unique value above.
        variable = variables[name]
        lo, hi = 0, storage[items[name].storage].capacity - items[name].extent
        while lo < hi:
            mid = (lo + hi) // 2
            solver.push()
            solver.add(variable <= mid)
            decision, reason = query()
            solver.pop()
            if decision == z3.unknown:
                return BufferAllocation("unknown", reason=reason)
            if decision == z3.sat:
                hi = mid
            else:
                lo = mid + 1
        solver.add(variable == lo)
    result, reason = query()
    if result != z3.sat:
        return BufferAllocation("unknown", reason=reason or "final placement was not satisfiable")
    model = solver.model()
    addresses = {name: model.eval(variable).as_long() for name, variable in variables.items()}
    check_buffer_allocation(buffers, banks, addresses, reservations=reservations)
    return BufferAllocation("sat", tuple(addresses.items()))
