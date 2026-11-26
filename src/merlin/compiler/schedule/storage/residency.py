"""Plan explicit, disjoint byte residency without discovering target facts or doing I/O.

Lifetimes describe ownership; they do not authorize aliasing. The conservative
planner keeps every allocation disjoint, including allocations with different
lifetimes. Providers supply the arena, DMA aperture, image reservations and sizes.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from hashlib import sha256


ROLES = frozenset({"immutable_parameter", "mutable_state", "staging", "scratch", "image"})
LOAD_PROTOCOLS = frozenset({"once_at_load", "per_phase", "per_call"})


def _integer(name: str, value: int, *, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _declaration(name, role, nbytes, alignment, lifetime, load_protocol):
    if not isinstance(name, str) or not name.strip():
        raise ValueError("residency name must be a nonempty string")
    if not isinstance(role, str) or role not in ROLES:
        raise ValueError(f"unsupported residency role: {role!r}")
    _integer("bytes", nbytes)
    _integer("alignment", alignment, minimum=1)
    if not isinstance(lifetime, str) or not lifetime.strip():
        raise ValueError("residency lifetime must be explicit")
    if not isinstance(load_protocol, str) or load_protocol not in LOAD_PROTOCOLS:
        raise ValueError(f"unsupported residency load protocol: {load_protocol!r}")


@dataclass(frozen=True)
class ResidencyRequest:
    name: str
    role: str
    bytes: int
    alignment: int
    lifetime: str
    load_protocol: str

    def __post_init__(self):
        _declaration(self.name, self.role, self.bytes, self.alignment, self.lifetime, self.load_protocol)


@dataclass(frozen=True)
class ResidencyEntry:
    name: str
    role: str
    bytes: int
    alignment: int
    address: int
    lifetime: str
    load_protocol: str

    def __post_init__(self):
        _declaration(self.name, self.role, self.bytes, self.alignment, self.lifetime, self.load_protocol)
        _integer("address", self.address)
        if self.address % self.alignment:
            raise ValueError(f"residency entry {self.name!r} has a misaligned address")

    @property
    def end(self):
        return self.address + self.bytes

    def record(self):
        return asdict(self)


@dataclass(frozen=True)
class ReservedInterval:
    """A fixed occupied interval, possibly outside the allocation arena.

    Images use the defaults; other reservations declare their actual role and
    lifecycle. All reservations appear in the resulting table and identity.
    """

    name: str
    address: int
    bytes: int
    alignment: int = 1
    role: str = "image"
    lifetime: str = "application"
    load_protocol: str = "once_at_load"

    def entry(self):
        return ResidencyEntry(**asdict(self))

    def __post_init__(self):
        self.entry()


class ResidencyOverflow(ValueError):
    """The exact high-water extent of the deterministic plan exceeds capacity."""

    def __init__(self, required_bytes: int, capacity_bytes: int):
        self.required_bytes = required_bytes
        self.capacity_bytes = capacity_bytes
        self.overflow_bytes = required_bytes - capacity_bytes
        super().__init__(
            f"residency requires {required_bytes} bytes, capacity is {capacity_bytes} bytes: "
            f"overflow by {self.overflow_bytes} bytes"
        )


@dataclass(frozen=True)
class ResidencyPlan:
    base: int
    capacity: int
    dma_base: int
    dma_capacity: int
    entries: tuple[ResidencyEntry, ...]
    reserved_names: tuple[str, ...]
    required_bytes: int

    def record(self):
        return {
            "schema": "merlin.residency.v1",
            "arena": {"base": self.base, "capacity": self.capacity},
            "dma_aperture": {"base": self.dma_base, "capacity": self.dma_capacity},
            "required_bytes": self.required_bytes,
            "reserved_names": list(self.reserved_names),
            "entries": [entry.record() for entry in self.entries],
        }

    def to_json(self):
        return json.dumps(self.record(), sort_keys=True, separators=(",", ":"))

    def digest(self):
        return sha256(self.to_json().encode("utf-8")).hexdigest()

    def entry(self, name: str):
        for entry in self.entries:
            if entry.name == name:
                return entry
        raise KeyError(name)


def _overlap(left: ResidencyEntry, right: ResidencyEntry):
    return bool(left.bytes and right.bytes and left.address < right.end and right.address < left.end)


def plan_residency(
    requests: Iterable[ResidencyRequest | Mapping],
    *,
    base: int,
    capacity: int,
    dma_base: int,
    dma_capacity: int,
    reserved: Iterable[ReservedInterval | Mapping] = (),
) -> ResidencyPlan:
    """Allocate requests by name with deterministic aligned first-fit placement.

    Capacity and the DMA aperture are explicit byte counts, with half-open
    intervals. The entire declared arena must fit the DMA aperture. Reserved
    intervals may describe external images; allocations avoid every reserved
    interval, even when a reservation straddles the arena boundary. Zero-byte
    declarations retain their table entries but occupy no addresses.

    Overflow includes alignment and reserved holes; it describes this plan,
    without claiming an optimal packing or permitting lifetime-based aliases.
    """
    for name, value in (("base", base), ("capacity", capacity), ("dma_base", dma_base), ("dma_capacity", dma_capacity)):
        _integer(name, value)
    if base < dma_base or base + capacity > dma_base + dma_capacity:
        raise ValueError("declared residency arena lies outside the DMA aperture")
    allocations = tuple(
        row if isinstance(row, ResidencyRequest) else ResidencyRequest(**row) for row in requests
    )
    reservations = tuple(
        (row if isinstance(row, ReservedInterval) else ReservedInterval(**row)).entry() for row in reserved
    )
    names = [row.name for row in (*allocations, *reservations)]
    if len(names) != len(set(names)):
        raise ValueError("residency declarations need distinct names")
    occupied = sorted(reservations, key=lambda row: (row.address, row.name))
    for index, row in enumerate(occupied):
        for previous in occupied[:index]:
            if _overlap(previous, row):
                raise ValueError(f"reserved intervals {previous.name!r} and {row.name!r} overlap")
    entries = list(reservations)
    required = max(
        (row.end - base for row in reservations
         if row.bytes and row.address < base + capacity and base < row.end),
        default=0,
    )
    for request in sorted(allocations, key=lambda row: row.name):
        address = ((base + request.alignment - 1) // request.alignment) * request.alignment
        for blocker in occupied:
            if request.bytes and blocker.bytes and address < blocker.end and blocker.address < address + request.bytes:
                address = ((blocker.end + request.alignment - 1) // request.alignment) * request.alignment
        entry = ResidencyEntry(**asdict(request), address=address)
        entries.append(entry)
        if request.bytes:
            required = max(required, entry.end - base)
            occupied.append(entry)
            occupied.sort(key=lambda row: (row.address, row.name))
    if required > capacity:
        raise ResidencyOverflow(required, capacity)
    return ResidencyPlan(
        base, capacity, dma_base, dma_capacity,
        tuple(sorted(entries, key=lambda row: row.name)),
        tuple(sorted(row.name for row in reservations)), required,
    )
