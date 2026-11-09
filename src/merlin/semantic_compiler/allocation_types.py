"""Shared explicit allocation geometry and graph records, without solver imports."""

from __future__ import annotations

from dataclasses import dataclass

from .rules import AddressConstraint


@dataclass(frozen=True)
class StorageBank:
    name: str
    backing: str
    capacity: int
    unit: str
    alignment: int = 1
    # Canonical backing coordinates; capacity is the exclusive upper bound.
    # A typed alias may expose only a suffix of another view's address window.
    start: int = 0

    def __post_init__(self) -> None:
        if (
            not self.name
            or not self.backing
            or not self.unit
            or any(type(x) is not int for x in (self.capacity, self.alignment, self.start))
            or self.capacity <= 0
            or self.alignment <= 0
            or not 0 <= self.start <= self.capacity
        ):
            raise ValueError("storage bank needs positive geometry and explicit units")

    def record(self) -> dict[str, str | int]:
        row = vars(self).copy()
        if not self.start:
            row.pop("start")  # Preserve existing zero-origin serialized inputs.
        return row

    @classmethod
    def from_record(cls, row: dict[str, str | int]) -> StorageBank:
        required = {"name", "backing", "capacity", "unit", "alignment"}
        if set(row) not in (required, required | {"start"}):
            raise ValueError("storage bank has missing or unknown fields")
        return cls(**row)


@dataclass(frozen=True)
class Reservation:
    """Physical interval unavailable to every typed view of one backing store.

    The target contract names the storage view and supplies the interval in its
    explicit address unit. Reservations last for this entire candidate. Shorter
    lifetimes require a qualified temporal model, not an assumed issue order.
    """

    storage: str
    start: int
    extent: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.storage, str)
            or not self.storage
            or not isinstance(self.start, int)
            or isinstance(self.start, bool)
            or self.start < 0
            or not isinstance(self.extent, int)
            or isinstance(self.extent, bool)
            or self.extent <= 0
        ):
            raise ValueError("reservation needs a storage, nonnegative start and positive extent")


@dataclass(frozen=True)
class Value:
    id: int
    symbol: str
    storage: str
    extent: int
    children: tuple[int, ...]
    source_node: str | None
    kind: str
    validity: tuple[AddressConstraint, ...] = ()
    input_read_offsets: tuple[int, ...] = ()
    completion_offset: int = 0
    in_place_inputs: tuple[int, ...] = ()
    # A source input is immutable across the kernel unless its boundary says reusable.
    preserve_input: bool = True
    instruction_instance: tuple | None = None
    projection_of: int | None = None
    value_preserving_copy: bool = False

    def __post_init__(self) -> None:
        if type(self.preserve_input) is not bool or (self.kind != "input" and not self.preserve_input):
            raise ValueError("input retention must be a boolean on input values")
        if self.kind == "event" and self.extent != 0:
            raise ValueError("a result-free instruction event consumes no storage")
        if (self.kind == "projection") != (self.projection_of is not None):
            raise ValueError("multi-result projection needs exactly one instruction owner")
        if type(self.value_preserving_copy) is not bool or (
            self.value_preserving_copy and (self.kind != "instruction" or len(self.children) != 1)
        ):
            raise ValueError("value-preserving copy needs one instruction operand")

    def read_offset(self, index: int) -> int:
        return self.input_read_offsets[index] if self.input_read_offsets else 0


@dataclass(frozen=True)
class CandidateGraph:
    values: tuple[Value, ...]
    outputs: tuple[int, ...]

    def value(self, value_id: int) -> Value:
        return self.values[value_id]


@dataclass(frozen=True)
class AllocationResult:
    status: str
    order: tuple[int, ...]
    addresses: dict[int, int]
    reason: str = ""
    issue_times: tuple[tuple[int, int], ...] = ()
