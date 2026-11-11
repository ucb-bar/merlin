"""Bounded residency of an immutable operand across repeated consumers.

The caller establishes invariance over the reuse scope and supplies physical
capacity and code expansion limits. This plan changes operand transport only:
it grants no reordering, arithmetic change, or alias with a writable buffer.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class OperandReuse:
    tile_count: int
    tile_bytes: int
    consumers: int

    @property
    def resident_bytes(self) -> int:
        return self.tile_count * self.tile_bytes

    @property
    def avoided_read_bytes(self) -> int:
        return self.resident_bytes * (self.consumers - 1)


def plan_operand_reuse(
    *, tile_count: int, tile_bytes: int, consumers: int, capacity_bytes: int, max_tiles: int
) -> OperandReuse | None:
    """Retain one complete invariant operand when reuse fits both limits.

    Partial residency requires a separate ordered panel schedule. Refusing
    this candidate does not prove that another transport schedule cannot fit.
    Target allocation, initialization and memory proofs remain mandatory.
    """
    for name, value in (("tile_count", tile_count), ("tile_bytes", tile_bytes), ("consumers", consumers)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name, value in (("capacity_bytes", capacity_bytes), ("max_tiles", max_tiles)):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    plan = OperandReuse(tile_count, tile_bytes, consumers)
    if consumers < 2 or tile_count > max_tiles or plan.resident_bytes > capacity_bytes:
        return None
    return plan
