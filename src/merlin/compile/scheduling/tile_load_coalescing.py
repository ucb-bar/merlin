"""Coalesce contiguous matrix columns into explicitly strided local tile rows."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TileLoadCoalescing:
    rows: int
    columns: int
    tile: int
    max_columns: int
    local_base: int
    block_stride: int
    capacity_rows: int

    def __post_init__(self):
        if (any(type(v) is not int or not 0 < v < 1 << 63 for v in (
                self.rows, self.columns, self.tile, self.max_columns, self.block_stride, self.capacity_rows))
                or type(self.local_base) is not int or not 0 <= self.local_base < 1 << 63
                or self.rows > self.tile or self.max_columns % self.tile
                or self.block_stride < self.rows
                or self.local_base + (self.columns - 1) // self.tile * self.block_stride + self.rows
                > self.capacity_rows):
            raise ValueError("coalesced loads need complete bounded nonoverlapping local tile rows")

    def commands(self):
        """Yield bounded-width logical source start, width and local base."""
        for column in range(0, self.columns, self.max_columns):
            local = self.local_base + column // self.tile * self.block_stride
            yield column, min(self.max_columns, self.columns - column), local

    def location(self, row, column):
        if (any(type(v) is not int for v in (row, column))
                or not 0 <= row < self.rows or not 0 <= column < self.columns):
            raise ValueError("load element leaves its complete logical rectangle")
        return self.local_base + column // self.tile * self.block_stride + row, column % self.tile
