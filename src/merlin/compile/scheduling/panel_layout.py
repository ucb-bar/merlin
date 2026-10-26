"""Column-panel weight layout ``[ceil(N/w)][K][w]`` for a provider-declared width.

A provider declares the panel width ``w`` from its own target facts (for
example the widest single movement payload); this module never chooses one.
Packing is a permutation of the source ``[K][N]`` weight plus explicit zero
lanes in a trailing partial panel. Those lanes are never a source element, so a
consumer must prove that every read stays inside one panel and the source
columns, and that every source element is placed exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PanelLayout:
    k: int
    n: int
    width: int

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in (self.k, self.n, self.width)):
            raise ValueError("panel layout extents and declared width must be positive integers")

    @property
    def panels(self) -> int:
        return -(-self.n // self.width)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.panels, self.k, self.width

    @property
    def row_pitch(self) -> int:
        """Elements between consecutive K rows of one panel."""
        return self.width

    def offset(self, row: int, col: int) -> int:
        """Element offset of source ``(row, col)`` in the packed buffer."""
        if not (0 <= row < self.k and 0 <= col < self.n):
            raise ValueError("panel coordinate outside the source weight")
        panel, lane = divmod(col, self.width)
        return (panel * self.k + row) * self.width + lane

    def source(self, offset: int, cols: int) -> tuple[int, int]:
        """Invert ``offset`` for a ``cols``-wide row read; refuse padding or panel crossing."""
        if type(offset) is not int or type(cols) is not int or offset < 0 or cols <= 0:
            raise ValueError("panel read needs a non-negative offset and positive width")
        panel, rest = divmod(offset, self.k * self.width)
        row, lane = divmod(rest, self.width)
        col = panel * self.width + lane
        if panel >= self.panels or lane + cols > self.width or col + cols > self.n:
            raise ValueError("panel read crosses a panel boundary or reads padding lanes")
        return row, col

    def pack(self, weight) -> np.ndarray:
        weight = np.asarray(weight)
        if weight.shape != (self.k, self.n):
            raise ValueError("source weight differs from the panel layout extents")
        packed = np.zeros(self.shape, dtype=weight.dtype)
        for panel in range(self.panels):
            start = panel * self.width
            stop = min(self.n, start + self.width)
            packed[panel, :, : stop - start] = weight[:, start:stop]
        return packed

    def unpack(self, packed) -> np.ndarray:
        """Exact inverse on source lanes; nonzero padding lanes are refused."""
        packed = np.asarray(packed)
        if packed.shape != self.shape:
            raise ValueError("packed buffer differs from the declared panel shape")
        tail = self.n - (self.panels - 1) * self.width
        if np.any(packed[-1, :, tail:]):
            raise ValueError("packed padding lanes are not zero")
        return np.concatenate(
            [packed[p, :, : min(self.width, self.n - p * self.width)] for p in range(self.panels)], axis=1
        )

    def record(self) -> dict:
        return {
            "schema": "merlin.panel_layout.v1",
            "source": [self.k, self.n],
            "packed": list(self.shape),
            "panel_width": self.width,
            "padding": "zero lanes in the trailing partial panel; never read as source",
        }
