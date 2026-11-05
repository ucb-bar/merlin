"""Source-checked column views for sibling contractions of one immutable input.

The virtual weight/output matrices are index spaces, never allocated tensors.
Each rectangle must stay inside one original member. The caller still owns
physical pointer aliases, instruction effects, per-result scalar readouts and
completion; this mapping establishes none of those target permissions.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd

from .shared_input_groups import SharedInputGroup


@dataclass(frozen=True)
class ProjectionRectangle:
    member: int
    row: int
    column: int
    rows: int
    columns: int
    row_stride_bytes: int
    byte_offset: int


@dataclass(frozen=True)
class SharedProjectionViews:
    source: SharedInputGroup
    weight_element_bytes: int
    output_element_bytes: tuple[int, ...]
    tile_columns: int

    def __post_init__(self):
        if type(self.source) is not SharedInputGroup:
            raise ValueError("projection views need an original shared-input source group")
        widths = self.source.widths
        if (
            not 2 <= len(widths) <= 128
            or any(type(v) is not int or v <= 0 for v in (self.source.m, self.source.k, *widths))
            or type(self.weight_element_bytes) is not int
            or self.weight_element_bytes != 1
            or type(self.tile_columns) is not int
            or self.tile_columns <= 0
            or any(width % self.tile_columns for width in widths)
            or sum(widths) >= 1 << 63
            or self.source.k * sum(widths) >= 1 << 63
            or self.source.m * sum(widths) * 4 >= 1 << 63
            or type(self.output_element_bytes) is not tuple
            or len(self.output_element_bytes) != len(widths)
            or any(type(v) is not int or v <= 0 for v in self.output_element_bytes)
            or any(self.source.k * width * self.weight_element_bytes >= 1 << 63 for width in widths)
            or any(self.source.m * width * unit >= 1 << 63 for width, unit in zip(widths, self.output_element_bytes))
        ):
            raise ValueError("projection views need bounded whole-column tiles and every original output width")

    @property
    def columns(self):
        return sum(self.source.widths)

    @property
    def boundaries(self):
        result = [0]
        for width in self.source.widths:
            result.append(result[-1] + width)
        return tuple(result)

    def verify_current(self, module):
        self.__post_init__()
        self.source.verify_current(module)

    def aligned_panel_tiles(self, maximum):
        """Panel widths that divide every original member, in increasing order.

        A selected panel and every transfer inside it must obey these member
        boundaries. This is not permission to widen an independently encoded
        physical transfer or to fuse scalar readouts across a boundary.
        """
        self.__post_init__()
        if type(maximum) is not int or not 1 <= maximum <= 4096:
            raise ValueError("explicit finite panel search bound required")
        common = 0
        for width in self.source.widths:
            common = gcd(common, width // self.tile_columns)
        return tuple(n for n in range(1, min(common, maximum) + 1) if common % n == 0)

    def rectangle(self, kind, row, column, rows, columns):
        """Map an actual complete rectangle to exactly one original tensor.

        The source and destination element widths are independent. A crossing
        rectangle refuses instead of silently assuming packed adjacent arrays.
        """
        self.__post_init__()
        if kind not in ("weight", "output") or any(type(v) is not int for v in (row, column, rows, columns)):
            raise ValueError("projection rectangle needs exact integer coordinates and a declared role")
        height = self.source.k if kind == "weight" else self.source.m
        if not (0 <= row < height and 0 < rows <= height - row and 0 <= column < self.columns and columns > 0):
            raise ValueError("projection rectangle exceeds the original logical domain")
        starts = self.boundaries
        member = next(i for i in range(len(starts) - 1) if starts[i] <= column < starts[i + 1])
        local = column - starts[member]
        width = self.source.widths[member]
        if columns > width - local:
            raise ValueError("projection rectangle crosses an original tensor boundary")
        unit = self.weight_element_bytes if kind == "weight" else self.output_element_bytes[member]
        return ProjectionRectangle(member, row, local, rows, columns, width * unit, (row * width + local) * unit)

    def report(self):
        self.__post_init__()
        return {
            "schema": "merlin.shared_projection_views.v1",
            "source": self.source.record(),
            "shape_m_k_n": [self.source.m, self.source.k, self.columns],
            "column_boundaries": list(self.boundaries),
            "weight_element_bytes": self.weight_element_bytes,
            "output_element_bytes": list(self.output_element_bytes),
            "tile_columns": self.tile_columns,
            "source_weights_or_outputs_concatenated": False,
            "host_payload_copy_bytes": 0,
            "scalar_readouts": "separate original per-member obligations",
            "physical_alias_and_completion": "caller obligations; no target grant",
        }


def plan_shared_projection_views(module, group, *, weight_element_bytes, output_element_bytes, tile_columns):
    """Bind a current source group to explicit per-tensor byte views."""
    result = SharedProjectionViews(group, weight_element_bytes, output_element_bytes, tile_columns)
    result.verify_current(module)
    return result
