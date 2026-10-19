"""Plan and emit private dense zero padding for an explicit consumer contract.

The consumer may implement zero extension on selected axes itself. Core owns
the remaining materialization; providers still prove representation, accesses,
completion and numerical equivalence of their implicit-zero implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod


@dataclass(frozen=True)
class DenseZeroPadding:
    logical_shape: tuple[int, int]
    physical_shape: tuple[int, int]
    implicit_zero_axes: tuple[int, ...] = ()

    def __post_init__(self):
        if (
            any(
                not isinstance(shape, tuple) or len(shape) != 2 or any(type(d) is not int or d <= 0 for d in shape)
                for shape in (self.logical_shape, self.physical_shape)
            )
            or any(a > b for a, b in zip(self.logical_shape, self.physical_shape, strict=True))
            or not isinstance(self.implicit_zero_axes, tuple)
            or any(type(axis) is not int or axis not in (0, 1) for axis in self.implicit_zero_axes)
            or len(set(self.implicit_zero_axes)) != len(self.implicit_zero_axes)
        ):
            raise ValueError("dense padding needs bounded static shapes and distinct valid implicit axes")

    @property
    def materialized_shape(self) -> tuple[int, int]:
        return tuple(
            self.logical_shape[axis] if axis in self.implicit_zero_axes else self.physical_shape[axis]
            for axis in (0, 1)
        )

    @property
    def needs_copy(self) -> bool:
        return self.materialized_shape != self.logical_shape

    @property
    def elements(self) -> int:
        return prod(self.materialized_shape) if self.needs_copy else 0


def emit_dense_zero_padding(plan: DenseZeroPadding, *, source: str, destination: str) -> str:
    """Emit a row-major zero/copy loop into caller-owned, disjoint storage.

    Both spans must remain accessible for their complete declared extents.
    No buffer is materialized when the consumer already handles every pad.
    This emits ordinary scalar assignment, preserving the input element type.
    """
    for name in (source, destination):
        if (
            not name
            or not name.isascii()
            or not (name[0].isalpha() or name[0] == "_")
            or any(not (c.isalnum() or c == "_") for c in name)
        ):
            raise ValueError("dense padding needs safe distinct C buffer names")
    if source == destination:
        raise ValueError("dense padding needs safe distinct C buffer names")
    if not plan.needs_copy:
        return ""
    rows, cols = plan.logical_shape
    _, stride = plan.materialized_shape
    return f"""  for (size_t i = 0; i < {plan.elements}u; ++i) {destination}[i] = 0;
  for (size_t row = 0; row < {rows}u; ++row)
    for (size_t col = 0; col < {cols}u; ++col)
      {destination}[row * {stride}u + col] = {source}[row * {cols}u + col];
"""
