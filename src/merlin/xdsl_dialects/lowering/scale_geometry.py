"""Follow scale variation and tensor axes through exact view geometry."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from merlin.common import mlir_query as mq

from .contraction_coverage import _product as _product


def _granularity(varies: set[int] | None, axes: tuple[set[int], set[int]] | None) -> str | None:
    if varies is None:
        return None
    if not varies:
        return "tensor"
    if axes is None:
        return None
    row, column = axes
    if varies <= column:
        return "column"
    if varies <= row:
        return "row"
    if varies <= (row | column):
        return "rank1"
    return None


def _join(scales: Sequence[str | None]) -> str | None:
    """The one granularity that expresses every scale of a group; ``None`` if any is unknown.

    An unknown scale is not a per-tensor scale. Treating it as one admits a group whose
    arithmetic the hardware cannot hold, which is the defect this pass exists to name.
    """
    if not scales or any(scale is None for scale in scales):
        return None if scales else "tensor"
    kinds = set(scales) - {"tensor"}
    if not kinds:
        return "tensor"
    if kinds == {"row", "column"} or "rank1" in kinds:
        return "rank1" if kinds <= {"row", "column", "rank1"} else None
    return next(iter(kinds)) if len(kinds) == 1 else None


def _axis_through(adapters: Sequence[Any], axis: int, source_shape: Sequence[int]) -> int | None:
    """Where tensor axis ``axis`` sits after the views and permutations in ``adapters``.

    ``adapters`` run from the root back toward the dequantize, so they are applied in reverse. A
    reshape keeps row-major order, so an axis is identified across it by its extent and by how many
    elements one step along it spans; it may vanish into a flattened dim and reappear, and is
    resolved only where an index is needed. A reshape that splits or merges the axis itself loses
    it, and so does a pad that changes its extent: the answer is then ``None``.
    """
    from .compute_groups import _VIEW_OPS

    shape = [int(v) for v in source_shape]
    extent, inner = shape[axis], _product(shape[axis + 1 :])

    def resolve(current: Sequence[int]) -> int | None:
        for index, value in enumerate(current):
            if int(value) == extent and _product(current[index + 1 :]) == inner:
                return index
        return None

    for op in reversed(list(adapters)):
        out_shape = [int(v) for v in mq.type_shape_dtype(op.results[0].type)[0]]
        name = mq.op_name(op)
        if name in _VIEW_OPS:
            if _product(out_shape) != _product(shape):
                return None
        else:
            here = resolve(shape)
            if here is None:
                return None
            if name == "linalg.transpose":
                permutation = _int_array(op, "permutation") or []
                if here not in permutation:
                    return None
                here = permutation.index(here)
            elif len(out_shape) != len(shape) or out_shape[here] != extent:
                return None
            inner = _product(out_shape[here + 1 :])
        shape = out_shape
    return resolve(shape)


def _int_array(op, key: str) -> list[int] | None:
    for table in mq._attr_tables(op):
        raw = table.get(key)
        if raw is None:
            continue
        values = getattr(raw, "get_values", None)
        if callable(values):
            return [int(v) for v in values()]
        data = getattr(raw, "data", None)
        if isinstance(data, (list, tuple)):
            return [int(getattr(getattr(v, "value", v), "data", v)) for v in data]
    return None


def _root_axes(root) -> tuple[set[int], set[int]] | None:
    """``(row, column)`` output dims of a contraction: those only its first / second operand carries.

    One scale per ``row`` dim is a scale per accumulator row, per ``column`` dim per column, both
    is rank-1. Read from the indexing maps, so it holds for a matmul and a convolution alike.
    """

    from merlin.compile.analysis import contraction_shapes as KS

    from .compute_groups import _NAMED_DIMS, _operand_dims

    name = mq.op_name(root)
    if name in _NAMED_DIMS:
        lhs, rhs, out = _NAMED_DIMS[name]
        return (lhs & out) - rhs, (rhs & out) - lhs
    dims = _operand_dims(root)
    iterators = KS._iterator_types(root)
    if not dims or len(dims) < 3 or not iterators:
        return None
    parallel = {i for i, kind in enumerate(iterators) if kind == "parallel"}
    lhs, rhs = dims[0] & parallel, dims[1] & parallel
    return lhs - rhs, rhs - lhs
