"""Dense input indexing for an explicitly checked singleton projection.

These helpers prove coordinates only. Callers still own source arithmetic,
storage extent, aliasing, lifetime and the consumer's numerical policy.
"""

from __future__ import annotations

from itertools import combinations
from math import prod


def _validate(result_shape, input_shape, axes):
    if (
        not isinstance(result_shape, tuple)
        or not result_shape
        or not isinstance(input_shape, tuple)
        or not isinstance(axes, tuple)
        or any(type(d) is not int or d <= 0 for d in (*result_shape, *input_shape))
        or len(axes) != len(input_shape)
        or any(type(a) is not int or not 0 <= a < len(result_shape) for a in axes)
        or tuple(sorted(set(axes))) != axes
        or any(d not in (1, result_shape[a]) for d, a in zip(input_shape, axes, strict=True))
    ):
        raise ValueError("dense input shape does not broadcast to result on the selected axes")


def singleton_projection_axes(mapping, result_shape: tuple[int, ...], input_shape: tuple[int, ...]) -> tuple[int, ...]:
    """Embed an actual affine projection in distinct ascending output axes.

    Some legal singleton maps have more input axes than output axes. This
    compact representation intentionally refuses those rather than dropping
    coordinates or guessing a trailing-axis interpretation.
    """
    from xdsl.ir.affine import AffineConstantExpr, AffineDimExpr

    from merlin.frontends.linalg_patterns import validate_singleton_projection_map

    validate_singleton_projection_map(mapping, result_shape, input_shape)
    for axes in combinations(range(len(result_shape)), len(input_shape)):
        if all(
            (isinstance(expr, AffineDimExpr) and expr.position == axis)
            or (isinstance(expr, AffineConstantExpr) and expr.value == 0 and extent == 1)
            for extent, expr, axis in zip(input_shape, mapping.results, axes, strict=True)
        ):
            _validate(result_shape, input_shape, axes)
            return axes
    raise ValueError("singleton projection has no ascending output-axis embedding")


def dense_broadcast_index(
    result_shape: tuple[int, ...],
    input_shape: tuple[int, ...],
    axes: tuple[int, ...],
    *,
    base: str = "base",
    index: str = "index",
) -> str:
    """Emit a flat C index into dense input storage for a bounded output lane.

    The caller guarantees ``base + index < prod(result_shape)`` and a C integer
    type wide enough for all shape products. No memory is accessed here.
    """
    _validate(result_shape, input_shape, axes)
    for name in (base, index):
        if not name.isascii() or not name.isidentifier():
            raise ValueError("dense broadcast indexing needs C identifier names")
    lane = f"({base} + {index})"
    if input_shape == result_shape and axes == tuple(range(len(result_shape))):
        return lane
    terms = []
    for i, (axis, extent) in enumerate(zip(axes, input_shape, strict=True)):
        if extent == 1:
            continue
        stride = prod(result_shape[axis + 1 :])
        coordinate = (
            f"({lane} % {result_shape[axis]}u)" if stride == 1 else f"({lane} / {stride}u % {result_shape[axis]}u)"
        )
        terms.append(f"({coordinate} * {prod(input_shape[i + 1 :])}u)")
    return " + ".join(terms) if terms else "0u"
