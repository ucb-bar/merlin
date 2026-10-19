"""Plan call-local preparation of repeated dense broadcast inputs.

Only structural reuse and a caller-selected aggregate byte budget are checked.
The caller proves immutable input spans, private disjoint prepared storage,
complete initialization and conversion semantics independent of lane order.
No cache is retained across invocations and no timing profitability is implied.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import prod

from .tensor_broadcast import dense_broadcast_index


@dataclass(frozen=True)
class BroadcastPreparation:
    input_index: int
    elements: int
    bytes: int
    repeated_preparations_removed: int


def plan_broadcast_preparation(
    result_shape: tuple[int, ...],
    input_shapes: tuple[tuple[int, ...], ...],
    input_axes: tuple[tuple[int, ...], ...],
    *,
    element_bytes: int,
    byte_budget: int,
    alignment_bytes: int = 1,
) -> tuple[BroadcastPreparation, ...]:
    """Prefer the greatest structural reuse per byte within a private budget.

    Full-span inputs are left in the original per-lane path. Every candidate
    is completely re-prepared on each call; identity, source labels and previous
    pointer values play no role. Providers must measure conversion/load costs.
    """
    if (
        type(element_bytes) is not int
        or element_bytes <= 0
        or type(byte_budget) is not int
        or byte_budget < 0
        or type(alignment_bytes) is not int
        or alignment_bytes <= 0
        or alignment_bytes & (alignment_bytes - 1)
        or not isinstance(input_shapes, tuple)
        or not isinstance(input_axes, tuple)
        or len(input_shapes) != len(input_axes)
    ):
        raise ValueError("broadcast preparation requires ordered input shapes and an explicit byte budget")
    dense_broadcast_index(result_shape, (), ())
    candidates = []
    for index, (shape, axes) in enumerate(zip(input_shapes, input_axes, strict=True)):
        dense_broadcast_index(result_shape, shape, axes)
        elements = prod(shape)
        reused = prod(result_shape) - elements
        if reused > 0:
            storage = (elements * element_bytes + alignment_bytes - 1) // alignment_bytes * alignment_bytes
            candidates.append(BroadcastPreparation(index, elements, storage, reused))
    selected, remaining = [], byte_budget

    def priority(p):
        return -Fraction(p.repeated_preparations_removed, p.bytes), p.input_index

    for candidate in sorted(candidates, key=priority):
        if candidate.bytes <= remaining:
            selected.append(candidate)
            remaining -= candidate.bytes
    return tuple(sorted(selected, key=lambda p: p.input_index))


def emit_broadcast_preparation(
    plan: BroadcastPreparation, *, source: str, destination: str, conversion: str, status: str | None = None
) -> str:
    """Emit one complete conversion of the caller-owned dense input span.

    The pure conversion returns the prepared value. An optional status pointer
    can accumulate refusal; callers must check it before publishing any result
    or issuing work that uses prepared storage. Inputs and status cannot alias
    the private destination. Element counts require a sufficiently wide size_t.
    """
    names = (source, destination, conversion, *(tuple() if status is None else (status,)))
    if (
        any(type(name) is not str or not name.isascii() or not name.isidentifier() for name in names)
        or source == destination
        or type(plan) is not BroadcastPreparation
        or any(
            type(value) is not int or value <= 0
            for value in (plan.elements, plan.bytes, plan.repeated_preparations_removed)
        )
        or type(plan.input_index) is not int
        or plan.input_index < 0
    ):
        raise ValueError("broadcast preparation needs bounded spans and safe distinct C names")
    suffix = "" if status is None else f", &{status}"
    return f"""for (size_t prepared_index = 0; prepared_index < {plan.elements}u; ++prepared_index)
  {destination}[prepared_index] = {conversion}({source}[prepared_index]{suffix});
"""
