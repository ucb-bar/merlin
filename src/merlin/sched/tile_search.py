"""Bounded divisor search using provider-supplied legality and structural work.

This selects a nonincreasing structural proxy, never a predicted cycle count.
Providers own physical costs, capacities, numerical partitioning and qualification.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import isqrt, prod


class _SearchBudget(ValueError):
    pass


def aligned_divisors(extent: int, alignment: int, *, max_factor_steps: int = 65536) -> tuple[int, ...]:
    if (type(extent) is not int or type(alignment) is not int or extent <= 0 or alignment <= 0
            or extent % alignment or type(max_factor_steps) is not int or max_factor_steps <= 0):
        raise ValueError("tile search needs positive divisible extents and a factorization budget")
    quotient = extent // alignment
    if isqrt(quotient) > max_factor_steps:
        raise _SearchBudget("factorization_budget")
    factors = set()
    for value in range(1, isqrt(quotient) + 1):
        if quotient % value == 0:
            factors.update((value * alignment, quotient // value * alignment))
    return tuple(sorted(factors, reverse=True))


@dataclass(frozen=True)
class TileSearchResult:
    baseline: tuple[int, ...]
    selected: tuple[int, ...]
    baseline_work: tuple[int, ...]
    selected_work: tuple[int, ...]
    examined: int
    legal_candidates: int
    complete: bool
    stop_reason: str


def select_divisor_tiles(shape, alignment, baseline, *, admit, work, frozen_axes=(),
                         max_candidates: int = 4096, max_factor_steps: int = 65536) -> TileSearchResult:
    """Select lower lexicographic work only when every supplied axis is no worse.

    ``admit(tile)`` is a provider legality proof, not an arithmetic theorem.
    ``work(tile)`` returns complete, nonnegative exact structural quantities in
    a consistent order. It must not silently substitute zero for unknown work.
    Frozen axes preserve caller-selected partitioning; other numerical and
    resource obligations remain with the provider. Equal work retains baseline.
    Budget exhaustion returns the original tile, never a partial-search winner.
    """
    if (type(shape) is not tuple or type(alignment) is not tuple or type(baseline) is not tuple
            or not shape or len(shape) != len(alignment) or len(shape) != len(baseline)
            or any(type(x) is not int or x <= 0 for group in (shape, alignment, baseline) for x in group)
            or any(n % a or n % b or b % a for n, a, b in zip(shape, alignment, baseline))
            or type(frozen_axes) is not tuple or len(set(frozen_axes)) != len(frozen_axes)
            or any(type(axis) is not int or not 0 <= axis < len(shape) for axis in frozen_axes)
            or type(max_candidates) is not int or max_candidates <= 0):
        raise ValueError("tile search needs aligned source/baseline extents and explicit bounded axes")

    def legal(tile):
        result = admit(tile)
        if type(result) is not bool:
            raise ValueError("tile legality must be an explicit Boolean")
        return result

    def evaluate(tile):
        result = work(tile)
        if type(result) is not tuple or not result or any(type(x) is not int or x < 0 for x in result):
            raise ValueError("tile work must contain complete nonnegative integer quantities")
        return result

    if not legal(baseline):
        raise ValueError("tile search baseline lacks provider admission")
    original = evaluate(baseline)
    try:
        domains = [(baseline[i],) if i in frozen_axes else
                   aligned_divisors(n, a, max_factor_steps=max_factor_steps)
                   for i, (n, a) in enumerate(zip(shape, alignment))]
    except _SearchBudget as error:
        return TileSearchResult(baseline, baseline, original, original, 0, 1, False, str(error))
    if prod(map(len, domains)) > max_candidates:
        return TileSearchResult(baseline, baseline, original, original, 0, 1, False, "candidate_budget")
    selected, selected_work, examined, accepted = baseline, original, 0, 0
    for tile in product(*domains):
        examined += 1
        if not legal(tile):
            continue
        accepted += 1
        values = evaluate(tile)
        if len(values) != len(original):
            raise ValueError("tile work axes changed during enumeration")
        if all(a <= b for a, b in zip(values, original)) and values < selected_work:
            selected, selected_work = tile, values
    return TileSearchResult(baseline, selected, original, selected_work, examined, accepted, True, "exhausted")
