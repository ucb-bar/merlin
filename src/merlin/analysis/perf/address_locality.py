"""Exact locality of a sequential requested-address trace, independent of hardware.

Each entry is one address, not a byte range: callers must explicitly expand ranges
in their actual request order if that is the desired trace. Regions are aligned at
zero and numbered ``address // granule``. No timing or physical traffic is inferred.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class AddressLocality:
    """Distances count distinct regions strictly between consecutive uses.

    ``recurrences_at_or_above`` excludes first touches. A distance of zero means
    no other region intervened. All pairs are sorted by their integer key.
    """

    granule: int
    requests: int
    first_touches: int
    distance_histogram: tuple[tuple[int, int], ...]
    recurrences_at_or_above: tuple[tuple[int, int], ...]


def _integer(value: object, name: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def address_locality(
    addresses: Iterable[int],
    *,
    granule: int,
    max_requests: int,
    capacities: tuple[int, ...] = (),
) -> AddressLocality:
    """Census exact requested addresses with an explicit resource bound.

    Invalid values, duplicate capacities, or more than ``max_requests`` entries
    raise ValueError; no partial result is returned. At most one entry beyond the
    bound is consumed. Capacities are nonnegative logical-region counts and must
    be a tuple with length <= max_requests. Empty input is valid, including a zero
    request budget. Address integers are nonnegative with no assumed machine width.

    A last-occurrence Fenwick tree counts distinct intervening regions in
    O(N log N) time and O(N) storage; optional thresholds use the sorted histogram,
    never an N-by-capacity scan. Python integer arithmetic remains exact.
    """
    granule = _integer(granule, "granule", 1)
    max_requests = _integer(max_requests, "max_requests", 0)
    if type(capacities) is not tuple or len(capacities) > max_requests:
        raise ValueError("capacities must be a tuple within the request budget")
    for capacity in capacities:
        _integer(capacity, "capacity", 0)
    if len(set(capacities)) != len(capacities):
        raise ValueError("duplicate capacity")
    regions = []
    for address in addresses:
        if len(regions) == max_requests:
            raise ValueError("request budget exceeded")
        regions.append(_integer(address, "address", 0) // granule)
    tree = [0] * (len(regions) + 1)

    def prefix(index: int) -> int:
        total = 0
        while index:
            total += tree[index]
            index -= index & -index
        return total

    def update(index: int, delta: int) -> None:
        while index < len(tree):
            tree[index] += delta
            index += index & -index

    last: dict[int, int] = {}
    histogram: Counter[int] = Counter()
    for index, region in enumerate(regions, 1):
        previous = last.get(region)
        if previous is not None:
            # One live mark for every distinct region seen so far. Marks after
            # the prior use are exactly the distinct intervening regions.
            histogram[len(last) - prefix(previous)] += 1
            update(previous, -1)
        update(index, 1)
        last[region] = index
    pairs = tuple(sorted(histogram.items()))
    distances = [distance for distance, _ in pairs]
    suffix = [0] * (len(pairs) + 1)
    for index in range(len(pairs) - 1, -1, -1):
        suffix[index] = suffix[index + 1] + pairs[index][1]
    thresholds = tuple((capacity, suffix[bisect_left(distances, capacity)]) for capacity in sorted(capacities))
    return AddressLocality(granule, len(regions), len(last), pairs, thresholds)
