"""Independent set-based oracle for sequential logical region recurrence."""
from collections import Counter
from itertools import product
import random

import pytest

from merlin.perf.address_locality import address_locality


def oracle(addresses, granule):
    regions = [address // granule for address in addresses]
    histogram = Counter()
    seen = set()
    for index, region in enumerate(regions):
        if region in seen:
            prior = max(i for i in range(index) if regions[i] == region)
            histogram[len(set(regions[prior + 1:index]))] += 1
        seen.add(region)
    return len(seen), tuple(sorted(histogram.items()))


def check(trace, granule):
    capacities = tuple(range(7))
    result = address_locality(iter(trace), granule=granule,
                              max_requests=max(7, len(trace)), capacities=capacities)
    cold, histogram = oracle(trace, granule)
    assert (result.first_touches, result.distance_histogram) == (cold, histogram)
    assert result.requests == len(trace)
    assert cold + sum(count for _, count in histogram) == len(trace)
    assert result.recurrences_at_or_above == tuple(
        (cap, sum(count for distance, count in histogram if distance >= cap))
        for cap in capacities)


def test_exhaustive_and_random_oracle():
    for size in range(7):
        for trace in product(range(3), repeat=size):
            check(trace, 1)
    rng = random.Random(281)
    for granule in (1, 3, 17, 129):
        for _ in range(30):
            check([rng.randrange(2048) for _ in range(100)], granule)


@pytest.mark.parametrize('trace', [[], [5] * 100, [1, 2] * 100,
                                  list(range(300)), list(range(100)) * 3,
                                  [0, 1, 2, 1, 2, 0], [2**120, 0, 2**120]])
def test_adversarial(trace):
    check(trace, 1)


def test_boundaries_and_order():
    check([0, 2, 3, 5, 6, 2, 0], 3)
    a = address_locality([0, 0, 1, 1], granule=1, max_requests=4)
    b = address_locality([0, 1, 0, 1], granule=1, max_requests=4)
    assert a.first_touches == b.first_touches == 2
    assert a.distance_histogram == ((0, 2),)
    assert b.distance_histogram == ((1, 2),)
    assert address_locality([], granule=3, max_requests=0).requests == 0


@pytest.mark.parametrize('kwargs', [dict(granule=0), dict(granule=True),
    dict(granule=1.5), dict(max_requests=-1), dict(max_requests=True),
    dict(capacities=(True,)), dict(capacities=(-1,)), dict(capacities=(1, 1)),
    dict(capacities=[1]), dict(capacities=(0, 1, 2, 3))])
def test_invalid_contract(kwargs):
    options = dict(granule=1, max_requests=3)
    options.update(kwargs)
    with pytest.raises(ValueError):
        address_locality([], **options)


@pytest.mark.parametrize('address', [-1, True, 0.5, '1', None])
def test_invalid_address(address):
    with pytest.raises(ValueError):
        address_locality([0, address], granule=1, max_requests=2)


def test_budget_consumes_only_one_excess_entry():
    consumed = []
    def trace():
        for value in range(20):
            consumed.append(value)
            yield value
    with pytest.raises(ValueError, match='budget'):
        address_locality(trace(), granule=1, max_requests=3)
    assert consumed == [0, 1, 2, 3]


def test_large_distinct_and_repeated_trace():
    result = address_locality(list(range(20000)) * 2, granule=1, max_requests=40000)
    assert result.first_touches == 20000
    assert result.distance_histogram == ((19999, 20000),)
