"""Aligned partitioning keeps minimum launches and avoids needless tail bodies."""
import pytest

from merlin.compile.slice_sizes import uniform_slice_elements


@pytest.mark.parametrize('elements,capacity,alignment,expected', [
    (10,6,1,5), (18,10,1,9), (22,10,2,10), (24,14,4,12),
    (1,17,8,16), (786432,199680,1024,196608),
    (3145728,199680,1024,196608), (131072,199680,1024,199680),
])
def test_equal_slices_when_alignment_and_invocation_count_allow_them(
        elements, capacity, alignment, expected):
    assert uniform_slice_elements(elements,capacity,alignment) == expected


def test_partitions_cover_each_element_once_at_the_minimum_invocation_count():
    for alignment in (1,3,8):
        for capacity in (8,17,33):
            for elements in range(1,200):
                chunk = uniform_slice_elements(elements,capacity,alignment)
                assert 0 < chunk <= capacity and chunk % alignment == 0
                sizes = [min(chunk,elements-start) for start in range(0,elements,chunk)]
                assert sum(sizes) == elements
                assert len(sizes) == (elements + capacity-capacity%alignment-1)//(capacity-capacity%alignment)


@pytest.mark.parametrize('args', [(0,1,1),(1,0,1),(1,1,0),(-1,3,1),
                                  (True,3,1),(3,False,1),(3,4,True),
                                  (3,4,1.5),(3,4,5)])
def test_invalid_or_impossible_extents_refuse(args):
    with pytest.raises(ValueError):
        uniform_slice_elements(*args)
