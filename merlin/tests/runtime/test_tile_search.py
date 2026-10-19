"""Bounded complete tile search preserves frozen partitions and known work."""

import pytest

from merlin.sched.tile_search import aligned_divisors, select_divisor_tiles


def test_all_aligned_divisors_include_non_power_of_two_choices():
    assert aligned_divisors(160, 16) == (160, 80, 32, 16)
    assert aligned_divisors(180, 12) == (180, 60, 36, 12)


def search(**extra):
    return select_divisor_tiles((12, 18), (2, 3), (6, 3),
                                 admit=lambda tile: tile[0] * tile[1] <= 72,
                                 work=lambda tile: (12 // tile[0] * (18 // tile[1]), 7), **extra)


def test_complete_search_honors_provider_capacity_and_frozen_axes():
    result = search()
    assert result.complete and result.selected == (12, 6)
    assert result.baseline_work == (12, 7) and result.selected_work == (3, 7)
    assert search(frozen_axes=(1,)).selected == (12, 3)


def test_equal_vectors_and_budget_exhaustion_keep_original_partition():
    assert select_divisor_tiles((12,), (2,), (6,), admit=lambda _: True, work=lambda _: (0,)).selected == (6,)
    result = search(max_candidates=2)
    assert not result.complete and result.examined == 0 and result.selected == result.baseline
    result = search(max_factor_steps=1)
    assert not result.complete and result.stop_reason == "factorization_budget"


def test_a_lower_primary_metric_cannot_hide_an_increased_secondary_metric():
    result = select_divisor_tiles((12,), (2,), (6,), admit=lambda _: True,
                                   work=lambda tile: (12 // tile[0], tile[0]))
    assert result.selected == (6,)


@pytest.mark.parametrize("bad", [(None,), (True,), (-1,), (1.0,), ()])
def test_unknown_or_noninteger_work_cannot_be_ranked(bad):
    with pytest.raises(ValueError, match="complete nonnegative"):
        select_divisor_tiles((12,), (2,), (6,), admit=lambda _: True, work=lambda _: bad)


def test_unknown_legality_and_changed_metric_axes_refuse():
    with pytest.raises(ValueError, match="Boolean"):
        select_divisor_tiles((12,), (2,), (6,), admit=lambda _: None, work=lambda _: (1,))
    with pytest.raises(ValueError, match="axes changed"):
        select_divisor_tiles((12,), (2,), (6,), admit=lambda _: True,
                             work=lambda tile: (1,) if tile == (6,) else (1, 2))
