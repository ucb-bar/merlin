"""Prospective validation grouping and honest same-domain static screening."""

import pytest

from merlin.transforms.pass_exploration import (
    ExplorationGeometry,
    ObservedMetrics,
    ValidationPartition,
    rank_observed_metrics,
)


def geometry(m, *, family='dense'):
    return ExplorationGeometry(family, ((m, 32), (32, 48)))


def test_alias_or_epilogue_renaming_cannot_create_a_geometry_holdout():
    with pytest.raises(ValueError, match='aliases'):
        ValidationPartition((geometry(16),), (geometry(16, family='another_readout'),))
    with pytest.raises(ValueError, match='distinct grouped'):
        ValidationPartition((geometry(16), geometry(16)), (geometry(32),))


def test_connected_families_and_geometries_are_frozen_before_tuning():
    partition = ValidationPartition((geometry(16),), (geometry(32), geometry(48, family='connected')),
                                    ('connected',))
    assert partition.manifest() == partition.manifest()
    assert partition.manifest()['training'][0]['geometry_sha256'] == geometry(16).identity
    with pytest.raises(ValueError, match='withheld families'):
        ValidationPartition((geometry(16, family='connected'),), (geometry(32, family='connected'),),
                            ('connected',))


def test_unknown_is_unranked_and_never_a_zero_cost_win():
    def metrics(traffic, drains):
        return ObservedMetrics('static_emitted_ir', 'counted-command-v1',
                               (('requested_bytes', traffic), ('drains', drains)))

    result = rank_observed_metrics((('known', metrics(512, 8)), ('unknown', metrics(None, 0)),
                                    ('better', metrics(256, 12))), objectives=('requested_bytes', 'drains'))
    assert [name for name, _ in result['ranked']] == ['better', 'known']
    assert result['unranked_unknown'] == (('unknown', ('requested_bytes',)),)
    assert result['performance_qualification'] is False


def test_mixed_provider_or_scope_does_not_rank_and_nonfinite_refuses():
    with pytest.raises(ValueError, match='providers/scopes/domains'):
        rank_observed_metrics((('a', ObservedMetrics('modeled_roi', 'provider-A/roi', (('cycles', 1),))),
                               ('b', ObservedMetrics('modeled_roi', 'provider-B/roi', (('cycles', 2),)))),
                              objectives=('cycles',))
    for value in (float('nan'), float('inf'), -1, True):
        with pytest.raises(ValueError, match='finite nonnegative'):
            ObservedMetrics('static_emitted_ir', 'd', (('value', value),))


def test_exact_large_counts_and_malformed_closed_keys():
    huge = 10**1000
    observed = ObservedMetrics('static_emitted_ir', 'exact-counts', (('commands', huge),))
    assert dict(observed.values)['commands'] == huge
    with pytest.raises(ValueError):
        ObservedMetrics('static_emitted_ir', 'd', (([], 1),))
    with pytest.raises(ValueError):
        rank_observed_metrics((('a', observed),), objectives=([],))
