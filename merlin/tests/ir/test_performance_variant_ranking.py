import pytest

from merlin.transforms.pass_exploration import ObservedMetrics, rank_performance_variants


def observation(name, cycles, instructions, utilization, *, basis='modeled_roi', domain='whole_call:v1'):
    return name, ObservedMetrics(basis, domain, (
        ('cycles', cycles), ('instructions', instructions), ('utilization', utilization),
    ))


def test_fewer_instructions_and_higher_utilization_cannot_override_slower_latency():
    result = rank_performance_variants((
        observation('fast', 100, 1000, .6),
        observation('busy_but_slow', 160, 100, .95),
    ))
    assert [name for name, _ in result['ranked']] == ['fast', 'busy_but_slow']
    assert result['provisional'] is True


def test_unknown_timing_never_falls_back_to_best_instruction_count():
    result = rank_performance_variants((
        observation('timed', 100, 1000, .6),
        observation('unmodeled', None, 1, 1.),
    ))
    assert [name for name, _ in result['ranked']] == ['timed']
    assert result['unranked_unknown'] == (('unmodeled', ('cycles',)),)


def test_all_unknown_remains_unranked():
    result = rank_performance_variants((observation('a', None, 10, .8),))
    assert result['ranked'] == ()


def test_static_counts_cannot_be_presented_as_performance():
    with pytest.raises(ValueError, match='static instruction'):
        rank_performance_variants((observation('a', 10, 10, .8, basis='static_emitted_ir'),))


def test_different_providers_or_scopes_do_not_share_a_rank():
    with pytest.raises(ValueError, match='providers/scopes/domains'):
        rank_performance_variants((
            observation('a', 10, 10, .8), observation('b', 8, 10, .8, domain='another_roi'),
        ))
    with pytest.raises(ValueError, match='providers/scopes/domains'):
        rank_performance_variants((
            observation('a', 10, 10, .8), observation('b', 8, 10, .8, basis='measured_roi'),
        ))
