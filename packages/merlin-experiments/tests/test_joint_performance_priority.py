"""Search advisories never promote provisional models or bypass correctness."""
import copy

import pytest
from merlin_experiments.phase1.feedback import joint_performance_priority as P


def observation():
    result = {'numeric': {'status': 'pass'}, 'tiers': {'L2': {'status': 'pass'}},
              'trace_check': {'status': 'pass'}}
    performance = {key: str(index) * 64 for index, key in enumerate(P._BINDING)}
    performance.update(status='QUALIFIED', timing_qualified=True, timing_mode='tuned',
                       complete_event_accounting=True, estimated_whole_soc_cycles=100,
                       structural_roofline={'structural_compute_floor_cycles': 20}, scope='complete call')
    performance['structural_comparison_binding'] = {key: performance[key] for key in P._BINDING}
    return result, performance


@pytest.mark.parametrize('floor', [None, 0, -1, True, float('nan'), float('inf'), 101])
def test_invalid_bound_never_ranks(floor):
    result, performance = observation()
    performance['structural_roofline']['structural_compute_floor_cycles'] = floor
    value = P.project(result, performance)
    assert value['status'] == 'UNKNOWN'
    assert value['gap_ratio'] is None


@pytest.mark.parametrize('key', P._BINDING)
def test_different_source_config_scope_or_model_refuses(key):
    result, performance = observation()
    performance['structural_comparison_binding'][key] = 'f' * 64
    assert P.project(result, performance)['status'] == 'UNKNOWN'


def test_provisional_order_remains_diagnostic_and_correctness_first():
    result, performance = observation()
    performance.update(status='PROVISIONAL', timing_qualified=False, timing_mode='untuned')
    value = P.project(result, performance)
    assert value['gap_ratio'] == 5
    assert value['diagnostic_headroom'] == 80
    assert value['qualified_advisory_score'] is None
    assert value['invocation_count'] is None
    result['trace_check']['status'] = 'UNKNOWN'
    assert P.project(result, performance)['status'] == 'CORRECTNESS_FIRST'


def test_impact_weight_and_model_confidence_separate_lanes():
    result, performance = observation()
    rows = []
    for name, count in [('small_impact', 1), ('large_impact', 10)]:
        current = copy.deepcopy(performance)
        current['public_workload_weight'] = dict(invocation_count=count,
            capsule_sha256=current['capsule_sha256'], public_source_sha256='a' * 64)
        current['search_priority'] = P.project(result, current)
        rows.append({'capsule': name, 'performance': current})
    provisional = copy.deepcopy(performance)
    provisional.update(status='PROVISIONAL', timing_qualified=False)
    provisional['search_priority'] = P.project(result, provisional)
    rows.append({'capsule': 'provisional', 'performance': provisional})
    summary = P.summarize(rows, roles={r['capsule']: 'large_kernel' for r in rows})
    group, = summary['comparison_groups']
    assert [row['capsule'] for row in group['qualified']] == ['large_impact', 'small_impact']
    assert group['qualified'][0]['qualified_advisory_score'] == 800
    assert len(group['provisional_diagnostics']) == 1
    assert summary['additional_gsim_requested'] is False


def test_unknown_model_never_gets_numeric_priority():
    result, performance = observation()
    performance['complete_event_accounting'] = False
    assert P.project(result, performance)['diagnostic_headroom'] is None
