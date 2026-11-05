"""Search classification changes rankings and automatic work, never gates."""
import pytest
from merlin_experiments.phase1.feedback import idle_performance as I
from merlin_experiments.phase1.feedback import joint_performance_priority as P
from merlin_experiments.phase1.workload_roles import classify


def member(size=16, **changes):
    return dict(name='region', cohort='scalability_performance', source_sha256='a' * 64,
                declared_operation='matmul', input_tensors=[dict(shape=[size, size])], **changes)


def test_small_scalability_control_never_becomes_primary_by_cohort_alone():
    value = classify(member())
    assert value['workload_role'] == 'diagnostic_control'
    assert value['performance_priority'] == 'diagnostic'
    assert not value['engine_authorization_changed']
    assert 'no routine performance tuning' in value['optimization_goal']
    assert classify(member(512))['performance_priority'] == 'primary'


def test_small_important_region_requires_source_bound_operator_selection():
    selection = dict(members={'region': dict(source_sha256='a' * 64, role='repeated_inference',
                                             reason='public repeated-inference source with shared parameters')})
    assert classify(member(), selection=selection)['workload_role'] == 'repeated_inference'
    selection['members']['region']['source_sha256'] = 'b' * 64
    with pytest.raises(ValueError, match='exact public source'):
        classify(member(), selection=selection)


def test_original_correctness_is_not_promoted_by_size_or_override():
    value = member(512)
    value['cohort'] = 'original_correctness'
    assert classify(value)['workload_role'] == 'correctness'
    with pytest.raises(ValueError, match='cannot replace'):
        classify(value, selection={'members': {'region': dict(source_sha256='a' * 64,
                  role='large_kernel', reason='incorrect attempt to change the original role')}})


def test_primary_filter_preserves_control_diagnostics_and_correctness_failures():
    rows = [dict(capsule=name, performance={'search_priority': {'status': 'CORRECTNESS_FIRST'}})
            for name in ('control', 'kernel')]
    cohorts = dict(primary=['control', 'kernel'], correctness_regression=[], calibration_diagnostics=[])
    value = P.summarize(rows, cohorts=cohorts,
                        roles={'control': 'diagnostic_control', 'kernel': 'large_kernel'})
    assert value['observed_members']['primary'] == ['kernel']
    assert [r['capsule'] for r in value['diagnostic_controls']['correctness_first']] == ['control']
    assert {r['capsule'] for r in value['correctness_first']} == {'control', 'kernel'}


def test_idle_tuning_cannot_reintroduce_controls_from_the_scalability_roster():
    verdict = dict(joint_test_selection=dict(performance_cohorts={'primary': ['control', 'kernel']},
                 workload_roles={'control': 'diagnostic_control', 'kernel': 'connected'}),
                 per_capsule=[dict(capsule=n, tiers={'L2': 'pass'}) for n in ('control', 'kernel')])
    names, status = I._cohort_l2([verdict])
    assert names == ['kernel']
    assert status == {'control': 'pass', 'kernel': 'pass'}


def test_newest_empty_primary_roster_does_not_fall_back_to_an_older_control():
    newest = dict(joint_test_selection=dict(performance_cohorts={'primary': ['control']},
                  workload_roles={'control': 'diagnostic_control'}))
    older = dict(joint_test_selection=dict(performance_cohorts={'primary': ['control']}))
    assert I._cohort_l2([newest, older])[0] == []


def test_response_and_grade_headroom_keep_controls_out_of_optimization_lists(tmp_path):
    import json

    from merlin_experiments.phase1.feedback import performance_brief as B
    from merlin_experiments.phase1.feedback import performance_headroom as H
    # Both are numerically valid with deliberately worse tiny-control headroom.
    (tmp_path / 'PUBLIC_CAPABILITIES.json').write_text(json.dumps(dict(
        schema='merlin.public_capability_inventory.v1', members={
        'control': dict(workload_role='diagnostic_control'), 'kernel': dict(workload_role='large_kernel')})))
    rows = [dict(capsule=name, tiers={'L2': 'pass'}, performance=dict(
        status='PROVISIONAL', complete_event_accounting=True, estimated_whole_soc_cycles=cycles,
        structural_roofline=dict(structural_compute_floor_cycles=10,
                                useful_macs=100, structural_peak_macs_per_cycle=10)))
        for name, cycles in [('control', 10000), ('kernel', 100)]]
    for result in (B.build(rows, workspace=tmp_path), H.build(tmp_path, {'per_capsule': rows})):
        assert [row['test_case'] for row in result['largest_headroom']] == ['kernel']
        assert [row['test_case'] for row in result['diagnostic_controls']] == ['control']



def test_missing_roles_never_enqueue_idle_optimization_or_rank_headroom(tmp_path):
    import json

    from merlin_experiments.phase1.feedback import performance_headroom as H

    verdict = dict(joint_test_selection=dict(performance_cohorts={'primary': ['unknown']}),
                   per_capsule=[dict(capsule='unknown', tiers={'L2': 'pass'}, performance=dict(
                       status='PROVISIONAL', complete_event_accounting=True, estimated_whole_soc_cycles=10000,
                       structural_roofline={'structural_compute_floor_cycles': 10}))])
    names, status = I._cohort_l2([verdict])
    assert names == [] and status == {'unknown': 'pass'}
    block = H.build(tmp_path, verdict)
    assert block['largest_headroom'] == []
    assert block['classification_needed'] == {'count': 1, 'cases': ['unknown']}
    assert block['diagnostic_controls'][0]['test_case'] == 'unknown'
    result = P.summarize(verdict['per_capsule'], cohorts={
        'primary': ['unknown'], 'correctness_regression': [], 'calibration_diagnostics': []})
    assert result['comparison_groups'] == [] and result['classification_needed'] == ['unknown']
    # Merely present inventory without declared roles does not infer a search role.
    (tmp_path / 'PUBLIC_CAPABILITIES.json').write_text(json.dumps({
        'schema': 'merlin.public_capability_inventory.v1', 'members': {'unknown': {}}}))
    assert H.build(tmp_path, verdict)['largest_headroom'] == []
