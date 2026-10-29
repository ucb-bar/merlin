import pytest
from merlin_experiments.phase1 import joint_calibration_readiness as R
from merlin_experiments.phase1 import joint_calibration as C
from test_joint_calibration_eligibility import valid_grade


def fixture(monkeypatch):
    caps = {n: dict(name=n, label='public', kind='layer') for n in ('a', 'b')}
    selection = dict(runtime={'path':'runtime', 'sha256':'r'}, members={
        n: {'capsule_sha256':R.PERF.capsule_identity(c)} for n,c in caps.items()})
    profile = dict(members={n:{'source_group_sha256':n} for n in caps}, parameters=[])
    monkeypatch.setattr(C, '_capsule', lambda pin: caps[pin['path']])
    monkeypatch.setattr(R.PERF, '_read', lambda pin: valid_grade(source=pin['compiler']))
    rows = [dict(id=str(i), capsule={'path':'a'}, result={'compiler':c}, compiler_sha256=c,
                 source_group_sha256='a') for i,c in enumerate(('c1','c2'))]
    return profile, selection, rows


def test_denial_visible_without_any_variant_or_physical_call(monkeypatch):
    profile, selection, _ = fixture(monkeypatch)
    calls = []
    def denied(plan):
        calls.append(plan)
        raise ValueError('blocked_structural_model_admission: repair WS structure')
    monkeypatch.setattr(R, 'assert_allowed', denied)
    report = R.panel(profile, selection, [], witnesses={'a','b'})
    assert len(calls) == 1  # Shared runtime checked once, including an empty panel.
    assert {r['status'] for r in report['members']} == {'blocked_structural_model'}
    assert not report['admissible_registry_ids'] and not report['dispatch_authorized']


def test_source_compiler_and_physical_roster_admission(monkeypatch):
    profile, selection, rows = fixture(monkeypatch)
    monkeypatch.setattr(R, 'assert_allowed', lambda plan: None)
    report = R.panel(profile, selection, rows, witnesses={'a'})
    assert [r['status'] for r in report['members']] == [
        'ready_for_native_identifiability_review', 'outside_physical_witness_roster']
    assert report['admissible_registry_ids'] == ['0','1']
    assert report['members'][0]['compiler_sha256s'] == ['c1','c2']
    # Repeated product of one source snapshot is not a distinct compiler variant.
    rows[1]['compiler_sha256'] = 'c1'
    assert R.panel(profile, selection, rows, witnesses={'a'})['members'][0]['status'] == 'pending_same_source_compiler_variant'
    selection['members']['a']['capsule_sha256'] = 'changed'
    report = R.panel(profile, selection, rows, witnesses={'a'})
    assert report['members'][0]['status'] == 'pending_passing_candidate_spike_product'
    assert report['admissible_registry_ids'] == []


def test_broken_review_not_misreported_as_need_for_measurements(monkeypatch):
    profile, selection, _ = fixture(monkeypatch)
    def broken(plan): raise ValueError('review evidence changed')
    monkeypatch.setattr(R, 'assert_allowed', broken)
    with pytest.raises(ValueError, match='review evidence changed'):
        R.panel(profile, selection, [], witnesses={'a'})


def test_native_service_publishes_denial_before_waiting_or_dispatch(tmp_path, monkeypatch):
    from merlin_experiments.phase1 import joint_calibration_service as S
    from merlin_experiments.phase1 import joint_runtime_revision
    profile, selection, _ = fixture(monkeypatch)
    profile['performance_selection'] = {'path':'selection'}
    profile_pin = {'path':'profile','sha256':'p'}
    monkeypatch.setattr(S.PERF, '_read', lambda pin: profile if pin == profile_pin else selection)
    monkeypatch.setattr(S.joint_control, 'selected', lambda: {
        'policy':{'native_calibration':profile_pin,'gsim':{'witness_capsules':['a','b']}}})
    monkeypatch.setattr(joint_runtime_revision, 'validate', lambda initial,current: None)
    def denied(plan): raise ValueError('blocked_structural_model_admission: repair selected model')
    monkeypatch.setattr(R, 'assert_allowed', denied)
    monkeypatch.setattr(C, 'select_variant_cases', lambda rows: pytest.fail('denied model must not chase variants'))
    result = S.step(profile_pin=profile_pin,state_path=tmp_path/'state.json',context=None,output=tmp_path/'out')
    assert result['status'] == 'blocked_structural_model_admission'
    assert (tmp_path/'out/panel-readiness.json').is_file()
    assert not list((tmp_path/'out').glob('round-*'))
