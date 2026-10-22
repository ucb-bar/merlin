"""Diagnostic orchestration fixtures do not launch native or physical engines."""
import contextlib
import json
from types import SimpleNamespace

import pytest

from merlin_experiments.phase1 import joint_diagnostic_collection as D
from merlin_experiments.phase1 import joint_calibration_launcher as L
from merlin_experiments.phase1 import joint_runtime_revision as R


@pytest.fixture
def setup(tmp_path, monkeypatch):
    capsule = {'name': 'public_case', 'kind': 'operator', 'label': 'public'}
    profile = {'members': {'public_case': {'source_group_sha256': 'group'}},
               'performance_selection': {'path': 'selection', 'sha256': 'selection'}}
    profile_pin = {'path': 'profile', 'sha256': 'profile'}
    state = dict(policy_sha256='policy', started_at_epoch=1,
                 policy={'native_calibration': profile_pin, 'gsim': {'witness_capsules': ['public_case']}})
    request = dict(schema=D.SCHEMA, policy_sha256='policy', profile=profile_pin, fit_eligible=False,
                   review_id='review', max_observations=1, questions={'public_case': 'Which cost is unresolved?'})
    pin = {'path': 'request', 'sha256': 'request'}
    selection = {'members': {'public_case': {'capsule_sha256': 'source'}}, 'facts': {'sha256': 'facts'}}
    runtime_pin = {'path': 'runtime', 'sha256': 'runtime'}
    grade = {'numeric': {'status': 'pass'}, 'tiers': {'L2': {'status': 'pass', 'engine': 'spike'}},
             'trace_check': {'status': 'pass'}, 'compiler_source': {'receipt': 'owned'}}
    entry = dict(id='entry', capsule={'path': 'capsule'}, result={'path': 'result'},
                 source_group_sha256='group', estimate_status='UNKNOWN', generated_root=str(tmp_path))
    docs = {'request': request, 'profile': profile, 'selection': selection, 'result': grade}
    original_read = D.PERF._read
    monkeypatch.setattr(D.PERF, '_read', lambda p: docs[p['path']] if p['path'] in docs else original_read(p))
    monkeypatch.setattr(D.PERF, 'selected_capsule', lambda s, c: c)
    monkeypatch.setattr(D.PERF, 'capsule_identity', lambda c: 'source')
    monkeypatch.setattr(D.C, '_capsule', lambda p: capsule)
    monkeypatch.setattr(D.joint_control, 'selected', lambda: state)
    monkeypatch.setattr(D.joint_control, 'stop_reason', lambda: None)
    monkeypatch.setattr(D.joint_control, 'gsim_ledger', lambda: tmp_path / 'original-ledger.json')
    monkeypatch.setattr(R, 'validate', lambda a, b: None)
    monkeypatch.setattr(D, 'choose', lambda *a: {'runtime': runtime_pin})
    runtime = SimpleNamespace(model_json='{}', verify=lambda: None)
    monkeypatch.setattr(D.PERF, '_runtime', lambda *a, **k: runtime)
    elf = tmp_path / 'program.elf'
    elf.write_bytes(b'fixture only')
    monkeypatch.setattr(D.PERF, 'original_executable', lambda root: elf)
    state_path = tmp_path / 'campaign.json'
    monkeypatch.setenv(D.joint_control.ENV, str(state_path))
    registry = tmp_path / 'calibration_registry' / 'one'
    registry.mkdir(parents=True)
    (registry / 'entry.json').write_text(json.dumps(entry))
    calls = []

    def observe(*a, **kw):
        calls.append(('observe', kw['require_complete']))
        return elf, {'report': {'path': 'observed', 'sha256': 'observed'}}, {
            'timing_complete': False, 'unresolved': ['unpriced_public_operation']}

    monkeypatch.setattr(D.C, '_observe', observe)
    @contextlib.contextmanager
    def physical_runtime(*a, **k):
        yield object()
    monkeypatch.setattr(L, 'physical_runtime', physical_runtime)
    monkeypatch.setattr(D.C, '_authorize_if_needed', lambda *a: calls.append(('authorize', a)))
    def measure(*a, **kw):
        assert kw['diagnostic'] is True
        calls.append(('measure', kw['output']))
        return D.C._write(kw['output'] / 'measurement.json', {'numeric': 'pass', 'cycles': 20})
    monkeypatch.setattr(D.C, '_measure', measure)
    return SimpleNamespace(pin=pin, request=request, state=state, state_path=state_path,
                           output=tmp_path / 'diagnostic-only', grade=grade, calls=calls,
                           profile=profile, selection=selection, entry=entry)


def test_unknown_model_collects_only_diagnostic_and_never_retries(setup):
    s = setup
    result = D.step(request_pin=s.pin, state_path=s.state_path, context=None, output=s.output)
    assert result['status'] == 'diagnostic_observed'
    assert result['fit_eligible'] is False and result['formal_certificate'] is False
    assert ('observe', False) in s.calls
    assert len([c for c in s.calls if c[0] == 'measure']) == 1
    assert not list(s.output.glob('round-*'))
    second = D.step(request_pin=s.pin, state_path=s.state_path, context=None, output=s.output)
    assert second['status'] == 'reviewed_diagnostic_limit_reached'
    assert len([c for c in s.calls if c[0] == 'measure']) == 1


def test_failed_custody_never_reaches_physical_admission(setup, monkeypatch):
    def refuse(*a, **k):
        raise ValueError('original candidate snapshot changed')
    monkeypatch.setattr(D.C, '_observe', refuse)
    s = setup
    result = D.step(request_pin=s.pin, state_path=s.state_path, context=None, output=s.output)
    assert result['status'] == 'diagnostic_refused'
    assert not s.calls
    assert not list(s.output.glob('attempt-*/physical-intent.json'))
    assert D.step(request_pin=s.pin, state_path=s.state_path, context=None,
                  output=s.output)['status'] == 'waiting_for_matching_unknown_model_product'


def test_physical_failure_retains_consumed_intent(setup, monkeypatch):
    def refuse(*a, **k):
        raise ValueError('physical output mismatch')
    monkeypatch.setattr(D.C, '_measure', refuse)
    s = setup
    assert D.step(request_pin=s.pin, state_path=s.state_path, context=None,
                  output=s.output)['status'] == 'diagnostic_refused'
    assert len(list(s.output.glob('attempt-*/physical-intent.json'))) == 1
    assert D.step(request_pin=s.pin, state_path=s.state_path, context=None,
                  output=s.output)['status'] == 'reviewed_diagnostic_limit_reached'


@pytest.mark.parametrize('change', ['numerics', 'trace', 'model', 'question'])
def test_absent_admission_evidence_refuses(setup, change):
    s = setup
    if change == 'numerics': s.grade['numeric']['status'] = 'fail'
    elif change == 'trace': s.grade['trace_check']['status'] = 'UNKNOWN'
    elif change == 'model': s.entry['estimate_status'] = 'PROVISIONAL'
    questions = {} if change == 'question' else s.request['questions']
    assert D.eligible(s.entry, profile=s.profile, selection=s.selection, questions=questions) is None


def test_request_cannot_extend_whitelist_or_be_fit_eligible(setup):
    s = setup
    s.request['questions']['outside'] = 'not selected'
    with pytest.raises(ValueError, match='original physical policy'):
        D.request(s.pin, s.state)
    del s.request['questions']['outside']
    s.request['fit_eligible'] = True
    with pytest.raises(ValueError, match='original physical policy'):
        D.request(s.pin, s.state)


def test_diagnostic_receipt_cannot_enter_fit_labels():
    with pytest.raises(ValueError, match='not admitted fit labels'):
        D.C._reopen_physical({'schema': 'merlin.joint_structure_diagnostic_measurement.v1',
                              'fit_eligible': False}, {}, None)


def validation_request(s):
    s.request.update(purpose=D.VALIDATION,
                     performance_selection=s.profile['performance_selection'])
    s.entry['estimate_status'] = 'PROVISIONAL'
    (s.state_path.parent / 'calibration_registry/one/entry.json').write_text(json.dumps(s.entry))


def test_reviewed_complete_model_validation_collects_without_fit_gate(setup, monkeypatch):
    s = setup
    validation_request(s)
    original = D.C._observe
    def complete(*a, **kw):
        elf, observation, report = original(*a, **kw)
        report.update(timing_complete=True, unresolved=[], estimated_cycles=23)
        return elf, observation, report
    monkeypatch.setattr(D.C, '_observe', complete)
    # Fitting remains explicitly denied. Diagnostic observation must neither
    # consult nor change this separate authority.
    from merlin_experiments.phase1 import joint_calibration_admission as admission
    def deny(*a, **kw):
        raise AssertionError('fit admission must not be consulted for diagnostic collection')
    monkeypatch.setattr(admission, 'assert_allowed', deny)
    result = D.step(request_pin=s.pin, state_path=s.state_path, context=None, output=s.output)
    assert result['status'] == 'diagnostic_observed'
    assert result['purpose'] == D.VALIDATION
    assert result['model_timing_complete'] is True
    assert result['fit_eligible'] is result['promotion_authority'] is result['formal_certificate'] is False
    assert len([c for c in s.calls if c[0] == 'measure']) == 1
    assert not list(s.output.glob('round-*'))
    assert D.step(request_pin=s.pin, state_path=s.state_path, context=None,
                  output=s.output)['status'] == 'reviewed_diagnostic_limit_reached'
    receipt = D.PERF._read(result['measurement'])
    receipt.update(schema='merlin.joint_structure_diagnostic_measurement.v1', fit_eligible=False)
    with pytest.raises(ValueError, match='not admitted fit labels'):
        D.C._reopen_physical(receipt, {}, None)


def test_old_request_still_skips_replay_that_now_has_complete_timing(setup, monkeypatch):
    s = setup
    original = D.C._observe
    def complete(*a, **kw):
        elf, observation, report = original(*a, **kw)
        report.update(timing_complete=True, unresolved=[])
        return elf, observation, report
    monkeypatch.setattr(D.C, '_observe', complete)
    result = D.step(request_pin=s.pin, state_path=s.state_path, context=None, output=s.output)
    assert result['status'] == 'model_now_complete_no_diagnostic'
    assert not [c for c in s.calls if c[0] in {'authorize', 'measure'}]


@pytest.mark.parametrize('change', ['missing_pin', 'changed_epoch', 'wrong_purpose', 'numerics', 'trace'])
def test_validation_cannot_bypass_selection_or_original_execution(setup, change):
    s = setup
    validation_request(s)
    if change == 'missing_pin': del s.request['performance_selection']
    elif change == 'changed_epoch': s.state['performance_selection'] = {'path': 'new', 'sha256': 'new'}
    elif change == 'wrong_purpose': s.request['purpose'] = 'fit'
    elif change == 'numerics': s.grade['numeric']['status'] = 'fail'
    elif change == 'trace': s.grade['trace_check']['status'] = 'UNKNOWN'
    if change in {'missing_pin', 'changed_epoch', 'wrong_purpose'}:
        with pytest.raises(ValueError):
            D.step(request_pin=s.pin, state_path=s.state_path, context=None, output=s.output)
    else:
        result = D.step(request_pin=s.pin, state_path=s.state_path, context=None, output=s.output)
        assert result['status'] == 'waiting_for_matching_validation_product'
    assert not s.calls
