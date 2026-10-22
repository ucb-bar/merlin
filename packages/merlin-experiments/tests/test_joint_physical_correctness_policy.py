"""Mandatory numerical obligations do not spend calibration permissions."""
import json
from pathlib import Path

import pytest

from merlin_experiments.phase1 import joint_control as J
from merlin_experiments.phase1.feedback import joint_curriculum as C
from merlin_experiments.phase1.feedback import joint_test_scope as S


@pytest.fixture
def required(tmp_path, monkeypatch):
    names = [f'case{i:03}' for i in range(153)]
    policy = tmp_path / 'policy.json'
    policy.write_text(json.dumps({'version': 1,
        'correctness_curriculum': {'original_capsules': names},
        'final_only_capsules': names[-6:],
        'gsim_correctness': {'capsules': names},
        'gsim': {'witness_capsules': ['extra']}}))
    spec = J.policy(policy)
    state = tmp_path / 'joint_campaign.json'
    state.write_text(json.dumps({'policy': spec, 'started_at_epoch': 1000}))
    monkeypatch.setenv(J.ENV, str(state))
    monkeypatch.setattr(J.time, 'time', lambda: 1100)
    return names, state


def test_all_153_including_six_original_small_models_remain_eligible(required, tmp_path):
    names, _ = required
    root = tmp_path / 'capsules'
    for name in names:
        p = root / name
        p.mkdir(parents=True)
        (p / 'capsule.yaml').write_text(f'name: {name}\nlabel: public\noperation: {{op: matmul}}\n')
    assert J.correctness_capsules() == set(names)
    assert J.development_roots(root) == root
    selected, report = S.select(root, submission=tmp_path / 'compiler')
    assert report['required_development'] == 153
    records = [('matmul', 1, n, root / n) for n in names]
    selected, report = C.plan(records, policy=J.selected()['policy'], attempts={}, passed={})
    assert len(selected) == report['original_development_required'] == 153
    assert report['original_final_only_retained'] == 0
    assert J.stage_evidence({})['joint_campaign']['physical_correctness_required'] == names
    with pytest.raises(RuntimeError, match='final model window'):
        J.require_full_model()  # External full target programs retain their separate window.


def test_required_runs_do_not_consume_witness_questions_or_small_budget(required):
    names, _ = required
    for name in names[:10]:
        with J.physical_correctness(name) as timeout:
            assert timeout == 1200
    ledger = json.loads(J.correctness_ledger().read_text())
    assert len(ledger['admissions']) == 10
    assert all(r['purpose'] == 'required_correctness' for r in ledger['admissions'])
    assert not J.gsim_ledger().exists()
    for _ in range(2):
        with J.physical_witness('extra'):
            pass
    with pytest.raises(ValueError, match='operator-authorized'):
        with J.physical_witness('extra'):
            pytest.fail('calibration permission limit lost')


def test_explicit_required_selfcheck_uses_correctness_budget(required):
    names, _ = required
    seen = []
    @J.selfcheck_policy
    def run(argv):
        seen.append(argv)
        return 0
    assert run(['--sim', 'gsim', '--capsules', names[0], '--tiers', 'L3']) == 0
    assert seen and not J.gsim_ledger().exists()
    assert run(['--sim', 'gsim', '--capsules', 'not-selected']) == 2
    assert run(['--sim', 'gsim', '--capsules', ','.join(names[:2])]) == 2


def test_shared_process_concurrent_admissions_are_individually_retired(required):
    names, state = required
    value = J.selected();value['policy']['gsim_correctness']['max_jobs'] = 2
    state.write_text(json.dumps(value))
    with J.physical_correctness(names[0]):
        with J.physical_correctness(names[1]):
            assert len(json.loads(J.correctness_ledger().read_text())['active']) == 2
            with pytest.raises(ValueError, match='concurrency'):
                with J.physical_correctness(names[2]):
                    pytest.fail('third slot admitted')
        active = json.loads(J.correctness_ledger().read_text())['active']
        assert len(active) == 1 and active[0]['capsule'] == names[0]
    assert json.loads(J.correctness_ledger().read_text())['active'] == []


def test_timeout_failure_and_budget_exhaustion_never_become_pass(required):
    names, state = required
    with pytest.raises(TimeoutError):
        with J.physical_correctness(names[0]):
            raise TimeoutError('target timed out')
    data = json.loads(J.correctness_ledger().read_text())
    assert not data['active'] and not any('pass' in row for row in data['admissions'])
    data['reserved_s'] = J.selected()['policy']['gsim_correctness']['worker_budget_s']
    J.correctness_ledger().write_text(json.dumps(data))
    with pytest.raises(ValueError, match='obligation remains pending'):
        with J.physical_correctness(names[1]):
            pytest.fail('exhausted budget admitted')


def test_correctness_can_drain_until_original_deadline_but_not_epoch_stop(required, monkeypatch):
    names, state = required
    monkeypatch.setattr(J.time, 'time', lambda: 1000 + 43200 - 100)
    with J.physical_correctness(names[0]) as timeout:
        assert timeout == 100
    (state.parent / 'joint_epoch_stop.json').write_text(json.dumps(
        {'schema': 'merlin.joint_epoch_stop.v1', 'reason': 'reviewed checkpoint'}))
    with pytest.raises(ValueError, match='window'):
        with J.physical_correctness(names[1]):
            pytest.fail('epoch stopped admission')
    (state.parent / 'joint_epoch_stop.json').unlink()
    monkeypatch.setattr(J.time, 'time', lambda: 1000 + 43200)
    with pytest.raises(ValueError, match='window'):
        with J.physical_correctness(names[1]):
            pytest.fail('deadline passed')


@pytest.mark.parametrize('change', [
    {'capsules': ['case000']}, {'max_jobs': 21}, {'worker_budget_s': 864001},
    {'capsules': ['all']}, {'capsules': ['case000', 'case000']}])
def test_policy_cannot_drop_original_obligations_or_exceed_reviewed_bounds(required, tmp_path, change):
    policy = J.selected()['policy'];policy['gsim_correctness'].update(change)
    path = tmp_path / 'invalid.json';path.write_text(json.dumps(policy))
    with pytest.raises(ValueError):
        J.policy(path)


@pytest.mark.parametrize('inherited_path', [None, '/original/joint_gsim_correctness_budget.json'])
def test_model_or_library_epoch_retains_physical_correctness_accounting(required, monkeypatch, inherited_path):
    from merlin_experiments.phase1 import joint_epochs as E
    state = J.selected();state['policy_sha256'] = 'policy-pin'
    if inherited_path is not None:
        state['gsim_correctness_ledger'] = inherited_path
    record = {'previous_run': '/previous'}
    monkeypatch.setattr(E, 'selected', lambda options: record)
    monkeypatch.setattr(E, '_verify', lambda record: (state, {}))
    monkeypatch.setattr(E, 'performance_pin', lambda: None)
    got = E.inherited_state(None, state['policy'], 'policy-pin')
    assert got['gsim_correctness_ledger'] == (inherited_path or '/previous/joint_gsim_correctness_budget.json')
    assert got['started_at_epoch'] == state['started_at_epoch']
