"""Rolling numerical receipts and timing projections remain separate owners."""
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from merlin_experiments.phase1.feedback import rolling_scalability as R
from merlin_experiments.phase1.feedback import joint_performance as P
from merlin.targetgen import capsule_common, capsule_runner


@pytest.fixture
def scene(tmp_path, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr('time.monotonic', lambda: clock[0])
    monkeypatch.setattr(P, 'deadline', lambda timeout: clock[0] + min(timeout, 10))
    monkeypatch.setattr(capsule_runner, 'suite_for', lambda target: 'suite')
    monkeypatch.setattr(capsule_common, 'load_capsule', lambda p: {'name': p.name})
    cases = {name: tmp_path/'frozen'/name for name in ('small', 'large', 'tail')}
    selection = SimpleNamespace(cases=cases, context=SimpleNamespace(target='test'), verify=lambda: None)
    runs = tmp_path/'runs'
    receipts = {}
    for name in cases:
        path = runs/'runs/suite'/name/'capsule_result.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({'capsule': name, 'numeric': {'status': 'pass'},
            'tiers': {'L2': {'status': 'pass', 'engine': 'spike'}}, 'status': 'pass'}))
        receipts[name] = path
    calls = []
    def attach(row, root, **kw):
        calls.append((root, kw))
        clock[0] += 3
        row['performance'] = {'status': 'PROVISIONAL', 'estimated_whole_soc_cycles': 123}
        row['compiler_diagnostics'] = {'status': 'UNKNOWN'}
        row['private_should_not_project'] = 'fixture'
    monkeypatch.setattr(P, 'attach', attach)
    feedback = R.PerformanceFeedback(selection, compiler=tmp_path/'candidate', runs_root=runs, timeout=60)
    return SimpleNamespace(feedback=feedback, selection=selection, receipts=receipts,
                           calls=calls, clock=clock, runs=runs)


def test_each_case_has_its_own_budget_excluding_grade_wait_and_preserves_native_receipts(scene):
    s = scene
    before = {name: path.read_bytes() for name, path in s.receipts.items()}
    s.clock[0] += 500  # Compilation and native grading are not observation time.
    callback = {'capsule': 'small', 'numeric': {'status': 'fail'}}
    original = deepcopy(callback)
    s.feedback.completed(callback)
    s.feedback.completed({'capsule': 'large'})
    s.feedback.completed({'capsule': 'tail'})
    # Each case's budget is its own: earlier cases' observations (3 s each) never consume it.
    assert [call[1]['remaining_s'] for call in s.calls] == [10, 10, 10]
    assert all(call[1]['result']['numeric']['status'] == 'pass' for call in s.calls)
    assert s.calls[0][0] == s.receipts['small'].parent/'generated'
    assert callback == original
    assert {name: path.read_bytes() for name, path in s.receipts.items()} == before
    verdict = {'all_pass': None, 'n_passed': 2,
               'per_capsule': [{'capsule': 'small', 'status': 'pass', 'tiers': {'L2': 'pass'}},
                               {'capsule': 'unrun', 'status': 'gated'}]}
    s.feedback.project(verdict)
    assert verdict['all_pass'] is None and verdict['n_passed'] == 2
    assert verdict['per_capsule'][0]['performance']['status'] == 'PROVISIONAL'
    assert verdict['per_capsule'][0]['tiers'] == {'L2': 'pass'}
    assert 'private_should_not_project' not in verdict['per_capsule'][0]
    assert 'performance' not in verdict['per_capsule'][1]
    assert len(list((s.runs/'rolling_scalability_performance').glob('*.json'))) == 3


@pytest.mark.parametrize('changed', ['numeric_receipt', 'projection', 'frozen_selection'])
def test_changed_feedback_custody_refuses_before_any_verdict_mutation(scene, changed):
    s = scene
    s.feedback.completed({'capsule': 'small'})
    if changed == 'numeric_receipt':
        s.receipts['small'].write_text('{}')
    elif changed == 'projection':
        (s.runs/'rolling_scalability_performance/small.json').write_text('{}')
    else:
        def refuse(): raise ValueError('selected suite changed')
        s.selection.verify = refuse
    verdict = {'per_capsule': [{'capsule': 'small', 'status': 'pass'}]}
    original = deepcopy(verdict)
    with pytest.raises(ValueError): s.feedback.project(verdict)
    assert verdict == original


def test_unselected_or_duplicate_callbacks_cannot_observe_again(scene):
    s = scene
    s.feedback.completed({'capsule': 'original_core'})
    assert not s.calls
    s.feedback.completed({'capsule': 'small'})
    with pytest.raises(ValueError, match='duplicate'):
        s.feedback.completed({'capsule': 'small'})
    assert len(s.calls) == 1


def test_numerical_result_replacement_during_observation_refuses(scene, monkeypatch):
    s = scene
    def replace(row, root, **kw):
        s.receipts['small'].write_text('{}')
        row['performance'] = {'status': 'PROVISIONAL'}
    monkeypatch.setattr(P, 'attach', replace)
    with pytest.raises(ValueError, match='changed during observation'):
        s.feedback.completed({'capsule': 'small'})
    assert not s.feedback.records


def _vetoed(path, **extra):
    from merlin.targetgen import offload_census as O
    O._record_veto(path, {'status': 'fail', 'host_compute': {'verdict': 'host_compute'},
                          'failure': {'plane': 'host_compute', 'category': O.VETO_CATEGORY, 'detail': 'x'}})
    if extra:
        stored = json.loads(path.read_text())
        stored.update(extra)
        path.write_text(json.dumps(stored))


def test_recorded_host_compute_veto_drops_only_that_cases_feedback(scene):
    # The score vetoes a numerically passing case for host compute after its rolling observation and
    # rewrites its result; a fresh backend does this early. The grade must stand.
    s = scene
    for name in ('small', 'large'):
        s.feedback.completed({'capsule': name})
    _vetoed(s.receipts['small'])
    verdict = {'per_capsule': [{'capsule': 'small', 'status': 'fail'}, {'capsule': 'large', 'status': 'pass'}]}
    s.feedback.project(verdict)
    assert 'performance' not in verdict['per_capsule'][0]
    assert verdict['per_capsule'][1]['performance']['status'] == 'PROVISIONAL'


def test_a_veto_that_also_changes_execution_evidence_still_refuses(scene):
    s = scene
    s.feedback.completed({'capsule': 'small'})
    _vetoed(s.receipts['small'], numeric={'status': 'fail'})
    verdict = {'per_capsule': [{'capsule': 'small', 'status': 'fail'}]}
    original = deepcopy(verdict)
    with pytest.raises(ValueError, match='changed before feedback publication'):
        s.feedback.project(verdict)
    assert verdict == original


def test_a_multi_mebibyte_result_is_observed_like_a_small_one(scene):
    # The largest scalability cases write 3-8 MiB results; a 2 MiB pin bound refused them and the
    # refusal failed every grade that ran them.
    s = scene
    path = s.receipts['large']
    stored = json.loads(path.read_text())
    stored['padding'] = 'x' * (3 << 20)
    path.write_text(json.dumps(stored))
    s.feedback.completed({'capsule': 'large'})
    verdict = {'per_capsule': [{'capsule': 'large', 'status': 'pass'}]}
    s.feedback.project(verdict)
    assert verdict['per_capsule'][0]['performance']['status'] == 'PROVISIONAL'


def test_observations_run_concurrently_and_projection_waits_for_them(scene, monkeypatch):
    import threading

    from merlin.common import cpu_pool

    s = scene
    monkeypatch.setattr(cpu_pool, 'compute_capacity', lambda *a, **k: 4)
    monkeypatch.setattr(cpu_pool, 'current_thread_owns_cpu_lease', lambda: False)
    barrier = threading.Barrier(3, timeout=10)  # passes only when all three observe at once

    def attach(row, root, **kw):
        barrier.wait()
        row['performance'] = {'status': 'PROVISIONAL', 'estimated_whole_soc_cycles': 1}

    monkeypatch.setattr(P, 'attach', attach)
    for name in ('small', 'large', 'tail'):
        s.feedback.completed({'capsule': name})  # returns at once: the grade's callback is not held
    verdict = {'per_capsule': [{'capsule': name, 'status': 'pass'} for name in ('small', 'large', 'tail')]}
    s.feedback.project(verdict)
    assert all(row['performance']['status'] == 'PROVISIONAL' for row in verdict['per_capsule'])


def test_a_failed_concurrent_observation_loses_only_its_own_projection(scene, monkeypatch, capsys):
    from merlin.common import cpu_pool

    s = scene
    monkeypatch.setattr(cpu_pool, 'compute_capacity', lambda *a, **k: 4)
    monkeypatch.setattr(cpu_pool, 'current_thread_owns_cpu_lease', lambda: False)

    def attach(row, root, **kw):
        if root.parent.name == 'large':
            raise ValueError('provider refused')
        row['performance'] = {'status': 'PROVISIONAL'}

    monkeypatch.setattr(P, 'attach', attach)
    s.feedback.completed({'capsule': 'small'})
    s.feedback.completed({'capsule': 'large'})
    verdict = {'per_capsule': [{'capsule': 'small', 'status': 'pass'}, {'capsule': 'large', 'status': 'pass'}]}
    s.feedback.project(verdict)
    assert verdict['per_capsule'][0]['performance']['status'] == 'PROVISIONAL'
    assert 'performance' not in verdict['per_capsule'][1]
    assert 'observation unavailable for large: ValueError: provider refused' in capsys.readouterr().out
