import contextlib
import json

import pytest

from merlin_experiments.phase1.feedback import setup_timing as T


def test_boundaries_visible_and_original_exception_preserved(tmp_path):
    visits = []
    @contextlib.contextmanager
    def manager():
        visits.append('enter')
        try:
            yield 'value'
        finally:
            visits.append('exit')
    with T.selected(tmp_path, 'a' * 64) as timeline:
        with pytest.raises(ValueError, match='original'):
            with T.timed_context('verify', manager()) as value:
                assert value == 'value'
                doc = json.loads(timeline.path.read_text())
                assert doc['stages'][0]['status'] == 'returned'
                with T.stage('policy'):
                    raise ValueError('original')
    doc = json.loads(timeline.path.read_text())
    assert visits == ['enter', 'exit']
    assert [r['stage'] for r in doc['stages']] == ['verify.enter', 'policy', 'verify.exit']
    assert doc['stages'][1]['exception_type'] == 'ValueError'
    assert all(r['elapsed_s'] >= 0 for r in doc['stages'])
    assert doc['authority'] == 'diagnostic_only'


def test_suppression_and_failed_entry_not_changed(tmp_path):
    @contextlib.contextmanager
    def suppress():
        try:
            yield
        except ValueError:
            pass
    with T.selected(tmp_path, 'b' * 64) as timeline:
        with T.timed_context('suppress', suppress()):
            raise ValueError('handled')
        @contextlib.contextmanager
        def refuse():
            raise RuntimeError('custody')
            yield
        with pytest.raises(RuntimeError, match='custody'):
            with T.timed_context('refuse', refuse()):
                pytest.fail('refused context body ran')
    rows = json.loads(timeline.path.read_text())['stages']
    assert [r['stage'] for r in rows] == ['suppress.enter', 'suppress.exit', 'refuse.enter']
    assert rows[-1]['status'] == 'raised'


def test_concurrent_capsule_identity_is_exact_and_contexts_are_restored(tmp_path):
    import os
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from contextvars import copy_context

    barrier = threading.Barrier(2)
    def worker(name, tier):
        with T.capsule_scope(name, tier=tier):
            with T.selected(tmp_path, name + '-source') as timeline:
                with T.stage('package_policy'):
                    barrier.wait(timeout=10)
            document = json.loads(timeline.path.read_text())
            assert document['capsule'] == name and document['tier'] == tier
            assert document['pid'] == os.getpid()
            assert document['native_tid'] == threading.get_native_id()
            assert document['thread_ident'] == threading.get_ident()
            assert document['thread_name'] == threading.current_thread().name
            assert document['stages'][0]['native_tid'] == document['native_tid']
            assert document['source_sha256'] == name + '-source'
        assert T._CAPSULE.get() == ('parent', None)
        assert T._ACTIVE.get() is None
        return document

    with T.capsule_scope('parent'):
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = [pool.submit(copy_context().run, worker, name, tier)
                    for name, tier in [('first', None), ('second', 'L3')]]
            documents = [job.result(timeout=10) for job in jobs]
        assert T._CAPSULE.get() == ('parent', None)
    assert T._CAPSULE.get() == (None, None)
    assert documents[0]['native_tid'] != documents[1]['native_tid']


def test_native_capsule_worker_supplies_name_without_inventing_tier(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from merlin.targetgen import capsule_runner as C

    monkeypatch.setenv('MERLIN_OPERATOR_GRADE_PROCESSES', '0')
    monkeypatch.setattr(C, 'load_package', lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(C, 'integrity_scan', lambda p: None)
    monkeypatch.setattr(C, 'build_package', lambda p: None)
    monkeypatch.setattr(C, '_split_ineligible', lambda caps, target: (caps, []))
    monkeypatch.setattr(C._tier_policy, 'covering_set', lambda caps: [])
    monkeypatch.setattr(C, '_CALIBRATION_CAP', 0)
    observed = {}
    def capsule(cap, *args, **kwargs):
        with T.selected(tmp_path, 'native-source') as timeline:
            with T.stage('package_policy'):
                pass
        observed[cap['name']] = json.loads(timeline.path.read_text())
        return {'capsule': cap['name'], 'status': 'pass', 'tiers': {}}
    monkeypatch.setattr(C, 'run_capsule', capsule)
    C._run_suite([{'name': 'A'}, {'name': 'B'}], tmp_path,
                 runs_root=tmp_path / 'runs', max_workers=2)
    assert set(observed) == {'A', 'B'}
    for name, row in observed.items():
        assert row['capsule'] == name and row['tier'] is None
        assert row['authority'] == 'diagnostic_only'
    assert T._CAPSULE.get() == (None, None)


def test_live_nested_stage_retains_timeout_and_exact_request_identity(tmp_path):
    import subprocess

    @T.measured('link')
    def link():
        document = json.loads(timeline.path.read_text())
        assert document['current_stage'] == 'link'
        assert document['stages'][-1]['parent_stage_id'] == 0
        raise subprocess.TimeoutExpired(['compiler'], 7)

    with T.capsule_scope('case', tier='L2', request_id='request', snapshot_id='snapshot'):
        with T.selected(tmp_path, 'source', operation='native_capsule') as timeline:
            with pytest.raises(subprocess.TimeoutExpired):
                with T.stage('emission'):
                    link()
            with T.stage('result_processing'):
                pass
    document = json.loads(timeline.path.read_text())
    assert document['request_id'] == 'request' and document['snapshot_id'] == 'snapshot'
    assert document['source_sha256'] == 'source' and document['capsule'] == 'case'
    assert document['current_stage'] is None and document['last_stage'] == 'result_processing'
    assert document['last_failure']['exception_type'] == 'TimeoutExpired'
    assert document['last_failure']['stage'] == 'link'
    assert 'must not be summed' in document['intervals']


def test_nested_invocation_links_parent_without_claiming_missing_ids(tmp_path):
    with T.selected(tmp_path, None, operation='native_capsule') as outer:
        with T.selected(tmp_path, 'source', operation='lower_interface') as inner:
            with T.stage('compiler_command'):
                pass
    document = json.loads(inner.path.read_text())
    assert document['parent_timeline'] == str(outer.path)
    assert 'request_id' not in document and 'snapshot_id' not in document
    assert document['operation'] == 'lower_interface'
    assert T._ACTIVE.get() is None


def test_policy_stage_observation_preserves_decision_and_refusal(tmp_path, monkeypatch):
    from merlin.targetgen.sandbox import host_surfaces as H, bwrap as B

    visits = []
    def record(label, result):
        def call(*args, **kwargs):
            visits.append(label)
            return result
        return call
    monkeypatch.setattr(H, 'host_input_surfaces', record('private', []))
    monkeypatch.setattr(B, 'answer_surfaces', record('answers', []))
    monkeypatch.setattr(B, 'apply_answer_masks', record('masks', ['original-mount']))
    monkeypatch.setattr(B, 'coverage_gap', record('coverage', []))
    with T.selected(tmp_path, 'source') as timeline:
        assert H._final_answer_policy([], None, tmp_path, {}, repo=tmp_path) == (
            ['original-mount', '--unsetenv', 'MERLIN_FROZEN_PYTHON_CONTEXT'], ())
    assert visits == ['private', 'answers', 'masks', 'coverage']
    assert len(json.loads(timeline.path.read_text())['stages']) == 5
    monkeypatch.setattr(B, 'coverage_gap', lambda *args: ['exposed'])
    with T.selected(tmp_path, 'source') as refused:
        with pytest.raises(RuntimeError, match='remains exposed'):
            H._final_answer_policy([], None, tmp_path, {}, repo=tmp_path)
    assert json.loads(refused.path.read_text())['last_failure']['stage'] == 'package_policy.apply_masks'


def test_submission_wait_is_measured_only_when_explicitly_selected(tmp_path):
    with T.queued():
        with T.selected(tmp_path, 'source', queue_wait=True) as outer:
            with T.selected(tmp_path, 'source') as inner:
                with T.stage('emission'):
                    pass
    rows = json.loads(outer.path.read_text())['stages']
    assert rows[0]['stage'] == 'scheduler_submission_wait' and rows[0]['elapsed_s'] >= 0
    assert [row['stage'] for row in json.loads(inner.path.read_text())['stages']] == ['emission']


def test_native_oracle_boundary_retains_original_observer_and_timeout(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import subprocess
    from merlin.targetgen import oracle_readback as O, executed_program
    from merlin.targetgen.contract import compile as C

    observed = []
    monkeypatch.setattr(O, 'validate_engine_policy', lambda *args, **kwargs: None)
    monkeypatch.setattr(executed_program, 'retain', lambda result, **kwargs: result)
    policy = SimpleNamespace(transport='console')
    def execute(*args, **kwargs):
        assert json.loads(timeline.path.read_text())['current_stage'] == 'oracle_compile_link_admit'
        kwargs['before_execution'](tmp_path / 'program.elf')
        assert json.loads(timeline.path.read_text())['current_stage'] == 'oracle_simulation_readback.test'
        raise subprocess.TimeoutExpired(['engine'], 7)
    monkeypatch.setattr(C, 'run_on_oracle', execute)
    with T.selected(tmp_path, 'source') as timeline:
        with pytest.raises(subprocess.TimeoutExpired):
            O._run({}, '', sim='test', target='target', backend=object(), workdir=tmp_path,
                   timeout=7, policy=policy, before_execution=lambda elf: observed.append(elf))
    assert observed == [tmp_path / 'program.elf']
    result = json.loads(timeline.path.read_text())
    assert result['last_failure']['stage'] == 'oracle_simulation_readback.test'
    assert result['stages'][0]['status'] == 'returned'
