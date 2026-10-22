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
