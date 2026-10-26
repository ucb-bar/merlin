"""Actual CAS model pinning refreshes custody before workers; no target runs."""
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context

import pytest
from test_compiler_profile_execution import execution
from test_compiler_profile_concurrency import original

from merlin.common import bundle_snapshot as B, content_store
from merlin.targetgen import capsule_runner as C, package_runtime as P
from merlin.targetgen.sandbox import bwrap
from merlin_experiments.phase1.feedback import compiler_profile_runtime as R


@pytest.fixture
def model_execution(execution, monkeypatch, tmp_path):
    monkeypatch.setenv('MERLIN_SNAPSHOT_LEASE_CACHE', '1')
    monkeypatch.setenv(content_store.LOCATION_ENV, str(tmp_path / 'own-cas'))
    owner, candidate, work = execution
    source = owner.repo / 'model'
    source.mkdir()
    (source / 'capsule.yaml').write_text('name: small_model\nkind: model\n')
    (source / 'capsule.interface.mlir').write_text('module {}\n')
    (source / 'pytorch-opset.json').write_text('{}\n')
    owner.workspace = tmp_path / 'model-workspace' / 'workspace'
    owner.workspace.mkdir(parents=True)
    owner.bundle = {'allowed': [{'path': str(source), 'mode': 'ro'}], 'host_inputs': [], 'denied': []}
    bwrap.materialize_bundle_inputs(owner.workspace, owner.bundle, repo=owner.repo)
    frozen = B.snapshot_path(B.bundle_snapshot_root(owner.workspace), source, owner.repo)
    return owner, candidate, work, frozen


def test_actual_model_pinning_refreshes_once_before_suite_workers(model_execution, monkeypatch):
    owner, candidate, work, frozen = model_execution
    # Pin through the frozen inputs' own store, so pinning links their inodes and the entry must refresh.
    shared = content_store.store_root()
    monkeypatch.setattr(content_store, 'store_root', lambda namespace=None: shared)
    guards, boundaries = [], []
    real_guard, real_verify = B._RosterGuard.__init__, B.verify_bundle_snapshot
    def guard(self, *args, **kwargs):
        guards.append(True)
        real_guard(self, *args, **kwargs)
    def verify(*args, **kwargs):
        boundaries.append(B._ACTIVE_VERIFICATION.get() is None)
        return real_verify(*args, **kwargs)
    def run(pinned, *args, **kwargs):
        replacement = executor.snapshot_entry_for_call()
        assert replacement is not stale and stale.guard.fd is None
        assert B._selected_entry() is replacement
        assert pinned[0]['__dir__'] != str(frozen)
        def command():
            with B.shared_entry_scope(executor.snapshot_entry_for_call()):
                with B.verified_snapshot_scope(owner.workspace, owner.bundle, repo=owner.repo):
                    assert B._ACTIVE_VERIFICATION.get() is not None
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = [pool.submit(copy_context().run, command) for _ in range(2)]
            for job in jobs:
                job.result(timeout=10)
        return [{'capsule': 'small_model', 'status': 'pass'}]
    monkeypatch.setattr(C, '_run_suite', run)
    with R.original_compiler_execution(owner, target_experiment=None, candidate=candidate, work_root=work):
        executor = P.active_package_executor()
        stale = executor.snapshot_entry_for_call()
        with monkeypatch.context() as patch:
            patch.setattr(B._RosterGuard, '__init__', guard)
            patch.setattr(B, 'verify_bundle_snapshot', verify)
            result = C.run_suite([{'name': 'small_model', 'kind': 'model', '__dir__': str(frozen)}], candidate)
        refreshed = executor.snapshot_entry_for_call()
        assert result[0]['status'] == 'pass'
        assert len(guards) == 1  # One re-admission, never a guard per command.
        assert boundaries.count(True) == 3  # Refresh plus BOTH fresh final reads.
        assert boundaries.count(False) == 2
    assert refreshed.guard.fd is None and stale.guard.fd is None


def test_refresh_refuses_changed_payload(model_execution):
    owner, candidate, work, frozen = model_execution
    with R.original_compiler_execution(owner, target_experiment=None, candidate=candidate, work_root=work):
        executor = P.active_package_executor()
        stale = executor.snapshot_entry_for_call()
        payload = frozen / 'pytorch-opset.json'
        payload.chmod(0o600)
        payload.write_text('{"changed":true}')
        with pytest.raises((RuntimeError, ValueError)):
            with executor.staged_suite_snapshot_scope_v1():
                pytest.fail('mutated payload admitted')
        assert executor._snapshot_entry is stale


def test_refresh_requires_issuing_thread_and_source(model_execution):
    owner, candidate, work, frozen = model_execution
    with R.original_compiler_execution(owner, target_experiment=None, candidate=candidate, work_root=work):
        executor = P.active_package_executor()
        def refresh():
            with executor.staged_suite_snapshot_scope_v1():
                pytest.fail('foreign thread admitted')
        with ThreadPoolExecutor(max_workers=1) as pool:
            with pytest.raises(ValueError, match='owner thread'):
                pool.submit(copy_context().run, refresh).result(timeout=10)
        before = executor.source_sha256
        executor.source_sha256 = 'different source'
        try:
            with pytest.raises(ValueError, match='owner thread'):
                with executor.staged_suite_snapshot_scope_v1():
                    pytest.fail('foreign source admitted')
        finally:
            executor.source_sha256 = before


def test_refresh_cannot_close_guard_in_use_by_worker(original):
    import threading
    owner, candidate, work = original
    entered, done = threading.Event(), threading.Event()
    with R.original_compiler_execution(owner, target_experiment=None, candidate=candidate, work_root=work):
        executor = P.active_package_executor()
        def worker():
            with executor.concurrent_entrypoint_scope_v1():
                entered.set()
                assert done.wait(10)
        with ThreadPoolExecutor(max_workers=1) as pool:
            job = pool.submit(copy_context().run, worker)
            assert entered.wait(10)
            try:
                with pytest.raises(ValueError, match='overlap original workers'):
                    with executor.staged_suite_snapshot_scope_v1():
                        pytest.fail('live guard replaced')
            finally:
                done.set()
                job.result(timeout=10)


def test_namespaced_model_pinning_leaves_the_frozen_inputs_and_their_entry_alone(model_execution, monkeypatch):
    owner, candidate, work, frozen = model_execution
    links = {path: path.stat().st_nlink for path in frozen.iterdir()}
    guards = []
    real_guard = B._RosterGuard.__init__

    def guard(self, *args, **kwargs):
        guards.append(True)
        real_guard(self, *args, **kwargs)

    def run(pinned, *args, **kwargs):
        assert executor.snapshot_entry_for_call() is stale  # still valid: nothing touched its inodes
        assert {path: path.stat().st_nlink for path in frozen.iterdir()} == links
        return [{'capsule': 'small_model', 'status': 'pass'}]

    monkeypatch.setattr(C, '_run_suite', run)
    with R.original_compiler_execution(owner, target_experiment=None, candidate=candidate, work_root=work):
        executor = P.active_package_executor()
        stale = executor.snapshot_entry_for_call()
        with monkeypatch.context() as patch:
            patch.setattr(B._RosterGuard, '__init__', guard)
            result = C.run_suite([{'name': 'small_model', 'kind': 'model', '__dir__': str(frozen)}], candidate)
        assert result[0]['status'] == 'pass' and guards == []  # no re-admission was needed
