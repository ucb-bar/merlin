"""Re-admit guarded immutable inputs; never reuse a numerical verdict."""
import threading
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context

import pytest
from test_compiler_profile_concurrency import original

from merlin.common import bundle_snapshot as B
from merlin.targetgen import package_runtime as P
from merlin.verify import lean_emission_context as E
from merlin_experiments.phase1.feedback import compiler_profile_runtime as R


def payload(owner):
    return next(p for p in B.bundle_snapshot_root(owner.workspace).rglob('*.py')
                if p.is_file() and p.stat().st_size < 1024)


def test_twenty_outer_operations_refresh_once_and_each_finishes_fresh(original, monkeypatch):
    owner, candidate, work = original
    with R.original_compiler_execution(owner, target_experiment=None, candidate=candidate, work_root=work):
        executor = P.active_package_executor()
        stale = executor.snapshot_entry_for_call()
        payload(owner).touch()  # A real kernel event, with unchanged bytes.
        refreshes, reads, entries = [], [], []
        refresh, read = B.shared_entry_verification, B._payload_content
        def counted(*args, **kwargs):
            refreshes.append(True)
            return refresh(*args, **kwargs)
        def hashed(*args, **kwargs):
            reads.append(True)
            return read(*args, **kwargs)
        monkeypatch.setattr(B, 'shared_entry_verification', counted)
        monkeypatch.setattr(B, '_payload_content', hashed)
        ready = threading.Barrier(20)
        def worker():
            with executor.concurrent_entrypoint_scope_v1():
                assert B._selected_entry() is stale
                ready.wait(timeout=30)
                with E._operation_snapshot(E._ACTIVE.get()):
                    entries.append(B._selected_entry())
                    assert B._ACTIVE_VERIFICATION.get().thread is threading.current_thread()
        with ThreadPoolExecutor(max_workers=20) as pool:
            jobs = [pool.submit(copy_context().run, worker) for _ in range(20)]
            for job in jobs:
                job.result(timeout=60)
        replacement = executor.snapshot_entry_for_call()
        assert replacement is not stale and entries == [replacement] * 20
        assert len(refreshes) == 1 and len(reads) == 21  # Refresh + 20 independent finals.
        assert executor._snapshot_retired_entries == [stale]
    assert replacement.guard.fd is None and stale.guard.fd is None


@pytest.mark.parametrize('mutation', ['payload', 'marker', 'policy', 'source', 'foreign', 'alias'])
def test_refresh_refuses_changed_authority(original, mutation):
    owner, candidate, work = original
    with pytest.raises((ValueError, RuntimeError)):
        with R.original_compiler_execution(owner, target_experiment=None, candidate=candidate, work_root=work):
            executor = P.active_package_executor()
            path = payload(owner)
            if mutation == 'payload':
                path.chmod(0o600)
                path.write_bytes(path.read_bytes() + b'\n# changed\n')
            elif mutation == 'marker':
                marker = B.bundle_snapshot_root(owner.workspace) / B.MARKER
                marker.chmod(0o600)
                marker.write_bytes(marker.read_bytes() + b'\n')
            elif mutation == 'policy':
                owner.bundle['denied'].append({'path': '/changed'})
            elif mutation == 'source':
                executor.source_sha256 = 'changed'
            elif mutation == 'alias':
                path.parent.chmod(0o700)
                saved = path.with_suffix('.original')
                path.rename(saved)
                path.symlink_to(saved)
            with executor.refreshed_operation_snapshot_v1(object() if mutation == 'foreign' else owner):
                pytest.fail('changed authority admitted')


def test_failed_watch_refresh_is_not_repeated_by_waiters(original, monkeypatch):
    owner, candidate, work = original
    with R.original_compiler_execution(owner, target_experiment=None, candidate=candidate, work_root=work):
        executor = P.active_package_executor()
        stale = executor.snapshot_entry_for_call()
        payload(owner).touch()
        attempts = []
        monkeypatch.setattr(B, 'shared_entry_verification', lambda *a, **k: attempts.append(True))
        for _ in range(20):
            with executor.refreshed_operation_snapshot_v1(owner):
                assert B._selected_entry() is stale
        assert attempts == [True] and executor._snapshot_refresh_attempts == 4


def test_old_active_scope_keeps_final_check_after_new_generation(original):
    owner, candidate, work = original
    with R.original_compiler_execution(owner, target_experiment=None, candidate=candidate, work_root=work):
        executor = P.active_package_executor()
        old = executor.snapshot_entry_for_call()
        with B.verified_snapshot_scope(owner.workspace, owner.bundle, repo=owner.repo):
            old_scope = B._ACTIVE_VERIFICATION.get()
            payload(owner).touch()
            with executor.refreshed_operation_snapshot_v1(owner):
                assert B._selected_entry() is not old
                assert B._ACTIVE_VERIFICATION.get() is old_scope
            assert B._ACTIVE_VERIFICATION.get() is old_scope
        assert old in executor._snapshot_retired_entries


def test_snapshot_change_during_refresh_refuses(original, monkeypatch):
    owner, candidate, work = original
    with R.original_compiler_execution(owner, target_experiment=None, candidate=candidate, work_root=work):
        executor = P.active_package_executor()
        stale = executor.snapshot_entry_for_call()
        payload(owner).touch()
        refresh = B.shared_entry_verification
        def changed(*args, **kwargs):
            replacement = refresh(*args, **kwargs)
            replacement.document = {**replacement.document, 'n_files': -1}
            return replacement
        monkeypatch.setattr(B, 'shared_entry_verification', changed)
        with pytest.raises(ValueError, match='frozen input identity'):
            with executor.refreshed_operation_snapshot_v1(owner):
                pytest.fail('changed document admitted')
        assert executor.snapshot_entry_for_call() is stale


def test_initial_unavailable_guard_can_be_readmitted(original, monkeypatch):
    owner, candidate, work = original
    native = B.shared_entry_verification
    calls = []
    def first_unavailable(*args, **kwargs):
        calls.append(True)
        return None if len(calls) == 1 else native(*args, **kwargs)
    monkeypatch.setattr(B, 'shared_entry_verification', first_unavailable)
    with R.original_compiler_execution(owner, target_experiment=None, candidate=candidate, work_root=work):
        executor = P.active_package_executor()
        assert executor.snapshot_entry_for_call() is None
        with executor.refreshed_operation_snapshot_v1(owner):
            assert B._selected_entry() is executor.snapshot_entry_for_call()
            assert B._selected_entry() is not None
        assert len(calls) == 2


def test_foreign_process_refuses_before_refresh_lock(original, monkeypatch):
    owner, candidate, work = original
    class ForbiddenLock:
        def __enter__(self):
            pytest.fail('foreign process entered inherited lock')
        def __exit__(self, *args):
            pass
    with R.original_compiler_execution(owner, target_experiment=None, candidate=candidate, work_root=work):
        executor = P.active_package_executor()
        executor._snapshot_refresh_lock = ForbiddenLock()
        with monkeypatch.context() as patch:
            patch.setattr(R.os, 'getpid', lambda: -1)
            with pytest.raises(ValueError, match='process owner'):
                with executor.refreshed_operation_snapshot_v1(owner):
                    pytest.fail('foreign process admitted')


def test_held_cpu_lease_uses_independent_verification(original, monkeypatch):
    from merlin.common import cpu_pool
    owner, candidate, work = original
    with R.original_compiler_execution(owner, target_experiment=None, candidate=candidate, work_root=work):
        executor = P.active_package_executor()
        old = executor.snapshot_entry_for_call()
        payload(owner).touch()
        monkeypatch.setattr(cpu_pool, 'current_thread_owns_cpu_lease', lambda: True)
        monkeypatch.setattr(B, 'shared_entry_verification', lambda *a, **k: pytest.fail('lease holder attempted refresh'))
        with executor.refreshed_operation_snapshot_v1(owner):
            assert B._selected_entry() is old
        assert executor._snapshot_refresh_attempts == 0


def test_resource_exhaustion_recovers_once_for_waiters_after_backoff(original, monkeypatch):
    import errno

    owner, candidate, work = original
    with R.original_compiler_execution(owner, target_experiment=None, candidate=candidate, work_root=work):
        executor = P.active_package_executor()
        stale = executor.snapshot_entry_for_call()
        payload(owner).touch()
        native, calls, clock = B.shared_entry_verification, [], [100.0]
        def exhausted_once(*args, **kwargs):
            calls.append(True)
            if len(calls) == 1:
                kwargs['diagnostics']['resource_errno'] = errno.ENOSPC
                return None
            return native(*args, **kwargs)
        monkeypatch.setattr(B, 'shared_entry_verification', exhausted_once)
        monkeypatch.setattr(R.time, 'monotonic', lambda: clock[0])
        for _ in range(20):
            with executor.refreshed_operation_snapshot_v1(owner):
                assert B._selected_entry() is stale
        assert len(calls) == 1
        assert executor._snapshot_refresh_attempts == 0
        clock[0] = 130.0
        barrier = threading.Barrier(8)
        observed = []
        def worker():
            with executor.concurrent_entrypoint_scope_v1():
                barrier.wait(timeout=10)
                with E._operation_snapshot(E._ACTIVE.get()):
                    observed.append(B._selected_entry())
        with ThreadPoolExecutor(max_workers=8) as pool:
            jobs = [pool.submit(copy_context().run, worker) for _ in range(8)]
            for job in jobs:
                job.result(timeout=30)
        replacement = executor.snapshot_entry_for_call()
        assert replacement is not stale and observed == [replacement] * 8
        assert len(calls) == 2 and executor._snapshot_refresh_attempts == 1
        assert executor._snapshot_retired_entries == [stale]


def test_resource_recovery_does_not_bridge_unwatched_payload_mutation(original, monkeypatch):
    import errno

    owner, candidate, work = original
    with pytest.raises((ValueError, RuntimeError)):
        with R.original_compiler_execution(owner, target_experiment=None, candidate=candidate, work_root=work):
            executor = P.active_package_executor()
            path = payload(owner)
            path.touch()
            native, calls, clock = B.shared_entry_verification, [], [100.0]
            def exhausted_once(*args, **kwargs):
                calls.append(True)
                if len(calls) == 1:
                    kwargs['diagnostics']['resource_errno'] = errno.ENOSPC
                    return None
                return native(*args, **kwargs)
            monkeypatch.setattr(B, 'shared_entry_verification', exhausted_once)
            monkeypatch.setattr(R.time, 'monotonic', lambda: clock[0])
            with executor.refreshed_operation_snapshot_v1(owner):
                pass
            path.chmod(0o600)
            before = path.read_bytes()
            assert before
            path.write_bytes(bytes([before[0] ^ 1]) + before[1:])
            clock[0] = 130.0
            with executor.refreshed_operation_snapshot_v1(owner):
                pytest.fail('same-size mutation survived resource recovery')


def test_shared_entry_reports_resource_failure_without_issuing_authority(original, monkeypatch):
    import errno

    owner, candidate, work = original
    native = B._RosterGuard
    def exhausted(root, members, **kwargs):
        guard = native(root, members, **kwargs)
        guard.close()
        guard._record_resource_failure(errno.ENOSPC)
        return guard
    monkeypatch.setattr(B, '_RosterGuard', exhausted)
    diagnostics = {'old': 'must be cleared'}
    assert B.shared_entry_verification(owner.workspace, owner.bundle, repo=owner.repo,
                                       diagnostics=diagnostics) is None
    assert diagnostics == {'resource_errno': errno.ENOSPC}


@pytest.mark.parametrize('operation,error,resource', [('initialize', 24, True), ('watch', 28, True), ('watch', 13, False)])
def test_guard_retains_only_resource_syscall_failures(original, monkeypatch, operation, error, resource):
    import ctypes
    import os
    from types import SimpleNamespace

    owner, _, _ = original
    root = B.bundle_snapshot_root(owner.workspace)
    descriptors = []
    class Function:
        def __init__(self, callback):
            self.callback = callback
        def __call__(self, *args):
            return self.callback(*args)
    def initialize(flags):
        if operation == 'initialize':
            ctypes.set_errno(error)
            return -1
        descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        descriptors.append(descriptor)
        return descriptor
    def watch(*args):
        ctypes.set_errno(error)
        return -1
    library = SimpleNamespace(inotify_init1=Function(initialize), inotify_add_watch=Function(watch))
    with monkeypatch.context() as patch:
        patch.setattr(ctypes, 'CDLL', lambda *a, **k: library)
        guard = B._RosterGuard(root, B._scanned_members(root))
    assert guard.fd is None and not guard.unchanged()
    assert guard.resource_errno == (error if resource else None)
    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)
