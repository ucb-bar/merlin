"""Framework operation publication keeps exact original request/watch custody."""
import os
import select
import threading
from contextlib import contextmanager

import pytest
from test_compiler_profile_concurrency import original
from test_concurrent_snapshot_refresh import payload

from merlin.common import bundle_snapshot as B
from merlin.targetgen import package_runtime as P
from merlin.verify import lean_emission_context as E
from merlin_experiments.phase1.feedback import compiler_profile_runtime as R


@pytest.mark.parametrize('mutation', ['payload', 'marker', 'policy'])
def test_publication_rejects_mid_operation_changes(original, mutation):
    owner, candidate, work = original
    with pytest.raises((RuntimeError, ValueError)):
        with R.original_compiler_execution(owner, target_experiment=None, candidate=candidate, work_root=work):
            with E._operation_snapshot(E._ACTIVE.get()):
                if mutation == 'policy':
                    owner.bundle['denied'].append({'path': '/changed'})
                else:
                    path = payload(owner) if mutation == 'payload' else B.bundle_snapshot_root(owner.workspace) / B.MARKER
                    # An external worker event must invalidate publication too.
                    def mutate():
                        path.chmod(0o600)
                        path.write_bytes(path.read_bytes() + b'\n')
                    worker = threading.Thread(target=mutate)
                    worker.start()
                    worker.join(timeout=5)
                    assert not worker.is_alive()


def test_guard_event_with_restored_bytes_requires_full_read(original, monkeypatch):
    owner, candidate, work = original
    with R.original_compiler_execution(owner, target_experiment=None, candidate=candidate, work_root=work):
        reads = []
        original_read = B._payload_content
        def read(*a, **k):
            reads.append(True)
            return original_read(*a, **k)
        monkeypatch.setattr(B, '_payload_content', read)
        with E._operation_snapshot(E._ACTIVE.get()):
            path = payload(owner)
            before = path.read_bytes()
            path.chmod(0o600)
            path.write_bytes(before + b'\n')
            path.write_bytes(before)
        assert reads  # An event cannot be erased by restoring the original bytes.


def test_implicit_entry_cannot_select_guarded_publication(original, monkeypatch):
    owner, candidate, work = original
    native = R._Executor.refreshed_operation_snapshot_v1
    @contextmanager
    def implicit(self, service):
        with native(self, service):
            yield None
    monkeypatch.setattr(R._Executor, 'refreshed_operation_snapshot_v1', implicit)
    with R.original_compiler_execution(owner, target_experiment=None, candidate=candidate, work_root=work):
        reads = []
        original_read = B._payload_content
        def read(*a, **k):
            reads.append(True)
            return original_read(*a, **k)
        monkeypatch.setattr(B, '_payload_content', read)
        with E._operation_snapshot(E._ACTIVE.get()):
            pass
        assert len(reads) == 1  # Original independent final read retained.


@pytest.mark.skipif(not hasattr(os, 'fork'), reason='requires fork custody test')
def test_fork_cannot_borrow_parent_operation_entry(original):
    owner, candidate, work = original
    with R.original_compiler_execution(owner, target_experiment=None, candidate=candidate, work_root=work):
        executor = P.active_package_executor()
        entry = executor.snapshot_entry_for_call()
        read_fd, write_fd = os.pipe()
        child = os.fork()
        if child == 0:
            os.close(read_fd)
            try:
                with E._operation_snapshot(E._ACTIVE.get()):
                    result = b'admitted'
            except (ValueError, RuntimeError):
                result = b'refused'
            os.write(write_fd, result)
            os.close(write_fd)
            os._exit(0)
        os.close(write_fd)
        try:
            ready, _, _ = select.select([read_fd], [], [], 5)
            if not ready:
                os.kill(child, 9)
                pytest.fail('fork waited on an inherited owner lock')
            assert os.read(read_fd, 32) == b'refused'
        finally:
            os.close(read_fd)
            os.waitpid(child, 0)
        assert entry.reusable(entry.root, entry.repo, entry.policy_bytes)
