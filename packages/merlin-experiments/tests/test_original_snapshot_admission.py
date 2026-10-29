"""The common execution route acquires identity capacity before cold custody."""
from contextlib import contextmanager
import fcntl
import hashlib
import os

import pytest

from merlin.common import bundle_snapshot as B
from merlin_experiments.phase1 import joint_control
from merlin_experiments.phase1.feedback import compiler_profile_runtime as R, snapshot_admission as A
from test_compiler_profile_execution import execution as execution  # noqa: F401


def selected(execution, monkeypatch):
    owner, candidate, work = execution
    compute = candidate.parent / 'compute'
    compute.mkdir(mode=0o700)
    monkeypatch.setenv('MERLIN_COMPUTE_CPU_SLOTS', str(compute))
    monkeypatch.setenv('MERLIN_COMPUTE_CPUS', str(min(os.sched_getaffinity(0))))
    monkeypatch.setattr(joint_control, 'selected', lambda: {})
    monkeypatch.setattr(A, 'identity', lambda service, source:
                        hashlib.sha256((source / 'compiler.py').read_bytes()).hexdigest())
    return owner, candidate, work, compute / 'immutable-snapshots'


def test_cold_verification_runs_only_inside_shared_identity_admission(execution, monkeypatch):
    owner, candidate, work, slots = selected(execution, monkeypatch)
    original = B.verified_snapshot_scope
    observed = []

    @contextmanager
    def checked(*args, **kwargs):
        fd = os.open(slots / 'slot-0', os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            observed.append(True)
        finally:
            os.close(fd)
        with original(*args, **kwargs) as document:
            yield document

    monkeypatch.setattr(B, 'verified_snapshot_scope', checked)
    with R.original_compiler_execution(owner, target_experiment=None, candidate=candidate, work_root=work):
        pass
    assert observed
    fd = os.open(slots / 'slot-0', os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(fd)


def test_source_change_while_waiting_refuses_before_cold_verification(execution, monkeypatch):
    owner, candidate, work, _ = selected(execution, monkeypatch)
    original = A.admitted

    @contextmanager
    def change(*args, **kwargs):
        with original(*args, **kwargs):
            (candidate / 'compiler.py').write_text('changed during queue wait')
            yield

    monkeypatch.setattr(A, 'admitted', change)
    monkeypatch.setattr(B, 'verified_snapshot_scope', lambda *a, **k: pytest.fail('cold work began'))
    with pytest.raises(ValueError, match='while awaiting'):
        with R.original_compiler_execution(owner, target_experiment=None, candidate=candidate, work_root=work):
            pytest.fail('changed source admitted')


def test_source_change_during_request_cannot_leave_successful_identity(execution, monkeypatch):
    owner, candidate, work, _ = selected(execution, monkeypatch)
    with pytest.raises(ValueError, match='during its admitted request'):
        with R.original_compiler_execution(owner, target_experiment=None, candidate=candidate, work_root=work):
            (candidate / 'compiler.py').write_text('changed during request')
