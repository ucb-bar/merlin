"""Background custody yields queue admission to useful work, never its checks."""

import fcntl
import os
import threading
import time

import pytest

from merlin.common import cpu_pool as C
from merlin_experiments.phase1.feedback.compiler_profile_runtime import _verification_compute


def test_verification_preserves_exception_priority_and_environment(monkeypatch):
    monkeypatch.delenv('MERLIN_COMPUTE_CPUS', raising=False)
    monkeypatch.delenv('MERLIN_COMPUTE_CPU_SLOTS', raising=False)
    monkeypatch.setenv(C.PRIORITY_ENV, str(C.GRADE))
    with C.cpu_priority(C.AUTHOR):
        with pytest.raises(ValueError, match='custody failed'):
            with _verification_compute():
                assert C.lease_priority() == C.VERIFICATION
                assert os.environ[C.PRIORITY_ENV] == str(C.GRADE)
                raise ValueError('custody failed')
        assert C.lease_priority() == C.AUTHOR


def test_ready_execution_and_calibration_precede_queued_verification(tmp_path, monkeypatch):
    cpu = min(os.sched_getaffinity(0))
    slots = tmp_path / 'slots'
    slots.mkdir()
    monkeypatch.setenv('MERLIN_COMPUTE_CPUS', str(cpu))
    monkeypatch.setenv('MERLIN_COMPUTE_CPU_SLOTS', str(slots))
    order, errors, workers = [], [], []

    def worker(level):
        try:
            with C.cpu_priority(level):
                with (_verification_compute() if level == C.VERIFICATION else C.shared_compute_pool(timeout_s=5)):
                    order.append(level)
        except BaseException as exc:
            errors.append(exc)

    with (slots / f'cpu-{cpu}.lock').open('w') as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        try:
            # Admit verification first, then ensure every useful class overtakes
            # its queued ticket when the current holder releases the CPU.
            for count, level in enumerate((C.VERIFICATION, C.AUXILIARY, C.PREPARATION, C.GRADE, C.AUTHOR), 1):
                thread = threading.Thread(target=worker, args=(level,))
                workers.append(thread)
                thread.start()
                deadline = time.monotonic() + 4
                while len(list((slots / 'waiters').glob('*.wait'))) != count:
                    assert time.monotonic() < deadline
                    time.sleep(.005)
            assert order == []
        finally:
            fcntl.flock(held, fcntl.LOCK_UN)
            for thread in workers:
                thread.join(6)
    assert not errors
    assert all(not thread.is_alive() for thread in workers)
    assert order == [C.AUTHOR, C.GRADE, C.PREPARATION, C.AUXILIARY, C.VERIFICATION]
