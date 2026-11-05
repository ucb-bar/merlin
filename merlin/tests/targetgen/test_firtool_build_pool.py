"""Firtool uses an exclusive build lease, including nested and failed calls."""
import fcntl
import os
import subprocess

import pytest

from merlin.common.cpu_pool import selected_pool
from merlin.targetgen.rtl import source_selection as S


@pytest.mark.parametrize('fails', [False, True])
def test_actual_subprocess_boundary_holds_and_releases_slot(tmp_path, monkeypatch, fails):
    cpu = min(os.sched_getaffinity(0))
    original = os.sched_getaffinity(0)
    monkeypatch.setenv('MERLIN_BUILD_CPUS', str(cpu))
    monkeypatch.setenv('MERLIN_BUILD_SLOT_ROOT', str(tmp_path))
    lock = tmp_path / f'cpu-{cpu}.lock'

    def subprocess_call(command, **kwargs):
        assert os.sched_getaffinity(0) == {cpu}
        with lock.open('rb') as probe:
            with pytest.raises(BlockingIOError):
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert kwargs == {'capture_output': True, 'text': True, 'check': False}
        if fails:
            raise OSError('synthetic compiler spawn failure')
        return subprocess.CompletedProcess(command, 1, 'stdout preserved', 'stderr preserved')

    monkeypatch.setattr(S.subprocess, 'run', subprocess_call)
    if fails:
        with pytest.raises(OSError, match='spawn failure'):
            S._run_firtool(['explicit-compiler'])
    else:
        result = S._run_firtool(['explicit-compiler'])
        assert (result.returncode, result.stdout, result.stderr) == (1, 'stdout preserved', 'stderr preserved')
    assert os.sched_getaffinity(0) == original
    with lock.open('rb') as probe:
        fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_nested_same_thread_selected_slot_does_not_reacquire(tmp_path, monkeypatch):
    cpu = min(os.sched_getaffinity(0))
    monkeypatch.setenv('MERLIN_BUILD_CPUS', str(cpu))
    monkeypatch.setenv('MERLIN_BUILD_SLOT_ROOT', str(tmp_path))
    monkeypatch.setattr(S.subprocess, 'run', lambda command, **kwargs: subprocess.CompletedProcess(command, 0, '', ''))
    with selected_pool((cpu,), slot_root=tmp_path):
        assert S._run_firtool(['explicit-compiler']).returncode == 0
        assert os.sched_getaffinity(0) == {cpu}


def test_selected_pool_without_shared_slot_root_refuses(monkeypatch):
    monkeypatch.setenv('MERLIN_BUILD_CPUS', str(min(os.sched_getaffinity(0))))
    monkeypatch.delenv('MERLIN_BUILD_SLOT_ROOT', raising=False)
    with pytest.raises(ValueError, match='explicit slot root'):
        S._run_firtool(['not-launched'])
