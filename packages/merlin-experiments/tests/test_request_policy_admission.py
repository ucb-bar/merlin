"""Request policy waiters do not reserve CPUs before cold preparation finishes."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import threading

import pytest

from merlin_experiments.phase1.feedback import compiler_profile_runtime as R
from merlin_experiments.phase2 import campaign as C
from test_compiler_profile_execution import execution as execution  # noqa: F401


@pytest.mark.parametrize('first_fails', [False, True])
def test_policy_waits_before_compute_and_releases_after_failure(execution, monkeypatch, first_fails):
    owner, candidate, work = execution
    executor = R._Executor(owner, None, candidate, work)
    entered, release, second_started, second_leased = (threading.Event() for _ in range(4))
    local = threading.local()

    @contextmanager
    def compute():
        if getattr(local, 'second', False):
            second_leased.set()
        local.leased = True
        try:
            yield
        finally:
            local.leased = False

    def policy(*args, **kwargs):
        assert local.leased
        if not getattr(local, 'second', False):
            entered.set()
            assert release.wait(2)
            if first_fails:
                raise ValueError('original policy refusal')
        return 'policy'

    monkeypatch.setattr(R, '_verification_compute', compute)
    monkeypatch.setattr(C, 'package_sandbox_policy', policy)

    def second():
        local.second = True
        second_started.set()
        return executor._policy(work / 'second', candidate)

    with ThreadPoolExecutor(max_workers=2) as workers:
        first = workers.submit(executor._policy, work / 'first', candidate)
        assert entered.wait(2)
        later = workers.submit(second)
        assert second_started.wait(2)
        assert not second_leased.wait(0.05)
        release.set()
        if first_fails:
            with pytest.raises(ValueError, match='original policy refusal'):
                first.result()
        else:
            assert first.result() == 'policy'
        assert later.result() == 'policy'
    assert second_leased.is_set()


def test_waiters_after_first_success_compute_warm_policies_concurrently(execution, monkeypatch):
    owner, candidate, work = execution
    executor = R._Executor(owner, None, candidate, work)
    cold_entered, cold_release = threading.Event(), threading.Event()
    both_warm = threading.Barrier(2)
    count_lock = threading.Lock()
    started = []

    @contextmanager
    def compute():
        yield

    def policy(_target, workspace, *_args, **_kwargs):
        with count_lock:
            started.append(workspace.name)
        if workspace.name == 'cold':
            cold_entered.set()
            assert cold_release.wait(2)
        else:
            # Neither already-queued waiter may retain the cold admission lock.
            both_warm.wait(timeout=2)
        return workspace.name

    monkeypatch.setattr(R, '_verification_compute', compute)
    monkeypatch.setattr(C, 'package_sandbox_policy', policy)
    with ThreadPoolExecutor(max_workers=3) as workers:
        cold = workers.submit(executor._policy, work / 'cold', candidate)
        assert cold_entered.wait(2)
        warm = [workers.submit(executor._policy, work / f'warm-{i}', candidate) for i in range(2)]
        cold_release.set()
        assert cold.result() == 'cold'
        assert [f.result() for f in warm] == ['warm-0', 'warm-1']
    assert set(started) == {'cold', 'warm-0', 'warm-1'}


def test_failed_cold_policy_does_not_mark_prepared(execution, monkeypatch):
    owner, candidate, work = execution
    executor = R._Executor(owner, None, candidate, work)
    @contextmanager
    def compute():
        yield
    monkeypatch.setattr(R, '_verification_compute', compute)
    def refuse(*args, **kwargs):
        raise ValueError('real policy refusal')
    monkeypatch.setattr(C, 'package_sandbox_policy', refuse)
    with pytest.raises(ValueError, match='real policy refusal'):
        executor._policy(work / 'bad', candidate)
    assert executor._policy_prepared is False
    monkeypatch.setattr(C, 'package_sandbox_policy', lambda *a, **kw: 'rechecked')
    assert executor._policy(work / 'retry', candidate) == 'rechecked'
    assert executor._policy_prepared is True
