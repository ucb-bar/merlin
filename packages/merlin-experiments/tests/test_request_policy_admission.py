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
