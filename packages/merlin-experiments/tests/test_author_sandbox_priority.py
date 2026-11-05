"""Author preparation uses the shared priority queue; no target or provider run."""

import fcntl
import os
import threading
import time
from types import SimpleNamespace

import pytest

from merlin.common import cpu_pool as C
from merlin_experiments.phase1.providers import execution as E
from merlin.targetgen.sandbox import bwrap as BW


class Refusal(RuntimeError):
    pass


@pytest.mark.parametrize("explicit_runtime", [False, True])
def test_author_verification_priority_is_local_and_preserves_refusal(tmp_path, monkeypatch, explicit_runtime):
    original_environment = os.environ.get(C.PRIORITY_ENV)
    observed = []

    def refuse(*args, **kwargs):
        observed.append(C.lease_priority())
        assert os.environ.get(C.PRIORITY_ENV) == original_environment
        raise Refusal("unchanged verification failure")

    monkeypatch.setattr(BW, "base_argv", refuse)
    context = SimpleNamespace(repo=tmp_path, descriptor=tmp_path / "descriptor")
    with C.cpu_priority(C.PREPARATION):
        with pytest.raises(Refusal, match="unchanged verification failure"):
            if explicit_runtime:
                E.codex_sandbox_command("true", tmp_path, {}, context=context,
                                        private_run_dir=tmp_path, runtime=SimpleNamespace(verify=refuse))
            else:
                E.sandbox_command("true", tmp_path, {}, context=context)
        assert C.lease_priority() == C.PREPARATION
    assert observed == [C.AUTHOR]


def test_author_sandbox_verification_precedes_queued_grade_without_preemption(tmp_path, monkeypatch):
    cpu = min(os.sched_getaffinity(0))
    slots = tmp_path / "slots"
    slots.mkdir()
    order, errors = [], []
    workers = []
    context = SimpleNamespace(repo=tmp_path, descriptor=tmp_path / "descriptor")

    def verify(*args, **kwargs):
        with C.selected_pool((cpu,), slot_root=slots, timeout_s=5):
            order.append("author verification")
        raise Refusal("stop before constructing synthetic sandbox")

    monkeypatch.setattr(BW, "base_argv", verify)

    def grade():
        with C.cpu_priority(C.GRADE), C.selected_pool((cpu,), slot_root=slots, timeout_s=5):
            order.append("grade")

    def author():
        with C.cpu_priority(C.PREPARATION):
            try:
                E.sandbox_command("true", tmp_path, {}, context=context)
            except Refusal:
                pass
            assert C.lease_priority() == C.PREPARATION

    def run(fn):
        try:
            fn()
        except BaseException as exc:
            errors.append(exc)

    def wait_for_tickets(count):
        deadline = time.monotonic() + 4
        while len(list((slots / "waiters").glob("*.wait"))) != count:
            assert time.monotonic() < deadline
            time.sleep(.005)

    with (slots / f"cpu-{cpu}.lock").open("w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        try:
            for fn, count in ((grade, 1), (author, 2)):
                worker = threading.Thread(target=run, args=(fn,))
                workers.append(worker)
                worker.start()
                wait_for_tickets(count)
            assert order == []  # The live holder was never preempted.
        finally:
            fcntl.flock(held, fcntl.LOCK_UN)
            for worker in workers:
                worker.join(6)
    assert not errors
    assert all(not worker.is_alive() for worker in workers)
    assert order == ["author verification", "grade"]
    assert not list((slots / "waiters").glob("*.wait"))
