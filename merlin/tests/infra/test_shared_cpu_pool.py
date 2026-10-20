"""Shared step leases stay exclusive across processes and cannot escape their owner."""

import fcntl
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context

import pytest

from merlin.common import cpu_pool as C


@pytest.fixture
def compute(monkeypatch, tmp_path):
    cpu = min(os.sched_getaffinity(0))
    root = tmp_path / "slots"
    monkeypatch.setenv("MERLIN_COMPUTE_CPUS", str(cpu))
    monkeypatch.setenv("MERLIN_COMPUTE_CPU_SLOTS", str(root))
    return cpu, root


def test_shared_build_and_simulation_processes_never_overlap_on_one_core(compute):
    cpu, _ = compute
    script = """
import json, os, time
from merlin.common.cpu_pool import shared_compute_pool
with shared_compute_pool(timeout_s=5):
    started = time.monotonic()
    allowed = sorted(os.sched_getaffinity(0))
    time.sleep(.08)
    print(json.dumps([allowed, started, time.monotonic()]))
"""

    def launch(_):
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                                timeout=10, check=True)
        return json.loads(result.stdout)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = sorted(pool.map(launch, range(4)), key=lambda row: row[1])
    assert all(row[0] == [cpu] for row in results)
    assert all(left[2] <= right[1] for left, right in zip(results, results[1:]))


def test_nested_steps_reuse_one_slot_but_incompatible_namespace_refuses(compute):
    cpu, root = compute
    original = os.sched_getaffinity(0)
    with C.shared_compute_pool(timeout_s=0):
        with C.shared_compute_pool(timeout_s=0):
            assert os.sched_getaffinity(0) == {cpu}
        with pytest.raises(RuntimeError, match="slot directory"):
            with C.selected_pool((cpu,), slot_root=root.parent / "other", timeout_s=0):
                pytest.fail("an incompatible nested lease was acquired")
        with pytest.raises(RuntimeError, match="widen"):
            with C.selected_pool((cpu, cpu + 1)):
                pytest.fail("a held slot's affinity widened")
    assert os.sched_getaffinity(0) == original
    with C.shared_compute_pool(timeout_s=0):
        pass


def test_copied_context_cannot_reuse_another_threads_active_slot(compute):
    errors = []
    with C.shared_compute_pool(timeout_s=0):
        context = copy_context()

        def attempt():
            try:
                with C.shared_compute_pool(timeout_s=0):
                    errors.append("incorrect reuse")
            except TimeoutError:
                errors.append("fresh acquisition refused")

        thread = threading.Thread(target=context.run, args=(attempt,))
        thread.start()
        thread.join(timeout=2)
        assert not thread.is_alive()
    assert errors == ["fresh acquisition refused"]


def test_expired_context_must_acquire_again(compute):
    cpu, root = compute
    with C.shared_compute_pool(timeout_s=0):
        context = copy_context()
    with (root / f"cpu-{cpu}.lock").open("r+") as competing:
        fcntl.flock(competing, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def attempt():
            with C.shared_compute_pool(timeout_s=0):
                pytest.fail("an expired lease was reused")

        with pytest.raises(TimeoutError):
            context.run(attempt)


def test_forked_child_does_not_reuse_parent_lease(compute):
    with C.shared_compute_pool(timeout_s=0):
        child = os.fork()
        if child == 0:
            try:
                with C.shared_compute_pool(timeout_s=0):
                    os._exit(2)
            except TimeoutError:
                os._exit(0)
            except BaseException:
                os._exit(3)
        _, status = os.waitpid(child, 0)
        assert os.waitstatus_to_exitcode(status) == 0


def test_exception_and_failed_affinity_restore_release_lock(compute, monkeypatch):
    cpu, root = compute
    original = os.sched_getaffinity(0)
    with pytest.raises(ValueError, match="step failed"):
        with C.shared_compute_pool(timeout_s=0):
            raise ValueError("step failed")
    assert os.sched_getaffinity(0) == original
    setter = os.sched_setaffinity
    calls = []

    def restore_failure(pid, mask):
        calls.append(mask)
        if len(calls) == 2:
            raise OSError("restore failed")
        setter(pid, mask)

    monkeypatch.setattr(os, "sched_setaffinity", restore_failure)
    try:
        with pytest.raises(OSError, match="restore failed"):
            with C.shared_compute_pool(timeout_s=0):
                pass
        with (root / f"cpu-{cpu}.lock").open("r+") as probe:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        setter(0, original)


def test_shared_selection_is_explicit_strict_and_legacy_is_unchanged(tmp_path, monkeypatch):
    for key in ("MERLIN_COMPUTE_CPUS", "MERLIN_COMPUTE_CPU_SLOTS"):
        monkeypatch.delenv(key, raising=False)
    original = os.sched_getaffinity(0)
    with C.shared_compute_pool():
        assert os.sched_getaffinity(0) == original
    assert not C.shared_compute_enabled({})
    assert C.grading_cpus({"MERLIN_GRADE_CPUS": "1,3"}) == (1, 3)
    for values in ({"MERLIN_COMPUTE_CPUS": "1"}, {"MERLIN_COMPUTE_CPU_SLOTS": str(tmp_path)},
                   {"MERLIN_COMPUTE_CPUS": "", "MERLIN_COMPUTE_CPU_SLOTS": ""},
                   {"MERLIN_COMPUTE_CPUS": "1", "MERLIN_COMPUTE_CPU_SLOTS": "relative"},
                   {"MERLIN_COMPUTE_CPUS": "1,1", "MERLIN_COMPUTE_CPU_SLOTS": str(tmp_path)}):
        with pytest.raises(ValueError):
            C.shared_compute_enabled(values)
    for budget in (-1, float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="finite"):
            with C.selected_pool((min(original),), slot_root=tmp_path / "budget", timeout_s=budget):
                pytest.fail("invalid wait budget was accepted")
