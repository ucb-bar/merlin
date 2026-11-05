"""Synchronous capsule arithmetic shares CPUs with independent simulator jobs."""

import fcntl
import os

import pytest

from merlin.common.cpu_pool import shared_compute_pool
from merlin.targetgen.capsule_runner import _grade_cpu_step


def test_grade_calculation_leases_and_releases_selected_cpu(tmp_path, monkeypatch):
    cpu = min(os.sched_getaffinity(0))
    original = os.sched_getaffinity(0)
    slots = tmp_path / "slots"
    monkeypatch.setenv("MERLIN_COMPUTE_CPUS", str(cpu))
    monkeypatch.setenv("MERLIN_COMPUTE_CPU_SLOTS", str(slots))

    def calculation():
        assert os.sched_getaffinity(0) == {cpu}
        with (slots / f"cpu-{cpu}.lock").open("r+") as probe:
            with pytest.raises(BlockingIOError):
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return 17

    assert _grade_cpu_step(calculation) == 17
    assert os.sched_getaffinity(0) == original
    # A following simulator can acquire exactly the same shared slot.
    with shared_compute_pool(timeout_s=0):
        assert os.sched_getaffinity(0) == {cpu}


def test_grade_calculation_releases_cpu_on_failure(tmp_path, monkeypatch):
    cpu = min(os.sched_getaffinity(0))
    original = os.sched_getaffinity(0)
    monkeypatch.setenv("MERLIN_COMPUTE_CPUS", str(cpu))
    monkeypatch.setenv("MERLIN_COMPUTE_CPU_SLOTS", str(tmp_path / "slots"))

    def calculation():
        raise ValueError("calculation failed")

    with pytest.raises(ValueError, match="calculation failed"):
        _grade_cpu_step(calculation)
    assert os.sched_getaffinity(0) == original
    with shared_compute_pool(timeout_s=0):
        pass
