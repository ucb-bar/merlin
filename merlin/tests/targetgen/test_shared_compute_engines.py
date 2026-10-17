"""Build and simulator boundaries share one CPU lease without changing oracle tiers."""

import fcntl
import os
import subprocess
import threading
from contextlib import contextmanager

import pytest

from merlin.common import cpu_pool
from merlin.targetgen import rtl_engine_policy as policy
from merlin.targetgen.contract import compile as compiler
from merlin.targetgen.rtl import source_selection


@pytest.fixture
def shared_pool(tmp_path, monkeypatch):
    cpu = min(os.sched_getaffinity(0))
    slots = tmp_path / "compute"
    monkeypatch.setenv("MERLIN_COMPUTE_CPUS", str(cpu))
    monkeypatch.setenv("MERLIN_COMPUTE_CPU_SLOTS", str(slots))
    monkeypatch.setenv("MERLIN_GSIM_SLOT_ROOT", str(tmp_path / "native"))
    monkeypatch.setitem(policy.CAPSULE_WORKER_CAP, "gsim", 1)
    monkeypatch.setattr(policy, "_native_gsim_census", lambda **_: policy._NativeCensus(0, ()))
    return cpu, slots


def assert_cpu_owned(cpu, slots):
    assert os.sched_getaffinity(0) == {cpu}
    with (slots / f"cpu-{cpu}.lock").open("rb") as stream:
        with pytest.raises(BlockingIOError):
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)


@pytest.mark.parametrize("engine", ["spike", "verilator", "vcs", "gsim"])
def test_all_engine_scopes_hold_the_same_compute_namespace(shared_pool, engine):
    cpu, slots = shared_pool
    original = os.sched_getaffinity(0)
    with policy.simulation_runtime_slot(engine, wait_timeout_s=1):
        assert_cpu_owned(cpu, slots)
        with cpu_pool.shared_compute_pool(timeout_s=0):
            assert_cpu_owned(cpu, slots)
    assert os.sched_getaffinity(0) == original
    with (slots / f"cpu-{cpu}.lock").open("rb") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_gsim_waits_for_compute_before_reserving_native_capacity(shared_pool, tmp_path, monkeypatch):
    cpu, slots = shared_pool
    slots.mkdir()
    native = tmp_path / "native"
    with (slots / f"cpu-{cpu}.lock").open("wb") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(TimeoutError):
            with policy.gsim_runtime_slot(wait_timeout_s=0):
                pytest.fail("busy compute CPU must prevent a native reservation")
    assert not native.exists()


def test_shared_gsim_ignores_fixed_legacy_slot_cpu_mapping(shared_pool, monkeypatch):
    cpu, slots = shared_pool
    monkeypatch.setenv("MERLIN_GSIM_CPUS", "obsolete,invalid,fixed-roster")
    assert policy._gsim_cpu_roster() is None
    with policy.gsim_runtime_slot(wait_timeout_s=1):
        assert_cpu_owned(cpu, slots)
        with policy.gsim_runtime_slot(wait_timeout_s=0):
            assert_cpu_owned(cpu, slots)


def test_busy_native_capacity_leaves_compute_available_for_builds(shared_pool, monkeypatch):
    ready, finished = threading.Event(), threading.Event()
    errors = []
    original = policy._gsim_runtime_slot

    @contextmanager
    def observed(*args, **kwargs):
        ready.set()
        with original(*args, **kwargs):
            yield

    monkeypatch.setattr(policy, '_gsim_runtime_slot', observed)

    def wait_for_native():
        try:
            with policy.gsim_runtime_slot(wait_timeout_s=0.4):
                errors.append('unexpected native capacity')
        except TimeoutError:
            pass
        finally:
            finished.set()

    # A different native owner fills the cap without taking our compute CPU.
    with original(wait_timeout_s=0):
        worker = threading.Thread(target=wait_for_native)
        worker.start()
        try:
            assert ready.wait(1)
            with cpu_pool.shared_compute_pool(timeout_s=0.2):
                assert not finished.is_set()
        finally:
            worker.join(2)
    assert finished.is_set() and not errors


def test_gsim_native_retries_keep_their_compute_queue_position(shared_pool, monkeypatch):
    positions, attempts = [], []
    original = cpu_pool.shared_compute_pool

    @contextmanager
    def recorded(**kwargs):
        positions.append(kwargs.get("queued_since_ns"))
        with original(**kwargs):
            yield

    @contextmanager
    def native(**kwargs):
        attempts.append(kwargs)
        if len(attempts) < 3:
            raise TimeoutError("native capacity busy")
        yield

    monkeypatch.setattr(cpu_pool, "shared_compute_pool", recorded)
    monkeypatch.setattr(policy, "_gsim_runtime_slot", native)
    with policy.gsim_runtime_slot(wait_timeout_s=5):
        pass
    assert len(positions) == 3 and positions[0] is not None and len(set(positions)) == 1


def test_kernel_timeout_is_not_retried_as_an_admission_failure(shared_pool):
    with pytest.raises(TimeoutError, match='actual kernel timeout'):
        with policy.gsim_runtime_slot(wait_timeout_s=1):
            raise TimeoutError('actual kernel timeout')


@pytest.mark.parametrize("owner", ["firtool", "contract", "llvm"])
def test_actual_build_boundary_reuses_native_compute_lease(shared_pool, tmp_path, monkeypatch, owner):
    cpu, slots = shared_pool
    monkeypatch.setenv("MERLIN_BUILD_CPUS", "invalid-legacy-roster")
    monkeypatch.delenv("MERLIN_BUILD_SLOT_ROOT", raising=False)

    def build(argv, **kwargs):
        assert_cpu_owned(cpu, slots)
        return subprocess.CompletedProcess(argv, 0, "original output", "")

    if owner == "firtool":
        monkeypatch.setattr(source_selection.subprocess, "run", build)

        def invoke():
            return source_selection._run_firtool(["explicit-compiler"])
    else:
        from merlin.runtime import build_runtime

        monkeypatch.setattr(build_runtime, "run_in_build_view", build)
        if owner == "contract":

            def invoke():
                return compiler._observed_run(["explicit-compiler"], workdir=tmp_path, stage="object", timeout=1)
        else:
            from merlin.llvmlower import codegen

            def invoke():
                return codegen._run(["explicit-compiler"], outputs=(tmp_path / "object.o",), timeout_s=1)

    with policy.simulation_runtime_slot("spike", wait_timeout_s=1):
        result = invoke()
        if owner == "llvm":
            assert result is None
        else:
            assert result.stdout == "original output"


def test_standalone_spike_launch_uses_the_common_compute_lease(shared_pool, monkeypatch):
    from merlin.runtime.backends import spike

    cpu, slots = shared_pool

    def launch(argv, **kwargs):
        assert_cpu_owned(cpu, slots)
        assert kwargs["timeout"] == 1
        return subprocess.CompletedProcess(argv, 0, "DONE\n", "")

    monkeypatch.setattr(spike.subprocess, "run", launch)
    assert spike.run_elf("selected.elf", timeout=1) == "DONE\n"

