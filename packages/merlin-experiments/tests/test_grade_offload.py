"""Grader calculations run out of process on the leased CPU, with in-process results and exceptions."""

import os
import threading
import time

import pytest

from merlin.targetgen import grade_offload
from merlin.targetgen.capsule_runner import _grade_cpu_stages, _grade_cpu_step


@pytest.fixture(autouse=True)
def _private_content_store(tmp_path, monkeypatch):
    """Captured compiler source shares bytes through the content store; keep this test's private."""
    monkeypatch.setenv("MERLIN_BUNDLE_CAS", str(tmp_path / "cas"))

CALLS = []


def _where(value):
    return os.getpid(), sorted(os.sched_getaffinity(0)), value * 2


def _raises(value):
    CALLS.append(os.getpid())
    raise KeyError(f"missing {value}")


def _needs_slot():
    from merlin.common.cpu_pool import shared_compute_pool

    with shared_compute_pool():
        return os.getpid()


def _swallows_slot_refusal():
    from merlin.common.cpu_pool import shared_compute_pool

    try:
        with shared_compute_pool():
            return "computed", os.getpid()
    except Exception:  # noqa: BLE001 -- a step that degrades instead of raising
        return "degraded", os.getpid()


def _spin(n):
    total = 0
    for i in range(n):
        total += i * i
    return total


@pytest.fixture(autouse=True)
def _fresh_pool(monkeypatch):
    monkeypatch.delenv(grade_offload.ENV, raising=False)
    yield
    grade_offload.shutdown()


def test_registered_calculation_runs_in_a_worker_on_the_leased_cpu(tmp_path, monkeypatch):
    cpu = min(os.sched_getaffinity(0))
    monkeypatch.setenv("MERLIN_COMPUTE_CPUS", str(cpu))
    monkeypatch.setenv("MERLIN_COMPUTE_CPU_SLOTS", str(tmp_path / "slots"))
    grade_offload.register(_where)
    pid, cpus, doubled = _grade_cpu_step(_where, 21)
    assert pid != os.getpid() and cpus == [cpu] and doubled == 42


def test_unregistered_or_disabled_calculations_stay_in_process(monkeypatch):
    def local(value):
        return os.getpid(), value

    assert _grade_cpu_step(local, 1) == (os.getpid(), 1)
    grade_offload.register(_where)
    monkeypatch.setenv(grade_offload.ENV, "0")
    assert _grade_cpu_step(_where, 1)[0] == os.getpid()


def test_a_raising_calculation_raises_its_own_exception_in_process():
    grade_offload.register(_raises)
    CALLS.clear()
    with pytest.raises(KeyError, match="missing 3"):
        grade_offload.call(_raises, 3)
    # The worker's attempt is discarded; the exception the caller sees is raised in process.
    assert CALLS == [os.getpid()]


def test_unpicklable_arguments_stay_in_process():
    grade_offload.register(_where)
    lock = threading.Lock()
    pid, _, doubled = grade_offload.call(_where, [lock])
    assert pid == os.getpid() and doubled == [lock, lock]


def test_the_grader_calculations_are_registered():
    names = set(_grade_cpu_stages().values())
    assert {"golden", "l0_reference", "l1_simulate", "numeric_compare", "policy_match",
            "capture_compiler_source", "finish_compiler_source"} <= names


def test_compiler_source_capture_runs_out_of_process_with_identical_evidence(tmp_path):
    from merlin.targetgen.capsule_source_identity import _capture_compiler_source, _finish_compiler_source

    package = tmp_path / "package"
    (package / "compiler").mkdir(parents=True)
    (package / "compiler" / "pass.py").write_text("x = 1\n")
    (package / "manifest.yaml").write_text("name: p\n")
    _grade_cpu_stages()
    offloaded = _grade_cpu_step(_capture_compiler_source, package, tmp_path / "a")
    inline = _capture_compiler_source(package, tmp_path / "b")
    assert offloaded == inline and (tmp_path / "a" / offloaded["snapshot"] / "manifest.yaml").is_file()
    assert _grade_cpu_step(_finish_compiler_source, offloaded, tmp_path / "a")["status"] == "stable"


@pytest.mark.skipif(len(os.sched_getaffinity(0)) < 2, reason="needs two CPUs")
def test_threads_no_longer_serialize_on_one_interpreter():
    grade_offload.register(_spin)
    n = 3_000_000
    grade_offload.call(_spin, 10)  # start a worker outside the measurement
    started = time.monotonic()
    _spin(n)
    alone = time.monotonic() - started
    threads = [threading.Thread(target=grade_offload.call, args=(_spin, n)) for _ in range(2)]
    started = time.monotonic()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    # Serialized on one interpreter, two would take about twice as long as one.
    assert time.monotonic() - started < 1.7 * alone + 1.0


def test_workers_never_rerun_the_callers_main_script(tmp_path):
    import subprocess
    import sys

    script = tmp_path / "unguarded.py"
    script.write_text(
        "import os\n"
        "from merlin.targetgen import grade_offload\n"
        "from merlin.common.tree_hash import hash_tree\n"
        "print('top-level ran', flush=True)\n"
        "grade_offload.register(os.getpid)\n"  # a builtin: importable as posix.getpid
        "print('pid differs', grade_offload.call(os.getpid) != os.getpid(), flush=True)\n"
    )
    out = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=60,
                         env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}).stdout
    assert out.count("top-level ran") == 1 and "pid differs True" in out


@pytest.mark.parametrize("function", [_needs_slot, _swallows_slot_refusal])
def test_a_calculation_that_needs_a_slot_runs_on_the_callers_lease(tmp_path, monkeypatch, function):
    """The caller holds its slot while it waits; the worker must not queue for another one."""
    cpu = min(os.sched_getaffinity(0))
    monkeypatch.setenv("MERLIN_COMPUTE_CPUS", str(cpu))  # one slot, held by the caller below
    monkeypatch.setenv("MERLIN_COMPUTE_CPU_SLOTS", str(tmp_path / "slots"))
    grade_offload.register(function)
    started = time.monotonic()
    result = _grade_cpu_step(function)
    assert time.monotonic() - started < 30
    assert result in (os.getpid(), ("computed", os.getpid()))
