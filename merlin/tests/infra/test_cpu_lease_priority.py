"""One shared CPU pool divided only by priority: cross-process order, FIFO, liveness, no reservation.

A lease class orders WAITERS for a free slot. It must never reserve a CPU for an absent class, never
let a dead or stopped waiter block the pool, and never change same-thread nested reuse.
"""

from __future__ import annotations

import fcntl
import json
import multiprocessing
import os
import subprocess
import sys
import threading
import time

import pytest

from merlin.common import cpu_pool as C
from merlin.common.paths import python_import_roots

FORK = multiprocessing.get_context("fork")


@pytest.fixture
def pool(monkeypatch, tmp_path):
    """A one-CPU shared namespace in a private slot directory; no inherited class or preparation roster."""
    cpu = min(os.sched_getaffinity(0))
    root = tmp_path / "slots"
    monkeypatch.setenv("MERLIN_COMPUTE_CPUS", str(cpu))
    monkeypatch.setenv("MERLIN_COMPUTE_CPU_SLOTS", str(root))
    for key in (C.PRIORITY_ENV, "MERLIN_PREPARATION_CPUS", "MERLIN_GRADE_CPUS"):
        monkeypatch.delenv(key, raising=False)
    return cpu, root


def _queued(root):
    directory = root / "waiters"
    return sorted(name for name in os.listdir(directory) if name.endswith(".wait")) if directory.is_dir() else []


def _until(predicate, timeout=10.0):
    end = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < end, "condition was not reached"
        time.sleep(0.01)


def _hold(held, release):
    with C.shared_compute_pool(timeout_s=10):
        held.set()
        release.wait(20)


def _contend(level, label, results, hold_s):
    with C.cpu_priority(level), C.shared_compute_pool(timeout_s=30):
        results.put((label, time.monotonic()))
        time.sleep(hold_s)


def _start_holder():
    held, release = FORK.Event(), FORK.Event()
    holder = FORK.Process(target=_hold, args=(held, release))
    holder.start()
    assert held.wait(10)
    return holder, release


def _join(processes):
    for process in processes:
        process.join(30)
        assert process.exitcode == 0


def test_more_urgent_classes_take_the_slot_first_and_each_class_is_fifo(pool):
    _, root = pool
    holder, release = _start_holder()
    results = FORK.Queue()
    arrivals = [(C.AUXILIARY, "probe"), (C.GRADE, "grade"), (C.AUTHOR, "author-1"),
                (C.PREPARATION, "prepare"), (C.AUTHOR, "author-2"), (C.GRADE, "grade-2")]
    contenders = []
    try:
        for count, (level, label) in enumerate(arrivals, start=1):
            process = FORK.Process(target=_contend, args=(level, label, results, 0.05))
            process.start()
            contenders.append(process)
            _until(lambda count=count: len(_queued(root)) == count)  # each one queued before the next arrives
    finally:
        release.set()
    _join([holder, *contenders])
    order = [results.get(timeout=5) for _ in arrivals]
    assert [label for label, _ in sorted(order, key=lambda row: row[1])] == [
        "author-1", "author-2", "grade", "grade-2", "prepare", "probe"]
    assert _queued(root) == []


def test_lower_class_never_takes_a_free_slot_while_a_more_urgent_waiter_is_queued(pool):
    _, root = pool
    directory = root / "waiters"
    directory.mkdir(parents=True)
    queued_ns = time.monotonic_ns()
    entry = directory / f"{C.AUTHOR}-{queued_ns:020d}-1-1-live{C._WAITER_SUFFIX}"
    with open(entry, "w") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX)  # a live author waiter, still polling
        for level in (C.AUXILIARY, C.GRADE, C.AUTHOR):  # lower classes and a later author all defer
            with C.cpu_priority(level), pytest.raises(TimeoutError):
                with C.shared_compute_pool(timeout_s=0.2):
                    pytest.fail("a free slot was taken ahead of an earlier queued author request")
        with C.cpu_priority(C.AUTHOR), C.shared_compute_pool(timeout_s=0, queued_since_ns=queued_ns - 1):
            pass  # an author queued earlier than the entry is not blocked by it
        # A live owner that stopped polling (e.g. a stopped process group) stops outranking others.
        stale = (time.time_ns() - 2 * C._WAITER_STALL_NS) / 1e9
        os.utime(entry, (stale, stale))
        with C.cpu_priority(C.AUXILIARY), C.shared_compute_pool(timeout_s=2):
            pass
        assert entry.exists()  # still owned: it is skipped, never removed


def test_a_stale_free_slot_view_never_admits_a_lower_class_past_a_queued_one(pool, monkeypatch):
    _, root = pool
    directory = root / "waiters"
    directory.mkdir(parents=True)
    entry = directory / f"{C.AUTHOR}-{time.monotonic_ns():020d}-1-1-live{C._WAITER_SUFFIX}"
    counted = C._free_slots
    # A cached view that still shows a slot freed a moment ago, beside the one that is really free.
    monkeypatch.setattr(C, "_free_slots", lambda root, cpus, fresh=False: counted(root, cpus, fresh=True) + (
        0 if fresh else 1))
    with open(entry, "w") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX)
        with C.cpu_priority(C.AUXILIARY), pytest.raises(TimeoutError):
            with C.shared_compute_pool(timeout_s=0.2):
                pytest.fail("a stale view let a probe keep the slot a queued author request needs")
    with C.shared_compute_pool(timeout_s=1):
        pass  # every slot the probe touched was handed back (the author entry is gone with its owner)


def _contend_and_hold(level, label, results, release):
    with C.cpu_priority(level), C.shared_compute_pool(timeout_s=30):
        results.put((label, time.monotonic()))
        release.wait(30)


def test_enough_free_slots_admit_queued_waiters_together(monkeypatch, tmp_path):
    """Two slots free at once serve the author and the probe behind it in parallel, not one by one."""
    cpus = sorted(os.sched_getaffinity(0))[:2]
    if len(cpus) < 2:
        pytest.skip("two allowed CPUs are needed for a two-slot pool")
    root = tmp_path / "slots"
    monkeypatch.setenv("MERLIN_COMPUTE_CPUS", ",".join(map(str, cpus)))
    monkeypatch.setenv("MERLIN_COMPUTE_CPU_SLOTS", str(root))
    monkeypatch.delenv(C.PRIORITY_ENV, raising=False)
    holders = [_start_holder() for _ in cpus]
    results, release = FORK.Queue(), FORK.Event()
    contenders = []
    try:
        for count, (level, label) in enumerate(((C.AUTHOR, "author"), (C.AUXILIARY, "probe")), start=1):
            process = FORK.Process(target=_contend_and_hold, args=(level, label, results, release))
            process.start()
            contenders.append(process)
            _until(lambda count=count: len(_queued(root)) == count)
        for _, released in holders:
            released.set()
        acquired = dict(results.get(timeout=10) for _ in contenders)  # both hold a slot at the same time
    finally:
        release.set()
        for _, released in holders:
            released.set()
    _join([holder for holder, _ in holders] + contenders)
    assert set(acquired) == {"author", "probe"}


def test_a_free_slot_is_never_reserved_for_an_absent_class(pool):
    _, root = pool
    for level in (C.AUXILIARY, C.PREPARATION, C.GRADE, C.AUTHOR):
        with C.cpu_priority(level), C.shared_compute_pool(timeout_s=0):
            pass
    assert _queued(root) == []  # uncontended leases never touch the queue


def _register_and_die(root, level):
    C._Waiter(root, level, time.monotonic_ns())
    os._exit(0)  # no release: the kernel drops the lock, the queue file stays behind


def test_dead_waiters_are_removed_and_never_block(pool):
    _, root = pool
    root.mkdir(parents=True)
    child = FORK.Process(target=_register_and_die, args=(root, C.AUTHOR))
    child.start()
    child.join(10)
    assert [name[0] for name in _queued(root)] == [str(C.AUTHOR)]
    # The dead author entry is ahead of this grade lease; it is removed on sight, not waited for.
    with C.shared_compute_pool(timeout_s=2):
        pass
    assert _queued(root) == []
    # An unlocked entry behind every waiter (a killed probe) is swept by the next queued registration.
    (root / "waiters" / f"{C.AUXILIARY}-{time.monotonic_ns():020d}-1-1-dead{C._WAITER_SUFFIX}").touch()
    holder, release = _start_holder()
    try:
        results = FORK.Queue()
        waiter = FORK.Process(target=_contend, args=(C.AUTHOR, "author", results, 0))
        waiter.start()
        _until(lambda: [name[0] for name in _queued(root)] == [str(C.AUTHOR)])
    finally:
        release.set()
    _join([holder, waiter])
    assert _queued(root) == []


def test_a_retry_keeps_its_place_in_its_class(pool):
    _, root = pool
    holder, release = _start_holder()
    results = FORK.Queue()
    retried_since = time.monotonic_ns()

    def retry():
        with C.shared_compute_pool(timeout_s=30, queued_since_ns=retried_since):
            results.put(("retry", time.monotonic()))
            time.sleep(0.05)

    try:
        newcomer = FORK.Process(target=_contend, args=(C.GRADE, "newcomer", results, 0.05))
        newcomer.start()
        _until(lambda: len(_queued(root)) == 1)
        retrying = FORK.Process(target=retry)
        retrying.start()
        _until(lambda: len(_queued(root)) == 2)
    finally:
        release.set()
    _join([holder, newcomer, retrying])
    order = sorted((results.get(timeout=5) for _ in range(2)), key=lambda row: row[1])
    assert [label for label, _ in order] == ["retry", "newcomer"]


def test_nested_same_thread_lease_is_reused_even_behind_a_queued_author(pool):
    cpu, root = pool
    original = os.sched_getaffinity(0)
    directory = root / "waiters"
    with C.cpu_priority(C.AUXILIARY), C.shared_compute_pool(timeout_s=0):
        directory.mkdir(parents=True, exist_ok=True)
        entry = directory / f"{C.AUTHOR}-{time.monotonic_ns():020d}-1-1-live{C._WAITER_SUFFIX}"
        with open(entry, "w") as owner:
            fcntl.flock(owner, fcntl.LOCK_EX)
            with C.cpu_priority(C.AUXILIARY), C.shared_compute_pool(timeout_s=0):
                assert os.sched_getaffinity(0) == {cpu}
                assert C.current_thread_owns_cpu_lease()
    assert os.sched_getaffinity(0) == original


def test_lease_class_resolution_and_validation(monkeypatch):
    monkeypatch.delenv(C.PRIORITY_ENV, raising=False)
    assert C.lease_priority() == C.GRADE
    monkeypatch.setenv(C.PRIORITY_ENV, "3")
    assert C.lease_priority() == C.PREPARATION
    with C.cpu_priority(C.AUTHOR):
        assert C.lease_priority() == C.AUTHOR
        observed = []
        thread = threading.Thread(target=lambda: observed.append(C.lease_priority()))
        thread.start()
        thread.join()
        assert observed == [C.PREPARATION]  # a new thread falls back to the process class
    with C.cpu_priority(C.AUXILIARY, export=True):
        assert os.environ[C.PRIORITY_ENV] == "4"
    assert os.environ[C.PRIORITY_ENV] == "3"

    @C.with_priority(C.AUTHOR)
    def decorated():
        return C.lease_priority()

    assert decorated() == C.AUTHOR and C.lease_priority() == C.PREPARATION
    for bad in ("0", "5", "first", "1.0"):
        monkeypatch.setenv(C.PRIORITY_ENV, bad)
        with pytest.raises(ValueError, match=C.PRIORITY_ENV):
            C.lease_priority()
    for bad in (0, 5, True, "x"):
        with pytest.raises(ValueError):
            with C.cpu_priority(bad):
                pass


def test_compute_capacity_is_the_selected_roster(tmp_path):
    assert C.compute_capacity({}) == 0
    assert C.compute_capacity({"MERLIN_COMPUTE_CPUS": "4,5,9", "MERLIN_COMPUTE_CPU_SLOTS": str(tmp_path)}) == 3
    with pytest.raises(ValueError):
        C.compute_capacity({"MERLIN_COMPUTE_CPUS": "4"})


def _child_environment():
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(map(str, python_import_roots()))
    return environment


_LEASING_CHILD = (
    "import json, os; from merlin.common import cpu_pool as C\n"
    "with C.shared_compute_pool(timeout_s=30):\n"
    "    print(json.dumps([C.lease_priority(), sorted(os.sched_getaffinity(0))]))\n"
)


def test_worker_launch_carries_the_callers_class_only_with_a_shared_pool(pool):
    cpu, _ = pool
    environment = _child_environment()
    command = [sys.executable, "-c", _LEASING_CHILD]
    assert C.worker_pool_command(command, {}) is command  # legacy launch unchanged
    unclassified = C.worker_pool_command(command, environment)
    assert unclassified[:3] == ["taskset", "--cpu-list", str(cpu)] and "env" not in unclassified
    with C.cpu_priority(C.AUXILIARY):
        launched = C.worker_pool_command(command, environment)
        assert launched[3:5] == ["env", f"{C.PRIORITY_ENV}=4"]
        inherited = dict(environment, **{C.PRIORITY_ENV: "4"})
        assert "env" not in C.worker_pool_command(command, inherited)  # already inherited
    level, affinity = json.loads(subprocess.check_output(launched, env=environment, text=True, timeout=60))
    assert level == C.AUXILIARY and affinity == [cpu]


def test_preparation_folds_into_the_pool_without_deadlock(pool):
    """Unleased preparation waits on leasing workers; neither the parent nor its bounded steps strand a slot.

    One CPU, preparation on the pool roster (no separate roster), a GRADE competitor in another thread,
    and three preparation workers that each lease the only slot while their parent waits for them.
    """
    cpu, _ = pool
    environment = _child_environment()
    assert C.preparation_cpus() == (cpu,)
    stop = threading.Event()
    graded = []

    def grade():
        # Bounded: an unbounded stream of GRADE leases would (correctly) starve PREPARATION.
        with C.cpu_priority(C.GRADE):
            for _ in range(30):
                if stop.is_set():
                    break
                with C.shared_compute_pool(timeout_s=30):
                    graded.append(C.lease_priority())
                    time.sleep(0.005)

    competitor = threading.Thread(target=grade)
    competitor.start()
    try:
        with C.cpu_priority(C.PREPARATION):
            with C.shared_compute_pool(timeout_s=30):
                pass  # a bounded preparation step leases, then releases before any child wait
            with C.selected_pool(C.preparation_cpus()):
                workers = [
                    subprocess.Popen(C.worker_pool_command([sys.executable, "-c", _LEASING_CHILD], environment),
                                     env=environment, stdout=subprocess.PIPE, text=True)
                    for _ in range(3)
                ]
                assert not C.current_thread_owns_cpu_lease()  # the waiting parent holds no slot
                outputs = [worker.communicate(timeout=60)[0] for worker in workers]
    finally:
        stop.set()
        competitor.join(30)
    assert all(worker.returncode == 0 for worker in workers)
    assert [json.loads(output) for output in outputs] == [[C.PREPARATION, [cpu]]] * 3
    assert graded and set(graded) == {C.GRADE}
