"""Priority/FIFO admission uses eligible free CPUs, including overlapping rosters."""

import fcntl
import itertools
import json
import multiprocessing
import os
import threading
import time
from contextlib import ExitStack

import pytest

from merlin.common import cpu_pool as C

FORK = multiprocessing.get_context("fork")


def _cpus(count=2):
    cpus = tuple(sorted(os.sched_getaffinity(0))[:count])
    if len(cpus) < count:
        pytest.skip(f"requires {count} allowed CPUs")
    return cpus


def _until(predicate):
    deadline = time.monotonic() + 10
    while not predicate():
        assert time.monotonic() < deadline, "waiter did not reach its expected state"
        time.sleep(.005)


def _queued(root):
    return list((root / "waiters").glob("*.wait"))


def _run(root, cpus, level, label, entered, release, results):
    with C.cpu_priority(level), C.selected_pool(cpus, slot_root=root, timeout_s=15):
        results.put((label, tuple(os.sched_getaffinity(0))))
        entered.set()
        assert release.wait(15)


@pytest.mark.parametrize("processes", [False, True], ids=["threads", "processes"])
def test_busy_disjoint_head_does_not_block_a_free_cpu(tmp_path, processes):
    busy, free = _cpus()
    root = tmp_path / "slots"
    results = FORK.Queue()
    entries = []

    def start(cpus, level, label):
        entered, release = FORK.Event(), FORK.Event()
        kind = FORK.Process if processes else threading.Thread
        worker = kind(target=_run, args=(root, cpus, level, label, entered, release, results))
        worker.start()
        entries.append((worker, release))
        return entered

    try:
        assert start((busy,), C.GRADE, "holder").wait(10)
        head = start((busy,), C.AUTHOR, "blocked-author")
        _until(lambda: len(_queued(root)) == 1)
        ticket = json.loads(_queued(root)[0].read_text())
        assert ticket == {"version": 1, "cpus": [busy]}
        assert start((free,), C.AUXILIARY, "free-probe").wait(5)
        assert not head.is_set()  # no preemption of the real busy slot
        entries[0][1].set()
        assert head.wait(5)
    finally:
        for _, release in entries:
            release.set()
        for worker, _ in entries:
            worker.join(20)
            assert not worker.is_alive()
            if processes:
                assert worker.exitcode == 0
    observed = dict(results.get(timeout=5) for _ in entries)
    assert observed == {"holder": (busy,), "blocked-author": (busy,), "free-probe": (free,)}
    assert not _queued(root)


def test_overlapping_fifo_keeps_the_only_eligible_free_cpu_for_the_earlier_waiter(tmp_path):
    first, second = _cpus()
    root = tmp_path / "slots"
    root.mkdir()
    results = FORK.Queue()
    workers = []
    with ExitStack() as stack:
        holders = [stack.enter_context((root / f"cpu-{cpu}.lock").open("w")) for cpu in (first, second)]
        for holder in holders:
            fcntl.flock(holder, fcntl.LOCK_EX)
        try:
            arrivals = (((first, second), "flexible-first"), ((first,), "singleton-later"))
            for index, (roster, label) in enumerate(arrivals):
                entered, release = FORK.Event(), FORK.Event()
                process = FORK.Process(target=_run, args=(root, roster, C.GRADE, label, entered, release, results))
                process.start()
                workers.append((process, entered, release))
                _until(lambda: len(_queued(root)) == index + 1)
            fcntl.flock(holders[0], fcntl.LOCK_UN)
            assert workers[0][1].wait(5)
            assert not workers[1][1].is_set()
            # The later singleton still needs the slot actually held by the
            # first waiter, even when the other CPU becomes free.
            fcntl.flock(holders[1], fcntl.LOCK_UN)
            assert not workers[1][1].wait(.1)
            workers[0][2].set()
            assert workers[1][1].wait(5)
        finally:
            for holder in holders:
                fcntl.flock(holder, fcntl.LOCK_UN)
            for _, _, release in workers:
                release.set()
            for process, _, _ in workers:
                process.join(20)
                assert process.exitcode == 0
    assert dict(results.get(timeout=5) for _ in workers) == {"flexible-first": (first,), "singleton-later": (first,)}


def test_augmenting_path_preserves_flexible_ahead_waiter_and_admits_a_singleton(tmp_path):
    first, second = _cpus()
    root = tmp_path / "slots"
    root.mkdir()
    # A genuine locked ticket holds its priority while its owner has not yet
    # acquired a CPU. A greedy reservation of `first` would wrongly block us.
    earlier = C._Waiter(root, C.AUTHOR, time.monotonic_ns(), (first, second))
    try:
        with C.cpu_priority(C.GRADE), C.selected_pool((first,), slot_root=root, timeout_s=1):
            assert os.sched_getaffinity(0) == {first}
            with (root / f"cpu-{second}.lock").open("a+") as probe:
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        earlier.release()


def _gated_singleton(root, cpu, earlier_entered, entered, release, results):
    acquire = C._try_slots

    def after_earlier(*args, **kwargs):
        assert earlier_entered.wait(5)
        return acquire(*args, **kwargs)

    C._try_slots = after_earlier
    _run(root, (cpu,), C.GRADE, "singleton", entered, release, results)


def test_flexible_head_uses_full_queue_matching_before_the_singleton_acquires(tmp_path):
    first, second = _cpus()
    root = tmp_path / "slots"
    root.mkdir()
    results = FORK.Queue()
    head_entered, tail_entered, release = FORK.Event(), FORK.Event(), FORK.Event()
    head = FORK.Process(target=_run, args=(root, (first, second), C.AUTHOR,
                                         "head", head_entered, release, results))
    tail = FORK.Process(target=_gated_singleton, args=(root, first, head_entered,
                                                    tail_entered, release, results))
    with ExitStack() as stack:
        holders = [stack.enter_context((root / f"cpu-{cpu}.lock").open("w")) for cpu in (first, second)]
        for holder in holders:
            fcntl.flock(holder, fcntl.LOCK_EX)
        try:
            head.start()
            _until(lambda: len(_queued(root)) == 1)
            tail.start()
            _until(lambda: len(_queued(root)) == 2)
            for holder in holders:
                fcntl.flock(holder, fcntl.LOCK_UN)
            assert head_entered.wait(5)
            assert tail_entered.wait(5)  # both own real exclusive slots together
            assert dict(results.get(timeout=5) for _ in range(2)) == {"head": (second,), "singleton": (first,)}
        finally:
            release.set()
            for holder in holders:
                fcntl.flock(holder, fcntl.LOCK_UN)
            for process in (head, tail):
                if process.pid is not None:
                    process.join(20)
                    assert process.exitcode == 0


def test_legacy_roster_does_not_assume_access_to_other_waiters_cpus(tmp_path):
    first, second, third = _cpus(3)
    root = tmp_path / "slots"
    root.mkdir()
    unknown = C._Waiter(root, C.AUTHOR, time.monotonic_ns())
    flexible = C._Waiter(root, C.AUTHOR, time.monotonic_ns(), (second, third))
    try:
        with C.cpu_priority(C.GRADE), pytest.raises(TimeoutError):
            with C.selected_pool((first,), slot_root=root, timeout_s=.05):
                pytest.fail("an unknown earlier waiter was assumed to use another roster")
    finally:
        flexible.release()
        unknown.release()


def test_post_acquisition_refresh_moves_a_flexible_head_off_a_cached_out_singleton(tmp_path, monkeypatch):
    first, second = _cpus()
    root = tmp_path / "slots"
    root.mkdir()
    tail = C._Waiter(root, C.AUXILIARY, time.monotonic_ns(), (first,))
    assignment = C._queue_assignment
    cached_once = False
    attempted = []
    acquire = C._try_slots

    def stale(directory, key, cpus, *, fresh=False):
        nonlocal cached_once
        if not fresh and not cached_once:
            cached_once = True
            return [cpus], 0  # an actual singleton ticket is missing from the cache
        return assignment(directory, key, cpus, fresh=fresh)

    def observed(*args, **kwargs):
        fd, cpu = acquire(*args, **kwargs)
        if fd is not None:
            attempted.append(cpu)
        return fd, cpu

    monkeypatch.setattr(C, "_queue_assignment", stale)
    monkeypatch.setattr(C, "_try_slots", observed)
    try:
        with C.cpu_priority(C.AUTHOR), C.selected_pool((first, second), slot_root=root, timeout_s=1):
            assert os.sched_getaffinity(0) == {second}
            with (root / f"cpu-{first}.lock").open("a+") as probe:
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert attempted[:2] == [first, second]
    finally:
        tail.release()


def test_matching_matches_exhaustive_priority_optimum_on_overlapping_rosters():
    cpus = (0, 1, 2)
    subsets = [tuple(cpu for cpu in cpus if mask & (1 << cpu)) for mask in range(8)]
    rosters = [None, *subsets[1:]]
    for selected in itertools.product(rosters, repeat=3):
        for free in subsets:
            actual = C._matching(selected, free)
            choices = [(*[cpu for cpu in free if roster is None or cpu in roster], None) for roster in selected]
            feasible = [assignment for assignment in itertools.product(*choices)
                        if len([cpu for cpu in assignment if cpu is not None])
                        == len({cpu for cpu in assignment if cpu is not None})]
            optimum = max(tuple(cpu is not None for cpu in assignment) for assignment in feasible)
            assert tuple(index in actual for index in range(3)) == optimum
            assert len(set(actual.values())) == len(actual)
            assert all(cpu in free and (selected[index] is None or cpu in selected[index])
                       for index, cpu in actual.items())


@pytest.mark.parametrize("fail", [False, True])
def test_ticket_payload_partial_writes_and_errors_keep_the_original_cleanup(tmp_path, monkeypatch, fail):
    root = tmp_path / "slots"
    root.mkdir()
    before = set(C._WAITER_FDS)
    write = os.write
    calls = 0

    def partial(fd, payload):
        nonlocal calls
        calls += 1
        if fail and calls == 2:
            raise OSError("ticket write refused")
        return write(fd, payload[:3])

    monkeypatch.setattr(os, "write", partial)
    if fail:
        with pytest.raises(OSError, match="ticket write refused"):
            C._Waiter(root, C.AUTHOR, time.monotonic_ns(), (1, 2))
    else:
        waiter = C._Waiter(root, C.AUTHOR, time.monotonic_ns(), (1, 2))
        try:
            assert json.loads(waiter.path.read_text()) == {"version": 1, "cpus": [1, 2]}
            assert C._waiter_key(waiter.path.name) == waiter.key
        finally:
            waiter.release()
    assert calls > 1
    assert set(C._WAITER_FDS) == before
    assert not list((root / "waiters").iterdir())


@pytest.mark.parametrize("payload", [
    b"", b"{", b'{"version":1,"cpus":[true]}', b'{"version":1,"cpus":[1,1]}', b"x" * 65537,
])
def test_unknown_and_legacy_payloads_remain_conservative(tmp_path, payload):
    cpu, = _cpus(1)
    root = tmp_path / "slots"
    root.mkdir()
    earlier = C._Waiter(root, C.AUTHOR, time.monotonic_ns())
    try:
        os.pwrite(earlier.fd, payload, 0)
        with C.cpu_priority(C.GRADE), pytest.raises(TimeoutError):
            with C.selected_pool((cpu,), slot_root=root, timeout_s=.05):
                pytest.fail("an unknown earlier roster was ignored")
    finally:
        earlier.release()


@pytest.mark.parametrize("race", ["roster-change", "cpu-acquired", "new-urgent-ticket"])
def test_post_acquisition_reproof_releases_a_slot_that_would_displace_an_earlier_waiter(tmp_path, monkeypatch, race):
    first, second = _cpus()
    root = tmp_path / "slots"
    root.mkdir()
    earlier = None if race == "new-urgent-ticket" else C._Waiter(root, C.AUTHOR, time.monotonic_ns(), (second,))
    extra_holder = None
    acquire = C._try_slots
    raced = False

    def racing_acquisition(*args, **kwargs):
        nonlocal raced, earlier, extra_holder
        fd, cpu = acquire(*args, **kwargs)
        if fd is not None and not raced:
            raced = True
            if race == "roster-change":
                payload = json.dumps({"version": 1, "cpus": [first]}).encode()
                os.ftruncate(earlier.fd, 0)
                os.pwrite(earlier.fd, payload, 0)
            elif race == "cpu-acquired":
                # Fresh free-slot observations must see another real holder.
                payload = json.dumps({"version": 1, "cpus": [first, second]}).encode()
                os.ftruncate(earlier.fd, 0)
                os.pwrite(earlier.fd, payload, 0)
                extra_holder = (root / f"cpu-{second}.lock").open("a+")
                fcntl.flock(extra_holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
            else:
                earlier = C._Waiter(root, C.AUTHOR, time.monotonic_ns(), (first,))
        return fd, cpu

    monkeypatch.setattr(C, "_try_slots", racing_acquisition)
    try:
        with C.cpu_priority(C.GRADE), pytest.raises(TimeoutError):
            with C.selected_pool((first,), slot_root=root, timeout_s=.1):
                pytest.fail("raced acquisition displaced an earlier feasible waiter")
        assert raced
        with (root / f"cpu-{first}.lock").open("a+") as probe:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        if earlier is not None:
            earlier.release()
        if extra_holder is not None:
            extra_holder.close()
