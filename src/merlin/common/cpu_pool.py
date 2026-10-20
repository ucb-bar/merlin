"""Explicit Linux CPU pools for trusted subprocess launch threads.

A shared compute namespace (``MERLIN_COMPUTE_CPUS`` + ``MERLIN_COMPUTE_CPU_SLOTS``)
is the single limiter for every engine: one flock'd ``cpu-<n>.lock`` slot per
CPU, leased by builds, simulators, grading and preparation steps alike. It is
divided only by priority (see :func:`cpu_priority`); no CPU is reserved for a
class or an engine.
"""

from __future__ import annotations

import bisect
import fcntl
import functools
import heapq
import json
import math
import os
import secrets
import select
import stat
import threading
import time
import weakref
from collections.abc import Mapping
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path

from merlin.common.invocation_record import timed_event

_ACTIVE_LEASE = ContextVar("selected_cpu_lease", default=None)
_ISSUED_LEASES = weakref.WeakKeyDictionary()
# Linux UAPI values; some supported CPython builds omit the fcntl names.
_SEALS = 0x000F
_ADD_SEALS = getattr(fcntl, "F_ADD_SEALS", 1033)
_GET_SEALS = getattr(fcntl, "F_GET_SEALS", 1034)

# Lease priority classes; a lower number is served first. A class only orders
# waiters for a free slot: it never reserves a CPU, preempts a held slot or
# changes what a step computes, so a lower class runs whenever no higher-class
# (or earlier same-class) waiter is queued.
AUTHOR = 1  # work the author requested and waits on: self-checks, simulation jobs, model checks
GRADE = 2  # in-turn, background and round grade capsules; the default for an unclassified lease
PREPARATION = 3  # run and input preparation that no requester is blocked on yet
AUXILIARY = 4  # advisory probes and telemetry
PRIORITY_ENV = "MERLIN_CPU_PRIORITY"
_PRIORITIES = (AUTHOR, GRADE, PREPARATION, AUXILIARY)
_PRIORITY = ContextVar("cpu_lease_priority", default=None)
# Cross-process waiter queue inside each slot namespace. An entry is a uniquely
# named file whose owner holds an exclusive flock on it while waiting, so a dead
# owner's entry is recognisable (its lock is free) and removable by anyone.
_WAITER_DIR = "waiters"
_WAITER_SUFFIX = ".wait"
_WAITER_HEARTBEAT_NS = 250_000_000
# A live owner that stopped polling (e.g. a stopped process group) stops
# outranking others after this long; it still takes its turn when it resumes.
_WAITER_STALL_NS = 5_000_000_000
_ORPHAN_TEMPORARY_NS = 60_000_000_000
_SWEEP_INTERVAL_NS = 1_000_000_000
# One process-wide view of the queue and of the free slots serves every waiting
# thread; a foreign entry's lock is re-proven at most this often. A waiter
# admitted past others on a cached view re-proves its admission afresh.
_QUEUE_VIEW_TTL_NS = 2_000_000
_FREE_VIEW_TTL_NS = 5_000_000
_FOREIGN_LIVENESS_TTL_NS = 50_000_000
# Polls catch slots freed by other processes (this process wakes its own front
# waiters directly); a waiter deeper in the queue needs several releases first.
_FRONT_POLL_S = 0.005
_NEAR_POLL_S = 0.01
_DEEP_POLL_S = 0.05
_WAITER_FDS: set[int] = set()
_WAITER_FD_LOCK = threading.Lock()
# This process's queued waiters (key -> wake event); their liveness needs no lock proof.
_LOCAL_WAITERS: dict[tuple, threading.Event] = {}
_QUEUE_VIEWS: dict[Path, tuple[int, list]] = {}
_FREE_VIEWS: dict[tuple[Path, tuple[int, ...]], tuple[int, int]] = {}
_FOREIGN_LIVENESS: dict[Path, tuple[int, bool]] = {}
_LAST_SWEEP: dict[Path, int] = {}


def _waiters_after_fork_child() -> None:
    # A forked child must not keep its parent's queue entries alive after the
    # parent exits; closing the copies leaves the parent's locks in place.
    for fd in _WAITER_FDS:
        try:
            os.close(fd)
        except OSError:
            pass
    _WAITER_FDS.clear()
    _LOCAL_WAITERS.clear()
    _QUEUE_VIEWS.clear()
    _FOREIGN_LIVENESS.clear()
    _LAST_SWEEP.clear()
    _FREE_VIEWS.clear()
    _WAITER_FD_LOCK.release()


os.register_at_fork(
    before=_WAITER_FD_LOCK.acquire,
    after_in_parent=_WAITER_FD_LOCK.release,
    after_in_child=_waiters_after_fork_child,
)


def _parse_priority(raw: object) -> int:
    if type(raw) is int and raw in _PRIORITIES:
        return raw
    if isinstance(raw, str) and raw.strip().isdecimal() and int(raw.strip()) in _PRIORITIES:
        return int(raw.strip())
    raise ValueError(f"{PRIORITY_ENV} must be one of {', '.join(map(str, _PRIORITIES))}")


def _explicit_priority(environ: Mapping[str, str] | None = None) -> int | None:
    level = _PRIORITY.get()
    if level is not None:
        return level
    raw = (os.environ if environ is None else environ).get(PRIORITY_ENV, "").strip()
    return _parse_priority(raw) if raw else None


def lease_priority(environ: Mapping[str, str] | None = None) -> int:
    """The class this thread's next shared lease waits at.

    The context's :func:`cpu_priority` wins, then the inherited process class
    (``MERLIN_CPU_PRIORITY``), then :data:`GRADE`. New threads start with an
    empty context, so executors that must keep a class submit through
    ``contextvars.copy_context().run``; otherwise they fall back to the process
    class.
    """
    level = _explicit_priority(environ)
    return GRADE if level is None else level


@contextmanager
def cpu_priority(level: int, *, export: bool = False):
    """Lease shared compute at ``level`` inside this context.

    ``export`` also publishes the class to this process environment so every
    child inherits it; the environment is process-wide, so export only at a
    single-purpose process entrypoint. :func:`worker_pool_command` carries a
    context-only class into the workers it launches.
    """
    level = _parse_priority(level)
    token = _PRIORITY.set(level)
    previous = os.environ.get(PRIORITY_ENV)
    if export:
        os.environ[PRIORITY_ENV] = str(level)
    try:
        yield
    finally:
        if export:
            if previous is None:
                os.environ.pop(PRIORITY_ENV, None)
            else:
                os.environ[PRIORITY_ENV] = previous
        _PRIORITY.reset(token)


def with_priority(level: int):
    """Decorator form of :func:`cpu_priority` for an entrypoint function."""
    level = _parse_priority(level)

    def decorate(function):
        @functools.wraps(function)
        def leased_at_priority(*args, **kwargs):
            with cpu_priority(level):
                return function(*args, **kwargs)

        return leased_at_priority

    return decorate


def _waiter_key(name: str):
    """``(priority, queued_ns, pid, tid, nonce)`` for a queue entry name, else ``None``."""
    if not name.endswith(_WAITER_SUFFIX):
        return None
    fields = name[: -len(_WAITER_SUFFIX)].split("-")
    if len(fields) != 5 or not all(field.isdecimal() for field in fields[:4]) or not fields[4].isalnum():
        return None
    level, queued, pid, tid = (int(field) for field in fields[:4])
    if level not in _PRIORITIES:
        return None
    return level, queued, pid, tid, fields[4]


def _close_waiter_fd(fd: int) -> None:
    with _WAITER_FD_LOCK:
        _WAITER_FDS.discard(fd)
        os.close(fd)


def _remove_unlocked(path: Path, fd: int) -> None:
    """Unlink ``path`` only while it still names the inode whose lock we observed free."""
    try:
        current = path.stat(follow_symlinks=False)
        opened = os.fstat(fd)
        if (current.st_dev, current.st_ino) == (opened.st_dev, opened.st_ino):
            path.unlink()
    except FileNotFoundError:
        pass


def _held_since(path: Path) -> int | None:
    """Last heartbeat of a locked queue file; ``None`` once its owner is gone (it is then removed)."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError:
        return None  # Already gone, or not an entry a same-user waiter could own.
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return os.fstat(fd).st_mtime_ns
        _remove_unlocked(path, fd)
        return None
    finally:
        os.close(fd)


def _live_waiter(path: Path, now_ns: int) -> bool:
    """A queued owner still polling; a dead owner's entry is removed on sight."""
    beat = _held_since(path)
    return beat is not None and now_ns - beat < _WAITER_STALL_NS


def _wake_local_waiters(count: int = 2) -> None:
    """Hand a slot freed in this process to its front waiters now; others poll.

    Only the front of the local queue can be admitted by one release, so waking
    every waiter would just make the rest rescan the queue.
    """
    for key in heapq.nsmallest(count, tuple(_LOCAL_WAITERS)):
        event = _LOCAL_WAITERS.get(key)
        if event is not None:
            event.set()


def _queued_keys(directory: Path) -> list:
    """Sorted keys of the live waiters in one namespace queue, shared by this process's threads."""
    now = time.monotonic_ns()
    cached = _QUEUE_VIEWS.get(directory)
    if cached is not None and now - cached[0] < _QUEUE_VIEW_TTL_NS:
        return cached[1]
    try:
        names = os.listdir(directory)
    except FileNotFoundError:
        names = []
    pid, wall = os.getpid(), time.time_ns()
    keys = []
    for name in names:
        key = _waiter_key(name)
        if key is None:
            continue
        if key in _LOCAL_WAITERS:
            keys.append(key)
            continue
        if key[2] == pid:
            # Not one of ours: an earlier process that reused this pid, or a finished entry.
            _FOREIGN_LIVENESS.pop(directory / name, None)
        path = directory / name
        proven = _FOREIGN_LIVENESS.get(path)
        if proven is None or now - proven[0] >= _FOREIGN_LIVENESS_TTL_NS:
            proven = (now, _live_waiter(path, wall))
            _FOREIGN_LIVENESS[path] = proven
        if proven[1]:
            keys.append(key)
    if len(_FOREIGN_LIVENESS) > 4 * len(names) + 64:
        listed = {directory / name for name in names}
        for path in tuple(_FOREIGN_LIVENESS):  # A snapshot: other threads update the cache concurrently.
            if path.parent == directory and path not in listed:
                _FOREIGN_LIVENESS.pop(path, None)
    keys.sort()
    _QUEUE_VIEWS[directory] = (now, keys)
    return keys


def _free_slots(root: Path, cpus: tuple[int, ...], *, fresh: bool = False) -> int:
    """Count unheld slots; a shared probe never blocks a holder. One view serves this process."""
    now = time.monotonic_ns()
    cached = _FREE_VIEWS.get((root, cpus))
    if not fresh and cached is not None and now - cached[0] < _FREE_VIEW_TTL_NS:
        return cached[1]
    free = 0
    for cpu in cpus:
        try:
            fd = os.open(root / f"cpu-{cpu}.lock", os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        except FileNotFoundError:
            free += 1  # Never leased yet.
        else:
            try:
                fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                free += 1
            finally:
                os.close(fd)
    _FREE_VIEWS[(root, cpus)] = (now, free)
    return free


def _admission_holds(root: Path, cpus: tuple[int, ...], directory: Path, key) -> bool:
    """Re-prove, after taking a slot, that the waiters ahead still fit in the slots left free."""
    _QUEUE_VIEWS.pop(directory, None)
    return _free_slots(root, cpus, fresh=True) >= bisect.bisect_left(_queued_keys(directory), key)


class _Waiter:
    """One queued acquisition: a locked, uniquely named entry in the namespace queue."""

    def __init__(self, root: Path, level: int, queued_ns: int):
        self.directory = root / _WAITER_DIR
        self.directory.mkdir(mode=0o700, exist_ok=True)
        if self.directory.is_symlink() or not self.directory.is_dir():
            raise RuntimeError("CPU waiter queue is not an ordinary directory")
        nonce = secrets.token_hex(8)
        pid, tid = os.getpid(), threading.get_native_id()
        self.key = (level, queued_ns, pid, tid, nonce)
        self.path = self.directory / f"{level}-{queued_ns:020d}-{pid}-{tid}-{nonce}{_WAITER_SUFFIX}"
        temporary = self.directory / f".{nonce}.tmp"
        with _WAITER_FD_LOCK:
            self.fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
            _WAITER_FDS.add(self.fd)
        try:
            # Lock before the entry becomes visible under its queue name.
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.wake = _LOCAL_WAITERS[self.key] = threading.Event()
            os.rename(temporary, self.path)
        except BaseException:
            _LOCAL_WAITERS.pop(self.key, None)
            temporary.unlink(missing_ok=True)
            _close_waiter_fd(self.fd)
            raise
        _QUEUE_VIEWS.pop(self.directory, None)
        self.beat_ns = time.time_ns()
        self._sweep()

    def _sweep(self) -> None:
        """Drop entries of dead owners that no current waiter would otherwise reach (rate-limited)."""
        now = time.monotonic_ns()
        if now - _LAST_SWEEP.get(self.directory, -_SWEEP_INTERVAL_NS) < _SWEEP_INTERVAL_NS:
            return
        _LAST_SWEEP[self.directory] = now
        now_ns = time.time_ns()
        try:
            names = os.listdir(self.directory)
        except FileNotFoundError:
            return
        for name in names:
            path = self.directory / name
            if (key := _waiter_key(name)) is not None:
                if key not in _LOCAL_WAITERS:
                    _held_since(path)
            elif name.startswith(".") and name.endswith(".tmp"):
                # A creator locks its file before naming it; only an abandoned one stays unnamed.
                try:
                    if now_ns - path.stat(follow_symlinks=False).st_mtime_ns > _ORPHAN_TEMPORARY_NS:
                        _held_since(path)
                except FileNotFoundError:
                    pass

    def beat(self) -> None:
        now_ns = time.time_ns()
        if now_ns - self.beat_ns >= _WAITER_HEARTBEAT_NS:
            os.utime(self.fd)
            self.beat_ns = now_ns

    def release(self) -> None:
        try:
            _remove_unlocked(self.path, self.fd)
        finally:
            _LOCAL_WAITERS.pop(self.key, None)
            _QUEUE_VIEWS.pop(self.directory, None)
            _close_waiter_fd(self.fd)
            _wake_local_waiters(1)  # The next local waiter moved up one place.


def _try_slots(root: Path, cpus: tuple[int, ...], start: int = 0) -> tuple[int | None, int | None]:
    for offset in range(len(cpus)):
        cpu = cpus[(start + offset) % len(cpus)]
        fd = os.open(root / f"cpu-{cpu}.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            continue
        except BaseException:
            os.close(fd)
            raise
        return fd, cpu
    return None, None


@dataclass(eq=False)
class _Lease:
    pid: int
    thread: threading.Thread
    root: Path
    cpu: int
    fd: int
    check: object = None
    active: bool = True


def _lease_fields(lease):
    return lease.pid, lease.thread, lease.root, lease.cpu, lease.fd, lease.check


def _slot_identity(root, cpu, fd):
    path = root / f"cpu-{cpu}.lock"
    before = os.fstat(fd)
    current = path.stat(follow_symlinks=False)
    if (
        not stat.S_ISREG(before.st_mode)
        or not stat.S_ISREG(current.st_mode)
        or (before.st_dev, before.st_ino) != (current.st_dev, current.st_ino)
    ):
        raise RuntimeError("CPU lease lost its exact ordinary slot file")
    return before.st_dev, before.st_ino


def _verify_lease(lease):
    if (
        type(lease) is not _Lease
        or not lease.active
        or lease.pid != os.getpid()
        or lease.thread is not threading.current_thread()
        or _ISSUED_LEASES.get(lease) != _lease_fields(lease)
    ):
        raise RuntimeError("CPU lease is not its unchanged live original thread owner")
    _slot_identity(lease.root, lease.cpu, lease.fd)
    if lease.check is not None:
        lease.check()


def check_compiler_cpu():
    """Reopen live CPU custody before the original broker publishes a reply."""
    lease = _ACTIVE_LEASE.get()
    if lease is not None:
        _verify_lease(lease)


def current_thread_owns_cpu_lease():
    """Observe a live original slot, never a copied thread/process context."""
    lease = _ACTIVE_LEASE.get()
    if lease is None:
        return False
    if type(lease) is not _Lease:
        raise RuntimeError("CPU lease is not its unchanged live original thread owner")
    if lease.pid != os.getpid() or lease.thread is not threading.current_thread():
        return False
    _verify_lease(lease)
    if os.sched_getaffinity(0) != {lease.cpu}:
        raise RuntimeError("original CPU lease differs from its current thread affinity")
    return True


def _birth(pid, tid=None):
    suffix = f"task/{tid}/stat" if tid is not None else "stat"
    return Path(f"/proc/{pid}/{suffix}").read_text().rsplit(")", 1)[1].split()[19]


@dataclass(frozen=True)
class CompilerCPUTransfer:
    """Private inherited FDs, never a compiler request/environment capability."""

    metadata_fd: int
    slot_fd: int
    lifetime_fd: int

    @property
    def pass_fds(self):
        return self.metadata_fd, self.slot_fd, self.lifetime_fd


@contextmanager
def transfer_compiler_cpu(*, owner_sha256):
    """Transfer only this original thread's live slot to its trusted RPC child.

    The caller must join/reap that child before this scope and its enclosing CPU
    lease close. The token carries CPU custody only, never compiler authority.
    """
    lease = _ACTIVE_LEASE.get()
    if lease is None:
        yield None
        return
    _verify_lease(lease)
    if (
        not isinstance(owner_sha256, str)
        or len(owner_sha256) != 64
        or any(char not in "0123456789abcdef" for char in owner_sha256)
    ):
        raise ValueError("compiler CPU transfer needs its exact original owner digest")
    if os.sched_getaffinity(0) != {lease.cpu}:
        raise RuntimeError("compiler CPU transfer differs from its issuer affinity")
    fd = os.memfd_create("original-compiler-cpu", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
    read_fd, write_fd = os.pipe2(os.O_CLOEXEC)
    try:
        body = {
            "version": 1,
            "owner_sha256": owner_sha256,
            "issuer_pid": os.getpid(),
            "issuer_birth": _birth(os.getpid()),
            "issuer_tid": threading.get_native_id(),
            "issuer_thread_birth": _birth(os.getpid(), threading.get_native_id()),
            "root": str(lease.root),
            "cpu": lease.cpu,
            "slot_fd": lease.fd,
            "slot_identity": list(_slot_identity(lease.root, lease.cpu, lease.fd)),
        }
        os.write(fd, json.dumps(body, sort_keys=True).encode())
        fcntl.fcntl(fd, _ADD_SEALS, _SEALS)
        yield CompilerCPUTransfer(fd, lease.fd, read_fd)
        _verify_lease(lease)
    finally:
        os.close(write_fd)
        os.close(read_fd)
        os.close(fd)


@contextmanager
def borrow_compiler_cpu(transfer, *, owner_sha256):
    """Activate the exact private parent slot in one trusted broker thread."""
    if type(transfer) is not CompilerCPUTransfer or _ACTIVE_LEASE.get() is not None:
        raise RuntimeError("compiler CPU borrower lacks its exact original transfer")
    if fcntl.fcntl(transfer.metadata_fd, _GET_SEALS) & _SEALS != _SEALS:
        raise RuntimeError("compiler CPU metadata is not immutable")
    size = os.fstat(transfer.metadata_fd).st_size
    if not 0 < size <= 8192:
        raise RuntimeError("compiler CPU metadata exceeds its bounded record")
    body = json.loads(os.pread(transfer.metadata_fd, size, 0))
    if (
        type(body) is not dict
        or set(body)
        != {
            "version",
            "owner_sha256",
            "issuer_pid",
            "issuer_birth",
            "issuer_tid",
            "issuer_thread_birth",
            "root",
            "cpu",
            "slot_fd",
            "slot_identity",
        }
        or body["version"] != 1
        or body["owner_sha256"] != owner_sha256
        or type(body["cpu"]) is not int
        or type(body["issuer_pid"]) is not int
        or type(body["issuer_tid"]) is not int
        or body["slot_fd"] != transfer.slot_fd
    ):
        raise RuntimeError("compiler CPU metadata differs from its original owner")
    root, cpu = Path(body["root"]), body["cpu"]
    if not root.is_absolute() or root.resolve() != root or root.is_symlink():
        raise RuntimeError("compiler CPU slot root is not an ordinary canonical path")
    poll = select.poll()
    poll.register(transfer.lifetime_fd, select.POLLIN | select.POLLHUP | select.POLLERR)

    def live():
        if (
            os.getppid() != body["issuer_pid"]
            or poll.poll(0)
            or _birth(body["issuer_pid"]) != body["issuer_birth"]
            or _birth(body["issuer_pid"], body["issuer_tid"]) != body["issuer_thread_birth"]
            or os.sched_getaffinity(body["issuer_tid"]) != {cpu}
            or list(_slot_identity(root, cpu, transfer.slot_fd)) != body["slot_identity"]
        ):
            raise RuntimeError("compiler CPU issuer/slot lifetime is no longer live")
        original = Path(f"/proc/{body['issuer_pid']}/fd/{body['slot_fd']}").stat()
        current = os.fstat(transfer.slot_fd)
        if (original.st_dev, original.st_ino) != (current.st_dev, current.st_ino):
            raise RuntimeError("compiler CPU slot differs from its live parent descriptor")
        # The inherited descriptor shares the parent's open-file description;
        # this never waits for or acquires a different CPU slot.
        fcntl.flock(transfer.slot_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    live()
    original = os.sched_getaffinity(0)
    lease = _Lease(os.getpid(), threading.current_thread(), root, cpu, transfer.slot_fd, live)
    _ISSUED_LEASES[lease] = _lease_fields(lease)
    token = _ACTIVE_LEASE.set(lease)
    try:
        os.sched_setaffinity(0, {cpu})
        if os.sched_getaffinity(0) != {cpu}:
            raise RuntimeError("borrowed compiler CPU is unavailable in this partition")
        yield
        _verify_lease(lease)
    finally:
        lease.active = False
        _ACTIVE_LEASE.reset(token)
        os.sched_setaffinity(0, original)


def parse_roster(raw: str) -> tuple[int, ...]:
    if not raw:
        return ()
    tokens = raw.split(",")
    if any(not token.isdecimal() for token in tokens):
        raise ValueError("CPU pool requires comma-separated CPU numbers")
    cpus = tuple(int(token) for token in tokens)
    if len(set(cpus)) != len(cpus):
        raise ValueError("CPU pool requires distinct CPU numbers")
    return cpus


def _acquire_slot(
    root: Path, cpus: tuple[int, ...], *, timeout_s: float | None, queued_since_ns: int | None
) -> tuple[int, int]:
    """Take one free slot in priority order across every thread and process of the namespace.

    A waiter may take a free slot only while fewer live waiters are queued ahead
    of it (a more urgent class, or an earlier waiter of its own class) than slots
    are free: with one free slot only the most urgent, longest-waiting waiter
    proceeds, and a lower class never takes a slot a queued higher class needs.
    FIFO within a class; no slot is held back for a class with no waiter. A
    newcomer queues an entry only when it cannot take a slot at once.
    ``queued_since_ns`` keeps a retrying caller's place in its class.
    """
    cpus = tuple(cpus)
    level = lease_priority()
    queued_ns = time.monotonic_ns() if queued_since_ns is None else int(queued_since_ns)
    if queued_ns < 0:
        raise ValueError("CPU pool queue position must be a monotonic timestamp")
    directory = root / _WAITER_DIR
    key = (level, queued_ns, os.getpid(), threading.get_native_id(), "")
    started = time.monotonic()
    waiter = fd = cpu = None
    try:
        while fd is None:
            if waiter is not None:
                waiter.beat()
                key = waiter.key
            ahead = bisect.bisect_left(_queued_keys(directory), key)
            if ahead == 0 or (ahead < len(cpus) and _free_slots(root, cpus) > ahead):
                # Concurrently admitted waiters start at different slots to avoid colliding.
                fd, cpu = _try_slots(root, cpus, start=ahead)
                if fd is not None:
                    _FREE_VIEWS.pop((root, cpus), None)
                    try:
                        holds = ahead == 0 or _admission_holds(root, cpus, directory, key)
                    except BaseException:
                        os.close(fd)
                        raise
                    if holds:
                        break
                    # A stale view admitted us past a waiter that now needs this slot.
                    os.close(fd)
                    fd = None
                    _wake_local_waiters()
            if timeout_s is not None and time.monotonic() - started >= timeout_s:
                raise TimeoutError("CPU build pool wait budget expired")
            if waiter is None:
                waiter = _Waiter(root, level, queued_ns)
                continue
            waiter.wake.wait(_FRONT_POLL_S if ahead == 0 else _NEAR_POLL_S if ahead < len(cpus) else _DEEP_POLL_S)
            waiter.wake.clear()
    finally:
        if waiter is not None:
            try:
                waiter.release()
            except BaseException:
                if fd is not None:
                    os.close(fd)  # Never strand an acquired slot behind a failed dequeue.
                raise
    return fd, cpu


@contextmanager
def selected_pool(
    cpus: tuple[int, ...],
    *,
    slot_root: Path | None = None,
    timeout_s: float | None = None,
    queued_since_ns: int | None = None,
):
    """Children inherit this thread's affinity; restore it without preexec callbacks.

    An explicit slot directory grants one exclusive build per selected CPU across
    threads and processes, handed out in :func:`lease_priority` order. Without it,
    children may use the complete selected pool.
    """
    if not cpus:
        yield
        return
    if timeout_s is not None and (type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or timeout_s < 0):
        raise ValueError("CPU pool wait budget must be finite and nonnegative")
    held = _ACTIVE_LEASE.get()
    if held is not None and held.active and held.pid == os.getpid() and held.thread is threading.current_thread():
        _verify_lease(held)
        if slot_root is None:
            if set(cpus) != {held.cpu}:
                raise RuntimeError("nested CPU pool would widen its held lease")
        elif held.root != slot_root:
            raise RuntimeError("nested CPU slot directory differs from its held lease")
        if held.cpu not in cpus or os.sched_getaffinity(0) != {held.cpu}:
            raise RuntimeError("nested CPU pool differs from its held lease")
        yield
        _verify_lease(held)
        return
    original = os.sched_getaffinity(0)
    locked_fd = None
    token = None
    lease = None
    chosen = set(cpus)
    try:
        if slot_root is not None:
            if not slot_root.is_absolute() or slot_root.is_symlink():
                raise ValueError("CPU slot directory must be an absolute ordinary path")
            slot_root.mkdir(parents=True, exist_ok=True)
            locked_fd, cpu = _acquire_slot(slot_root, cpus, timeout_s=timeout_s, queued_since_ns=queued_since_ns)
            chosen = {cpu}
        os.sched_setaffinity(0, chosen)
        if os.sched_getaffinity(0) != chosen:
            raise RuntimeError("selected CPU pool is unavailable in the process partition")
        if locked_fd is not None:
            lease = _Lease(os.getpid(), threading.current_thread(), slot_root, next(iter(chosen)), locked_fd)
            _ISSUED_LEASES[lease] = _lease_fields(lease)
            token = _ACTIVE_LEASE.set(lease)
        yield
    finally:
        if token is not None:
            lease.active = False
            _ACTIVE_LEASE.reset(token)
        try:
            os.sched_setaffinity(0, original)
        finally:
            if locked_fd is not None:
                os.close(locked_fd)
                _FREE_VIEWS.pop((slot_root, tuple(cpus)), None)
                _wake_local_waiters()


def _shared_compute_selection(environ: Mapping[str, str]) -> tuple[tuple[int, ...], Path] | None:
    keys = ("MERLIN_COMPUTE_CPUS", "MERLIN_COMPUTE_CPU_SLOTS")
    if not any(key in environ for key in keys):
        return None
    cpus = parse_roster(environ.get(keys[0], "").strip())
    root = Path(environ.get(keys[1], "").strip())
    if (
        not cpus
        or not environ.get(keys[1], "").strip()
        or not root.is_absolute()
        or root.is_symlink()
        or (root.exists() and not root.is_dir())
    ):
        raise ValueError("shared compute requires CPU roster and absolute ordinary CPU slot directory")
    return cpus, root


def shared_compute_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """An incomplete selection refuses instead of falling back to separate pools."""
    return _shared_compute_selection(os.environ if environ is None else environ) is not None


def grading_cpus(environ: Mapping[str, str] | None = None) -> tuple[int, ...]:
    """Affinity for preparation workers; acquiring a slot is the actual step owner's job."""
    environment = os.environ if environ is None else environ
    selected = _shared_compute_selection(environment)
    return selected[0] if selected is not None else parse_roster(environment.get("MERLIN_GRADE_CPUS", ""))


def compute_capacity(environ: Mapping[str, str] | None = None) -> int:
    """CPUs in the selected shared compute pool; 0 when no pool is selected.

    Unset per-engine concurrency caps default to this, so the pool's slots are
    the only limiter and ``MERLIN_COMPUTE_CPUS`` alone sizes every engine.
    """
    selected = _shared_compute_selection(os.environ if environ is None else environ)
    return len(selected[0]) if selected is not None else 0


def preparation_cpus(environ: Mapping[str, str] | None = None) -> tuple[int, ...]:
    """Affinity for unleased preparation; its bounded CPU steps lease shared slots.

    Without a separate roster, preparation runs on the shared compute CPUs and
    its CPU-bound steps take slots from the same pool (at :data:`PREPARATION`
    when no more urgent requester owns them). An explicit roster may overlap
    the pool. Preparation scopes stay unleased because they wait on children
    and RPCs that acquire their own slots.
    """
    environment = os.environ if environ is None else environ
    raw = environment.get("MERLIN_PREPARATION_CPUS", "").strip()
    if not raw:
        return grading_cpus(environment)
    cpus = parse_roster(raw)
    if not cpus:
        raise ValueError("preparation CPU roster must be nonempty")
    _shared_compute_selection(environment)  # An incomplete pool selection still refuses.
    return cpus


@contextmanager
def shared_compute_pool(*, timeout_s: float | None = None, queued_since_ns: int | None = None):
    """Lease one compute CPU for a supervised build/simulator or bounded preparation step.

    Use the same namespace across engines. Never hold a capsule/parent lease while
    waiting for a child or RPC that acquires its own lease. Same-thread nested
    steps reuse one lease; absent selection leaves legacy callers unchanged.
    Waiters are served in :func:`lease_priority` order, FIFO within a class.
    """
    selected = _shared_compute_selection(os.environ)
    with ExitStack() as stack:
        label = "shared compute lease" if selected is not None else "no shared compute lease"
        with timed_event("compute_acquire", label):
            if selected is not None:
                cpus, root = selected
                stack.enter_context(
                    selected_pool(cpus, slot_root=root, timeout_s=timeout_s, queued_since_ns=queued_since_ns)
                )
        yield


def _carried_priority(env: Mapping[str, str]) -> int | None:
    """The caller's explicit lease class that a worker launched with ``env`` must carry."""
    if _shared_compute_selection(env) is None:
        return None
    return _explicit_priority()


def child_environment(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """A copy of ``environ`` (default: this process's) carrying the caller's lease class.

    Build a worker environment with this whenever it is recorded or sealed before
    launch. :func:`worker_pool_command` then finds the class already present and
    adds no ``env`` prefix, so the worker receives exactly the recorded mapping.
    Without a shared pool or an explicit class the copy is unchanged.
    """
    environment = dict(os.environ if environ is None else environ)
    level = _carried_priority(environment)
    if level is not None:
        environment[PRIORITY_ENV] = str(level)
    return environment


def worker_pool_command(command: list[str], env: dict[str, str]) -> list[str]:
    """Select unleased preparation affinity before the worker interpreter imports code.

    This parent-side command owns no CPU slot: actual child step owners acquire
    theirs after launch, so waiting parent workers cannot exhaust the pool. With
    a shared pool, the caller's lease class travels with the worker; an ``env``
    prefix supplies it when ``env`` lacks it, which changes the worker's
    environment, so a caller that records ``env`` first builds it with
    :func:`child_environment`.
    """
    level = _carried_priority(env)
    if level is not None and env.get(PRIORITY_ENV) != str(level):
        command = ["env", f"{PRIORITY_ENV}={level}", *command]
    cpus = preparation_cpus(env)
    if not cpus:
        return command
    return ["taskset", "--cpu-list", ",".join(map(str, cpus)), *command]
