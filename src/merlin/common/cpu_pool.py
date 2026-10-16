"""Explicit Linux CPU pools for trusted subprocess launch threads."""

from __future__ import annotations

import fcntl
import json
import math
import os
import select
import stat
import threading
import time
import weakref
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path

_ACTIVE_LEASE = ContextVar("selected_cpu_lease", default=None)
_ISSUED_LEASES = weakref.WeakKeyDictionary()
# Linux UAPI values; some supported CPython builds omit the fcntl names.
_SEALS = 0x000F
_ADD_SEALS = getattr(fcntl, "F_ADD_SEALS", 1033)
_GET_SEALS = getattr(fcntl, "F_GET_SEALS", 1034)


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
    if (not stat.S_ISREG(before.st_mode) or not stat.S_ISREG(current.st_mode)
            or (before.st_dev, before.st_ino) != (current.st_dev, current.st_ino)):
        raise RuntimeError("CPU lease lost its exact ordinary slot file")
    return before.st_dev, before.st_ino


def _verify_lease(lease):
    if (type(lease) is not _Lease or not lease.active or lease.pid != os.getpid()
            or lease.thread is not threading.current_thread()
            or _ISSUED_LEASES.get(lease) != _lease_fields(lease)):
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
    if (not isinstance(owner_sha256, str) or len(owner_sha256) != 64
            or any(char not in "0123456789abcdef" for char in owner_sha256)):
        raise ValueError("compiler CPU transfer needs its exact original owner digest")
    if os.sched_getaffinity(0) != {lease.cpu}:
        raise RuntimeError("compiler CPU transfer differs from its issuer affinity")
    fd = os.memfd_create("original-compiler-cpu", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
    read_fd, write_fd = os.pipe2(os.O_CLOEXEC)
    try:
        body = {"version": 1, "owner_sha256": owner_sha256, "issuer_pid": os.getpid(),
                "issuer_birth": _birth(os.getpid()), "issuer_tid": threading.get_native_id(),
                "issuer_thread_birth": _birth(os.getpid(), threading.get_native_id()),
                "root": str(lease.root), "cpu": lease.cpu, "slot_fd": lease.fd,
                "slot_identity": list(_slot_identity(lease.root, lease.cpu, lease.fd))}
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
    if (type(body) is not dict or set(body) != {"version", "owner_sha256", "issuer_pid", "issuer_birth",
            "issuer_tid", "issuer_thread_birth", "root", "cpu", "slot_fd", "slot_identity"}
            or body["version"] != 1 or body["owner_sha256"] != owner_sha256
            or type(body["cpu"]) is not int or type(body["issuer_pid"]) is not int
            or type(body["issuer_tid"]) is not int or body["slot_fd"] != transfer.slot_fd):
        raise RuntimeError("compiler CPU metadata differs from its original owner")
    root, cpu = Path(body["root"]), body["cpu"]
    if not root.is_absolute() or root.resolve() != root or root.is_symlink():
        raise RuntimeError("compiler CPU slot root is not an ordinary canonical path")
    poll = select.poll()
    poll.register(transfer.lifetime_fd, select.POLLIN | select.POLLHUP | select.POLLERR)

    def live():
        if (os.getppid() != body["issuer_pid"] or poll.poll(0)
                or _birth(body["issuer_pid"]) != body["issuer_birth"]
                or _birth(body["issuer_pid"], body["issuer_tid"]) != body["issuer_thread_birth"]
                or os.sched_getaffinity(body["issuer_tid"]) != {cpu}
                or list(_slot_identity(root, cpu, transfer.slot_fd)) != body["slot_identity"]):
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


@contextmanager
def selected_pool(cpus: tuple[int, ...], *, slot_root: Path | None = None, timeout_s: float | None = None):
    """Children inherit this thread's affinity; restore it without preexec callbacks.

    An explicit slot directory grants one exclusive build per selected CPU across
    threads and processes. Without it, children may use the complete selected pool.
    """
    if not cpus:
        yield
        return
    if timeout_s is not None and (
        type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or timeout_s < 0
    ):
        raise ValueError("CPU pool wait budget must be finite and nonnegative")
    held = _ACTIVE_LEASE.get()
    if (held is not None and held.active and held.pid == os.getpid()
            and held.thread is threading.current_thread()):
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
            started = time.monotonic()
            while locked_fd is None:
                for cpu in cpus:
                    fd = os.open(slot_root / f"cpu-{cpu}.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
                    try:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        os.close(fd)
                        continue
                    locked_fd = fd
                    chosen = {cpu}
                    break
                if locked_fd is None:
                    if timeout_s is not None and time.monotonic() - started >= timeout_s:
                        raise TimeoutError("CPU build pool wait budget expired")
                    time.sleep(0.01)
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


def _shared_compute_selection(environ: Mapping[str, str]) -> tuple[tuple[int, ...], Path] | None:
    keys = ("MERLIN_COMPUTE_CPUS", "MERLIN_COMPUTE_CPU_SLOTS")
    if not any(key in environ for key in keys):
        return None
    cpus = parse_roster(environ.get(keys[0], "").strip())
    root = Path(environ.get(keys[1], "").strip())
    if (not cpus or not environ.get(keys[1], "").strip() or not root.is_absolute() or root.is_symlink()
            or (root.exists() and not root.is_dir())):
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


def preparation_cpus(environ: Mapping[str, str] | None = None) -> tuple[int, ...]:
    """Affinity for unleased preparation; compute steps still take shared slots.

    An explicit preparation roster must not overlap the leased compute roster.
    Legacy callers without a separate roster retain the existing behavior.
    """
    environment = os.environ if environ is None else environ
    raw = environment.get("MERLIN_PREPARATION_CPUS", "").strip()
    if not raw:
        return grading_cpus(environment)
    cpus = parse_roster(raw)
    if not cpus:
        raise ValueError("preparation CPU roster must be nonempty")
    selected = _shared_compute_selection(environment)
    if selected is not None and set(cpus) & set(selected[0]):
        raise ValueError("preparation CPUs overlap leased compute CPUs")
    return cpus


@contextmanager
def shared_compute_pool(*, timeout_s: float | None = None):
    """Lease one compute CPU for a supervised build/simulator or bounded preparation step.

    Use the same namespace across engines. Never hold a capsule/parent lease while
    waiting for a child or RPC that acquires its own lease. Same-thread nested
    steps reuse one lease; absent selection leaves legacy callers unchanged.
    """
    selected = _shared_compute_selection(os.environ)
    if selected is None:
        yield
        return
    cpus, root = selected
    with selected_pool(cpus, slot_root=root, timeout_s=timeout_s):
        yield


def worker_pool_command(command: list[str], env: dict[str, str]) -> list[str]:
    """Select unleased preparation affinity before the worker interpreter imports code.

    This parent-side command owns no CPU slot: actual child step owners acquire
    theirs after launch, so waiting parent workers cannot exhaust the pool.
    """
    cpus = preparation_cpus(env)
    if not cpus:
        return command
    return ["taskset", "--cpu-list", ",".join(map(str, cpus)), *command]
