"""Explicit Linux CPU pools for trusted subprocess launch threads."""

from __future__ import annotations

import fcntl
import math
import os
import threading
import time
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path

_ACTIVE_LEASE = ContextVar("selected_cpu_lease", default=None)


@dataclass
class _Lease:
    pid: int
    thread: threading.Thread
    root: Path
    cpu: int
    active: bool = True


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
        if slot_root is None:
            if set(cpus) != {held.cpu}:
                raise RuntimeError("nested CPU pool would widen its held lease")
        elif held.root != slot_root:
            raise RuntimeError("nested CPU slot directory differs from its held lease")
        if held.cpu not in cpus or os.sched_getaffinity(0) != {held.cpu}:
            raise RuntimeError("nested CPU pool differs from its held lease")
        yield
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
            lease = _Lease(os.getpid(), threading.current_thread(), slot_root, next(iter(chosen)))
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
    """Select shared/legacy grading affinity before the worker interpreter imports code.

    This parent-side command owns no CPU slot: actual child step owners acquire
    theirs after launch, so waiting parent workers cannot exhaust the pool.
    """
    cpus = grading_cpus(env)
    if not cpus:
        return command
    return ["taskset", "--cpu-list", ",".join(map(str, cpus)), *command]
