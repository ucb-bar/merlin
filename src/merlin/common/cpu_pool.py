"""Explicit Linux CPU pools for trusted subprocess launch threads."""

from __future__ import annotations

import fcntl
import os
import time
from contextlib import contextmanager
from pathlib import Path


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
    original = os.sched_getaffinity(0)
    locked_fd = None
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
        yield
    finally:
        os.sched_setaffinity(0, original)
        if locked_fd is not None:
            os.close(locked_fd)


def worker_pool_command(command: list[str], env: dict[str, str]) -> list[str]:
    """Select grading affinity before the worker interpreter imports any code."""
    cpus = parse_roster(env.get("MERLIN_GRADE_CPUS", ""))
    if not cpus:
        return command
    return ["taskset", "--cpu-list", ",".join(map(str, cpus)), *command]
