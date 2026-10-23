"""Host-wide advisory slots for direct Verilator capsule execution."""

from __future__ import annotations

import fcntl
import os
import stat
import time
from contextlib import contextmanager
from pathlib import Path

from .rtl_engine_policy import capsule_worker_cap


def _private_directory(root: Path) -> None:
    root.mkdir(mode=0o700, exist_ok=True)
    info = root.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise RuntimeError(f"simulator slot directory is not private and owned by this user: {root}")


def _open_slot(path: Path) -> int:
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        os.close(fd)
        raise RuntimeError(f"simulator slot file is not private and owned by this user: {path}")
    return fd


def _try_reserve(root: Path) -> int | None:
    for index in range(capsule_worker_cap("verilator")):
        opened = _open_slot(root / f"slot_{index}.lock")
        try:
            fcntl.flock(opened, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(opened)
            continue
        except BaseException:
            os.close(opened)
            raise
        return opened
    return None


def _wait_for_slot(root: Path, deadline: float) -> tuple[int, float]:
    fd = None
    try:
        while fd is None:
            fd = _try_reserve(root)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Verilator slot wait exhausted the capsule deadline")
            if fd is None:
                time.sleep(min(0.1, remaining))
        return fd, remaining
    except BaseException:
        if fd is not None:
            os.close(fd)
        raise


@contextmanager
def verilator_slot(*, timeout: float, root: Path | None = None):
    """Reserve one of the same user's shared slots; release on exit or process death.

    The wait consumes the caller's wall budget. These advisory locks coordinate
    Merlin adapters; unrelated native launches do not participate in the policy.
    Callers must not nest reservations or run concurrent children in one slot.
    """
    root = root or Path("/tmp") / f"merlin_verilator_slots_{os.getuid()}"
    _private_directory(root)
    deadline = time.monotonic() + timeout
    fd, remaining = _wait_for_slot(root, deadline)
    try:
        yield remaining
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
