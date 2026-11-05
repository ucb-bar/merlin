"""Owned descendant discovery and termination for budgeted grading children."""

from __future__ import annotations

import os
import time
from pathlib import Path


def _die_with_parent() -> None:
    """Ask the kernel to SIGKILL this process when its parent goes away.

    The budgeted whole-model grade runs in its OWN SESSION so the parent can kill the process group
    when the ceiling expires. That protects against a grade overrunning; it does nothing when the
    PARENT is killed, because the parent's cleanup never runs. MEASURED (2026-08-30): stopping a
    regrade left the model-grade child and its Verilator alive on a shared host, reparented to init,
    with nothing left that knew to reap them.

    Set from inside the CHILD rather than through ``preexec_fn``: the parent grades capsules on a
    ThreadPoolExecutor, and ``preexec_fn`` is documented as unsafe in a threaded process. The cost is a
    small race -- the parent could die during interpreter startup -- which the getppid() re-check
    below closes.
    """
    try:
        import ctypes
        import signal as _sig

        ppid = os.getppid()
        PR_SET_PDEATHSIG = 1
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(PR_SET_PDEATHSIG, _sig.SIGKILL)
        if os.getppid() != ppid:  # already reparented: the signal will never come
            os._exit(1)
    except Exception:  # noqa: BLE001 — a best-effort safeguard, never a blocker
        pass


def _descendants(pid: int) -> list[int]:
    """Every live descendant of *pid*, deepest first, read from /proc.

    Must be collected BEFORE any signal is sent: once a process is orphaned its parent link is gone.
    A process GROUP is not enough. MEASURED (2026-08-30): a budgeted whole-model grade's Verilator
    child had called ``setsid`` of its own, so it sat in neither the grade child's group nor its
    session; ``killpg`` reaped the python and left the simulator running with PPID 1, holding a core
    on a shared host.
    """
    kids: dict[int, list[int]] = {}
    try:
        entries = list(Path("/proc").iterdir())
    except OSError:
        return []
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            # /proc/<pid>/stat is "pid (comm) state ppid ..."; comm may itself contain spaces and
            # parentheses, so split after the LAST ')' rather than on whitespace from the left.
            ppid = int((entry / "stat").read_text().rpartition(")")[2].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        kids.setdefault(ppid, []).append(int(entry.name))
    out: list[int] = []
    frontier = list(kids.get(pid, ()))
    while frontier:
        out.extend(frontier)
        frontier = [g for p in frontier for g in kids.get(p, ())]
    out.reverse()  # deepest generation first
    return out


def _running(pid: int) -> bool:
    """True while *pid* is a live process. A ZOMBIE is not: it has exited and merely awaits a wait(),
    so ``/proc/<pid>`` still exists and a liveness check written that way waits out its whole grace."""
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()[0]
    except (OSError, IndexError):
        return False
    return state != "Z"


def _kill_tree(pid: int, *, grace: float = 5.0) -> None:
    """SIGTERM then SIGKILL *pid* and every descendant it had when we looked."""
    import signal as _signal

    victims = _descendants(pid) + [pid]
    for sig in (_signal.SIGTERM, _signal.SIGKILL):
        for victim in victims:
            try:
                os.kill(victim, sig)
            except OSError:  # already gone, or not ours
                pass
        try:  # and the group, for anything spawned since the scan
            os.killpg(os.getpgid(pid), sig)
        except OSError:
            pass
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            if not any(_running(v) for v in victims):
                return
            time.sleep(0.1)
