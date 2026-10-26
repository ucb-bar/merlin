"""Who holds which CPUs, and keeping the console itself off the experiments' CPUs.

Allocations come from two places, both read-only:

* cgroup v2 groups (a bounded walk of the unified hierarchy) with an explicit ``cpuset.cpus``. Nothing
  about the machine is assumed: no CPU numbers and no slice names. The *general* pool is the CPUs the
  console's own cgroup (``/proc/self/cgroup``) may use; a group containing the console, or confined
  within those CPUs, is general. Any other group that confines CPUs carves them out of the pool and is
  a *reservation*, whatever it is called and however deep it sits;
* the CPU affinity (``Cpus_allowed_list``) of every live experiment runner, when it is narrower than
  the machine.

Without cgroup v2 (or without read access) only the runner affinities are reported.

:func:`place_console` moves the console's own processes (this data process and, when it is the parent,
the Bubble Tea frontend) to CPUs no live experiment holds and lowers their priority. When every CPU is
held it keeps the affinity and drops to ``SCHED_IDLE``, so the console only runs on time the experiment
leaves unused. It changes nothing but the console's own scheduling.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from .. import terminal_probe as probe
from .discovery import runner_processes
from .io import scan_dir

MAX_DEPTH = 4
MAX_GROUPS = 1024
FRONTEND_NAMES = frozenset({"merlin-console"})
NICE = 10


def affinity(pid: int, proc_root: Path = Path("/proc")) -> list[int]:
    raw = probe.read(proc_root / str(pid) / "status", 16384) or b""
    for line in raw.decode(errors="replace").splitlines():
        if line.startswith("Cpus_allowed_list:"):
            return probe.cpus(line.split(":", 1)[1].strip())
    return []


def online_cpus(proc_root: Path = Path("/proc")) -> list[int]:
    return sorted(probe.cpu_counters(proc_root))


def _procs(cgroup: Path, limit: int = 4096) -> int:
    """Processes in a cgroup subtree (bounded walk)."""
    count, stack, seen = 0, [cgroup], 0
    while stack and seen < 512:
        node = stack.pop()
        seen += 1
        raw = probe.read(node / "cgroup.procs", 1 << 20) or b""
        count += sum(1 for line in raw.split(b"\n") if line.strip())
        if count >= limit:
            break
        stack.extend(Path(e.path) for e in scan_dir(node, 1024) if e.is_dir(follow_symlinks=False))
    return count


def group_pids(name: str, cgroup_root: Path = Path("/sys/fs/cgroup"), limit: int = 1024) -> list[int]:
    """Process ids in a cgroup subtree (``name`` relative to the root; bounded walk)."""
    out, stack, seen = [], [cgroup_root / name.lstrip("/")], 0
    while stack and seen < 256 and len(out) < limit:
        node = stack.pop()
        seen += 1
        raw = probe.read(node / "cgroup.procs", 1 << 20) or b""
        out.extend(int(x) for x in raw.split() if x.isdigit())
        stack.extend(Path(e.path) for e in scan_dir(node, 1024) if e.is_dir(follow_symlinks=False))
    return out[:limit]


def own_cgroup(proc_root: Path = Path("/proc")) -> str | None:
    """This process's unified-hierarchy cgroup path (``/a/b``), or None without cgroup v2."""
    raw = probe.read(proc_root / "self" / "cgroup", 65536) or b""
    for line in raw.decode(errors="replace").splitlines():
        hierarchy, _, rest = line.partition(":")
        controllers, _, path = rest.partition(":")
        if hierarchy == "0" and controllers == "" and path.startswith("/"):
            return path
    return None


def _groups(cgroup_root: Path):
    """``(relative path, directory)`` of cgroups, breadth first, bounded in depth and count."""
    level, seen = [("", cgroup_root)], 0
    for _ in range(MAX_DEPTH):
        nxt = []
        for rel, node in level:
            for entry in scan_dir(node, 4096):
                if not entry.is_dir(follow_symlinks=False):
                    continue
                seen += 1
                if seen > MAX_GROUPS:
                    return
                child = (f"{rel}/{entry.name}", Path(entry.path))
                yield child
                nxt.append(child)
        level = nxt


def allocations(
    proc_root: Path = Path("/proc"),
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    runners: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """``[{name, kind: "reservation"|"general"|"run", cpus, partition, processes, pid}]``."""
    out: list[dict[str, Any]] = []
    online = set(online_cpus(proc_root))
    own = own_cgroup(proc_root)
    pool: set[int] = set()
    if own is not None:
        raw = probe.read(cgroup_root / own.lstrip("/") / "cpuset.cpus.effective", 4096)
        pool = set(probe.cpus((raw or b"").decode(errors="replace").strip()))
    if not pool:
        try:
            pool = set(os.sched_getaffinity(0))
        except OSError:
            pool = online
    for rel, node in _groups(cgroup_root):
        raw = probe.read(node / "cpuset.cpus", 4096)
        text = (raw or b"").decode(errors="replace").strip()
        cpus = probe.cpus(text) if text else []
        if not cpus or set(cpus) >= online:
            continue  # No confinement here.
        general = (own is not None and (own == rel or own.startswith(rel + "/"))) or set(cpus) <= pool
        partition = (probe.read(node / "cpuset.cpus.partition", 256) or b"").decode().strip() or None
        out.append(
            {
                "name": rel.lstrip("/"),
                "kind": "general" if general else "reservation",
                "cpus": cpus,
                "partition": partition,
                "processes": _procs(node),
                "pid": None,
            }
        )
    for run_id, pid in sorted((runners if runners is not None else runner_processes(proc_root)).items()):
        cpus = affinity(pid, proc_root)
        if cpus and set(cpus) != online:
            out.append({"name": run_id, "kind": "run", "cpus": cpus, "partition": None, "processes": None, "pid": pid})
    return out


def held_cpus(allocs: list[dict[str, Any]]) -> set[int]:
    """CPUs a live experiment holds: runner affinities, plus reservations that hold processes."""
    held: set[int] = set()
    for alloc in allocs:
        if alloc["kind"] == "run" or (alloc["kind"] == "reservation" and alloc["processes"]):
            held.update(alloc["cpus"])
    return held


def _console_pids() -> list[int]:
    pids = [os.getpid()]
    parent = os.getppid()
    comm = (probe.read(Path("/proc") / str(parent) / "comm", 256) or b"").decode(errors="replace").strip()
    if comm in FRONTEND_NAMES:
        pids.append(parent)
    return pids


def place_console(allocs: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Move the console to CPUs no experiment holds; return what was applied (never raises)."""
    allocs = allocations() if allocs is None else allocs
    held = held_cpus(allocs)
    pids = _console_pids()
    try:
        current = set(os.sched_getaffinity(0))
    except OSError:
        current = set(online_cpus())
    free = sorted(current - held) or sorted(set(online_cpus()) - held)
    policy = "nice"
    for pid in pids:
        try:
            if free:
                os.sched_setaffinity(pid, free)
            os.setpriority(os.PRIO_PROCESS, pid, max(NICE, os.getpriority(os.PRIO_PROCESS, pid)))
            if not free and hasattr(os, "SCHED_IDLE"):
                os.sched_setscheduler(pid, os.SCHED_IDLE, os.sched_param(0))
                policy = "idle"
        except (OSError, AttributeError, ValueError):
            continue
    try:
        cpus = sorted(os.sched_getaffinity(0))
    except OSError:
        cpus = []
    return {
        "cpus": cpus,
        "avoided": sorted(held),
        "policy": policy if free else "idle",
        "shared": not free,
        "pids": pids,
    }


class SelfCost:
    """CPU and memory the console itself uses (this process plus the frontend), sampled from /proc."""

    def __init__(self) -> None:
        self.previous: tuple[float, float] | None = None
        self.hz = os.sysconf("SC_CLK_TCK")

    def sample(self, pids: list[int]) -> dict[str, Any]:
        ticks, rss = 0, 0
        for pid in pids:
            row = probe.task_stat(pid)
            if row:
                ticks += row["cpu_ticks"]
            status = probe.read(Path("/proc") / str(pid) / "status", 16384) or b""
            for line in status.decode(errors="replace").splitlines():
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) > 1 and parts[1].isdigit():
                        rss += int(parts[1]) * 1024
        now, seconds = time.monotonic(), ticks / self.hz
        pct = None
        if self.previous and now > self.previous[0]:
            pct = max(0.0, 100 * (seconds - self.previous[1]) / (now - self.previous[0]))
        self.previous = (now, seconds)
        return {"cpu_pct": pct, "rss_bytes": rss or None}


__all__ = [
    "SelfCost",
    "affinity",
    "allocations",
    "group_pids",
    "held_cpus",
    "online_cpus",
    "own_cgroup",
    "place_console",
]
