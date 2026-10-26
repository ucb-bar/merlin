"""Every CPU of the machine: its utilization now and who it is allocated to.

The core map shows all online CPUs, not just the selected experiment's: each core carries its measured
utilization (``/proc/stat`` deltas between refreshes, ``None`` on the first sample) and its holders
from :func:`..placement.allocations` (reservation slices, live runners' affinities, the general pool).
``mine`` marks the cores the selected experiment's runner may use; ``console`` marks the cores the
console itself is confined to. Allocations are re-read every :data:`ALLOC_TTL_S` seconds (they need a
``/proc`` scan); utilization every refresh. Display telemetry only.

Section::

    {"cores": [{cpu, util_pct, holders: [name], mine, console}], "allocations": [{name, kind, cpus,
     partition, processes, pid}], "busy_cpus", "total_cpus", "experiment_cpus": [..],
     "console_cpus": [..], "volume": {path, free_bytes, total_bytes, free_pct, level}|None, "notes"}

``volume`` is the filesystem holding the selected experiment's run directory (``statvfs``, one call
per refresh). ``level`` is ``ok``, ``warn`` (below :data:`VOLUME_WARN_PCT` percent or
:data:`VOLUME_WARN_BYTES` free) or ``critical`` (below the ``CRITICAL`` thresholds), whoever fills it:
other runs on the same volume can stop this one.
"""

from __future__ import annotations

import os
import time
from typing import Any

from ... import terminal_probe as probe
from .. import placement

ALLOC_TTL_S = 10.0
FIRST_SAMPLE_S = 0.1
VOLUME_WARN_PCT, VOLUME_WARN_BYTES = 10.0, 50 << 30
VOLUME_CRITICAL_PCT, VOLUME_CRITICAL_BYTES = 3.0, 15 << 30


def volume(path) -> dict | None:
    """Free space of the filesystem holding ``path``; ``None`` when it cannot be measured."""
    if not path:
        return None
    try:
        info = os.statvfs(path)
    except OSError:
        return None
    total, free = info.f_blocks * info.f_frsize, info.f_bavail * info.f_frsize
    if total <= 0:
        return None
    pct = round(100 * free / total, 1)
    level = "ok"
    if pct < VOLUME_CRITICAL_PCT or free < VOLUME_CRITICAL_BYTES:
        level = "critical"
    elif pct < VOLUME_WARN_PCT or free < VOLUME_WARN_BYTES:
        level = "warn"
    return {"path": str(path), "free_bytes": free, "total_bytes": total, "free_pct": pct, "level": level}


def read(ctx) -> dict[str, Any]:
    state = ctx.state
    counters = probe.cpu_counters()
    previous = state.get("counters")
    if previous is None:
        # The first view should not be blank: take a short second sample once.
        time.sleep(FIRST_SAMPLE_S)
        previous, counters = counters, probe.cpu_counters()
    state["counters"] = counters
    if ctx.now - state.get("alloc_at", 0) >= ALLOC_TTL_S or "allocs" not in state:
        state["allocs"], state["alloc_at"] = placement.allocations(), ctx.now
    allocs = state["allocs"]
    exp = ctx.experiment
    run_id = exp.get("run_id")
    pid = exp.get("runner_pid") or (exp.get("owner") or {}).get("pid")
    mine: set[int] = set()
    if ctx.live:
        mine = {c for a in allocs if a["kind"] == "run" and a["name"] == run_id for c in a["cpus"]}
        if not mine and type(pid) is int:
            mine = set(placement.affinity(pid))
    try:
        console = set(os.sched_getaffinity(0))
    except OSError:
        console = set()
    holders: dict[int, list[str]] = {}
    # Runs first: a core's most specific holder is the run on it, then a reservation, then the pool.
    rank = {"run": 0, "reservation": 1, "general": 2}
    for alloc in sorted(allocs, key=lambda a: rank.get(a["kind"], 3)):
        for cpu in alloc["cpus"]:
            holders.setdefault(cpu, []).append(alloc["name"] if alloc["kind"] != "general" else "general")
    cores, busy = [], []
    for cpu in sorted(counters):
        util = None
        old, new = previous.get(cpu), counters[cpu]
        if old and new[0] > old[0] and 0 <= new[1] - old[1] <= new[0] - old[0]:
            util = round(100 * (1 - (new[1] - old[1]) / (new[0] - old[0])), 1)
            busy.append(util)
        cores.append(
            {
                "cpu": cpu,
                "util_pct": util,
                "holders": holders.get(cpu, []),
                "mine": cpu in mine,
                "console": cpu in console,
            }
        )
    notes = ["Host telemetry only: not simulation, numerical or performance evidence."]
    if not busy:
        notes.append("First sample: utilization appears after the next refresh.")
    return {
        "cores": cores,
        "allocations": [{**a, "cpus": probe_ranges(a["cpus"])} for a in allocs],
        "busy_cpus": round(sum(busy) / 100, 1) if busy else None,
        "total_cpus": len(cores),
        "experiment_cpus": sorted(mine),
        "console_cpus": sorted(console),
        "volume": volume(exp.get("run_dir")),
        "notes": notes,
    }


def probe_ranges(cpus: list[int]) -> str:
    """``[0, 1, 2, 5]`` as ``"0-2,5"``."""
    out, run = [], []
    for cpu in sorted(cpus):
        if run and cpu == run[-1] + 1:
            run.append(cpu)
            continue
        if run:
            out.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else str(run[0]))
        run = [cpu]
    if run:
        out.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else str(run[0]))
    return ",".join(out)


__all__ = ["probe_ranges", "read", "volume"]
