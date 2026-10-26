"""Host resources of one experiment: per-core activity while it runs, process-tree history always.

Three modes:

* ``live`` with a managed owner -- :class:`..terminal_probe.Sampler` bound to *this* launch's owner
  record (core utilization from ``/proc/stat`` deltas, thread placement, CPU leases and waiters);
* ``live`` without an owner -- the runner process tree (descendants of ``runner_pid``), its CPU
  affinity and the same ``/proc/stat`` deltas;
* ``recorded`` -- no live process; only the rounds' ``resource_samples.jsonl`` history.

The process-tree series (CPU %, RSS, host load) comes from ``rounds/round_NN.turnNN.resource_samples.jsonl``
in every mode. All of it is display telemetry, never simulation or performance evidence; thread
placement (``last_cpu``) is a point sample and the first live sample has no utilization yet.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from ... import terminal_probe as probe
from ... import terminal_views
from ..io import clean, epoch, is_dir, num, obj, rows, scan_dir
from . import downsample

TELEMETRY_NOTE = "Host telemetry only: not simulation, numerical or performance evidence."
_HISTORY = 4096
COMPANIONS_TTL_S = 15.0
#: ``experiment id -> {"at", "cores": [{cpu, measured_utilization_pct, threads}], "runner"}``: the latest
#: live sample, so :mod:`.activity` attributes core time without scanning ``/proc`` a second time.
SAMPLES: dict[str, dict[str, Any]] = {}


class TreeSampler:
    """``/proc`` sampling of one runner's descendant processes (no owner record, no cgroup trust)."""

    def __init__(
        self, pid: int, proc_root: Path = Path("/proc"), max_processes: int = 256, max_threads: int = 2048
    ) -> None:
        self.pid, self.proc_root = pid, Path(proc_root)
        self.max_processes, self.max_threads = max_processes, max_threads
        self.previous_cpus: dict[int, tuple[int, int]] = {}
        self.previous_threads: dict[tuple, int] = {}
        self.previous_time: float | None = None
        #: Other processes of the same experiment outside the runner's tree (its reservation's services).
        self.companions: list[int] = []
        self.previous_io: list[int] | None = None

    def descendants(self) -> list[int]:
        """The runner and its descendants, then the companion roots (:attr:`companions`) and theirs."""
        parents: dict[int, list[int]] = {}
        for entry in scan_dir(self.proc_root, 65536):
            if not entry.name.isdigit():
                continue
            row = probe.task_stat(int(entry.name), proc_root=self.proc_root)
            if row:
                parents.setdefault(row["ppid"], []).append(int(entry.name))
        out, seen, queue = [], set(), [self.pid, *reversed(self.companions)]
        while queue and len(out) < self.max_processes:
            pid = queue.pop()
            if pid in seen:
                continue
            seen.add(pid)
            out.append(pid)
            queue.extend(parents.get(pid, []))
        return out

    def affinity(self) -> list[int]:
        raw = probe.read(self.proc_root / str(self.pid) / "status", 16384) or b""
        for line in raw.decode(errors="replace").splitlines():
            if line.startswith("Cpus_allowed_list:"):
                return probe.cpus(line.split(":", 1)[1].strip())
        return []

    def sample(self) -> dict[str, Any]:
        started = time.monotonic()
        hz = os.sysconf("SC_CLK_TCK")
        elapsed = started - self.previous_time if self.previous_time is not None else None
        threads, next_threads, processes = [], {}, 0
        io = [0, 0]
        for pid in self.descendants():
            raw = probe.read(self.proc_root / str(pid) / "io", 4096) or b""
            for line in raw.split(b"\n"):
                key, _, value = line.partition(b":")
                if key in (b"rchar", b"wchar") and value.strip().isdigit():
                    io[key == b"wchar"] += int(value)
            raw = probe.read(self.proc_root / str(pid) / "cmdline", 65536)
            argv = [x.decode(errors="replace") for x in (raw or b"").split(b"\0") if x]
            action, _ = probe.classify(argv, {})
            processes += 1
            try:
                tids = sorted(int(p.name) for p in (self.proc_root / str(pid) / "task").iterdir() if p.name.isdigit())
            except OSError:
                continue
            for tid in tids:
                if len(threads) >= self.max_threads:
                    break
                row = probe.task_stat(pid, tid, self.proc_root)
                if row is None:
                    continue
                key = (pid, tid, row["start_ticks"])
                old = self.previous_threads.get(key)
                row["cpu_time_delta_s"] = max(0, row["cpu_ticks"] - old) / hz if old is not None and elapsed else None
                row["action"] = action
                next_threads[key] = row["cpu_ticks"]
                threads.append(row)
        counters = probe.cpu_counters(self.proc_root)
        cores = []
        for cpu in self.affinity():
            prior, now = self.previous_cpus.get(cpu), counters.get(cpu)
            util = None
            if prior and now and now[0] > prior[0] and 0 <= now[1] - prior[1] <= now[0] - prior[0]:
                util = 100 * (1 - (now[1] - prior[1]) / (now[0] - prior[0]))
            cores.append(
                {"cpu": cpu, "measured_utilization_pct": util, "threads": [t for t in threads if t["last_cpu"] == cpu]}
            )
        rates = None
        if self.previous_io is not None and elapsed:
            # Bytes through read()/write() (page cache included) of the processes alive in both samples.
            rates = {
                k: max(0.0, (io[i] - self.previous_io[i]) / elapsed) for i, k in enumerate(("read_bps", "write_bps"))
            }
        self.previous_io = io
        self.previous_time, self.previous_cpus, self.previous_threads = started, counters, next_threads
        return {"cores": cores, "jobs": [], "processes": processes, "threads": len(threads), "io": rates}


def _companions(cpus: list[int]) -> list[int]:
    """Processes of reservations (cgroups confining CPUs) that hold exactly the runner's CPUs or fewer of them.

    A run's own services (a calibration owner, say) are often launched beside the runner rather than
    under it, inside the reservation that confines the run; they are this experiment's work."""
    from .. import placement

    mine = set(cpus)
    if not mine:
        return []
    out: list[int] = []
    for alloc in placement.allocations():
        if alloc["kind"] == "reservation" and alloc["processes"] and set(alloc["cpus"]) <= mine:
            out.extend(placement.group_pids(alloc["name"], limit=256))
    return sorted(set(out))[:512]


def _core_rows(cores: list[dict], private: bool) -> list[dict[str, Any]]:
    out = []
    for core in cores:
        kind = terminal_views.activity(core) if core.get("measured_utilization_pct") is not None else "unknown"
        task = terminal_views.dominant(core)
        label = task.get("action") or ("idle" if kind == "idle" else "unattributed")
        if private and task.get("capsule"):
            label = f"{label} / {task['capsule']}"
        out.append(
            {
                "cpu": core.get("cpu"),
                "util_pct": num(core.get("measured_utilization_pct")),
                "kind": kind,
                "label": clean(label, 80),
            }
        )
    return out


#: Seconds between scans of ``/proc/locks`` for held gSIM slots (it can hold 10^5 lines).
SLOT_SCAN_S = 5.0


def _slot_dir(ctx) -> Path | None:
    """The gSIM slot directory: the launch's ``gsim-slots``, else the runner's ``MERLIN_GSIM_SLOT_ROOT``."""
    launch = ctx.experiment.get("launch_dir")
    if launch and is_dir(Path(launch) / "gsim-slots"):
        return Path(launch) / "gsim-slots"
    pid = ctx.experiment.get("runner_pid")
    if type(pid) is not int:
        return None
    cached = ctx.state.get("slot_dir")
    if cached and cached[0] == pid:
        return cached[1]
    raw = probe.read(Path("/proc") / str(pid) / "environ", 1 << 20) or b""
    found = None
    for item in raw.split(b"\0"):
        key, sep, value = item.decode(errors="replace").partition("=")
        if sep and key == "MERLIN_GSIM_SLOT_ROOT" and is_dir(value):
            found = Path(value)
    ctx.state["slot_dir"] = (pid, found)
    return found


def _held_locks(paths: list[Path]) -> int:
    """How many of ``paths`` some process holds a lock on."""
    keys = {}
    for path in paths:
        try:
            st = path.stat()
        except OSError:
            continue
        keys[(st.st_dev, st.st_ino)] = path
    return len(probe.locks(wanted=set(keys)))


def _gsim_active(ctx, live: bool) -> int | None:
    """gSIM simulations holding a slot now (live), else the budget's recorded active admissions.

    The budget's ``active`` rows are admissions of the runner, whose pid outlives each simulation, so
    a live count never comes from them: only a held slot lock is a simulation running."""
    experiment = ctx.experiment
    slots = _slot_dir(ctx) if live else None
    if slots is not None:
        cached = ctx.state.get("gsim_held")
        if cached and time.monotonic() - cached[0] < SLOT_SCAN_S:
            return cached[1]
        paths = [Path(e.path) for e in scan_dir(slots, 512) if e.name.startswith("slot_") and e.name.endswith(".lock")]
        held = _held_locks(paths)
        ctx.state["gsim_held"] = (time.monotonic(), held)
        return held
    if live:
        return None
    budget = ctx.reader.document(Path(experiment["run_dir"]) / "joint_gsim_correctness_budget.json", limit=16 << 20)
    if budget is None:
        return None
    return len(rows(budget.get("active")))


def _series(ctx) -> tuple[dict[str, list], dict[str, Any]]:
    """Process-tree history from every round's resource samples, read incrementally."""
    state = ctx.state.setdefault("samples", {})
    rounds = Path(ctx.experiment["run_dir"]) / "rounds"
    names = sorted(e.name for e in scan_dir(rounds, 8192) if e.name.endswith(".resource_samples.jsonl"))
    for name in names[-64:]:
        bucket = state.setdefault(name, [])

        def consume(row, bucket=bucket):
            t = epoch(row.get("sampled_at"))
            if t is None:
                return
            cpu = (num(row.get("user_cpu_s")) or 0) + (num(row.get("system_cpu_s")) or 0)
            bucket.append(
                (
                    t,
                    cpu,
                    num(row.get("rss_bytes")),
                    num(row.get("host_load_1m")),
                    num(row.get("processes")),
                    num(row.get("threads")),
                )
            )
            del bucket[:-_HISTORY]

        ctx.reader.poll(rounds / name, consume, on_reset=bucket.clear)
    t, cpu_pct, rss, load = [], [], [], []
    latest: dict[str, Any] = {}
    for name in sorted(state):
        bucket = state[name]
        for index, (stamp, cpu, rss_b, load1, procs, threads) in enumerate(bucket):
            pct = None
            if index:
                dt = stamp - bucket[index - 1][0]
                if dt > 0:
                    pct = max(0.0, 100 * (cpu - bucket[index - 1][1]) / dt)
            t.append(stamp)
            cpu_pct.append(pct)
            rss.append(rss_b)
            load.append(load1)
            latest = {
                "cpu_pct": pct,
                "rss_bytes": rss_b,
                "threads": threads,
                "processes": procs,
                "load1": load1,
                "sampled_at": stamp,
            }
    points = downsample(list(zip(t, cpu_pct, rss, load)))
    series = {
        "t": [p[0] for p in points],
        "cpu_pct": [p[1] for p in points],
        "rss_bytes": [p[2] for p in points],
        "load1": [p[3] for p in points],
    }
    return series, latest


def read(ctx) -> dict[str, Any]:
    experiment = ctx.experiment
    owner = obj(experiment.get("owner"))
    notes = [TELEMETRY_NOTE]
    series, latest = _series(ctx)
    live_owner = ctx.live and owner.get("alive") and owner.get("path") and experiment.get("task")
    runner = experiment.get("runner_pid")
    sample: dict[str, Any] | None = None
    if live_owner:
        sampler = ctx.state.get("sampler")
        if sampler is None or ctx.state.get("sampler_key") != owner["path"]:
            sampler = probe.Sampler(Path(experiment["task"]), ctx.operator_private, owner_path=owner["path"])
            ctx.state.update(sampler=sampler, sampler_key=owner["path"])
        sample = sampler.sample()
        if sample.get("status") == "selection_unavailable":
            notes.append("Owner selection unreadable or its digest changed; core view unavailable.")
            sample = None
    elif ctx.live and type(runner) is int:
        sampler = ctx.state.get("tree")
        if sampler is None or sampler.pid != runner:
            sampler = TreeSampler(runner)
            ctx.state["tree"] = sampler
        if ctx.now - ctx.state.get("companions_at", 0) >= COMPANIONS_TTL_S:
            sampler.companions = _companions(sampler.affinity())
            ctx.state["companions_at"] = ctx.now
        sample = sampler.sample()
        notes.append(f"No managed owner: cores are the runner pid {runner} affinity; no lease or waiter records.")
    key = str(experiment.get("id") or experiment.get("run_dir"))
    if sample is not None:
        SAMPLES[key] = {
            "at": ctx.now,
            "cores": sample.get("cores") or [],
            "runner": runner if type(runner) is int else owner.get("pid"),
        }
    else:
        SAMPLES.pop(key, None)
    if sample is not None:
        cores = _core_rows(sample.get("cores") or [], ctx.operator_private)
        known = [c["util_pct"] for c in cores if c["util_pct"] is not None]
        jobs = sample.get("jobs") or []
        queue = {
            "waiters": sum(1 for j in jobs if j.get("kind") == "cpu_waiter"),
            "leases": sum(1 for j in jobs if j.get("kind") == "cpu_lease"),
            "gsim_active": _gsim_active(ctx, True),
        }
        if not known:
            notes.append("First live sample: utilization appears after the next refresh.")
        process = {
            "io": sample.get("io"),
            "cpu_pct": latest.get("cpu_pct"),
            "rss_bytes": latest.get("rss_bytes"),
            "threads": latest.get("threads") if latest else sample.get("threads"),
            "processes": latest.get("processes") if latest else sample.get("processes"),
            "load1": latest.get("load1") if latest else os.getloadavg()[0],
        }
        return {
            "mode": "live",
            "cores": cores,
            "busy_cpus": sum(known) / 100 if known else None,
            "total_cpus": len(cores),
            "queue": queue,
            "process": process,
            "series": series,
            "notes": notes,
        }
    queue = {"waiters": None, "leases": None, "gsim_active": None}
    recorded = _gsim_active(ctx, False)
    if recorded:
        notes.append(f"{recorded} gSIM job(s) were recorded active when the budget record last changed.")
    process = {k: latest.get(k) for k in ("cpu_pct", "rss_bytes", "threads", "processes", "load1")}
    if latest:
        notes.append("Not running: values are the last recorded process-tree sample.")
    return {
        "mode": "recorded" if series["t"] else "none",
        "cores": [],
        "busy_cpus": None,
        "total_cpus": None,
        "queue": queue,
        "process": process,
        "series": series,
        "notes": notes,
    }


__all__ = ["SAMPLES", "TreeSampler", "read"]
