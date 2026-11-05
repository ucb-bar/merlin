"""What every core of the experiment works on, over time: authoring, startup, checks, builds, tests, L2, L3.

Each refresh takes the live per-thread sample :mod:`.resources` already made (CPU time since the last
sample and the CPU each thread last ran on) and gives every busy thread an *activity*:

* the process's own identity, from its argv (:func:`..terminal_probe.classify`, which names the
  agent, the brokers, Spike and the selected gSIM executable from the runner's own environment), or
  the identity of its nearest classified ancestor -- everything below the agent is authoring, below a
  self-check broker validation, below a model grade testing, unless the process is itself a
  simulator or a compiler run by the grader;
* for the runner's own threads, the innermost *running* stage record of that thread
  (``merlin.compiler_setup_timing.v1`` pid + native tid, from :mod:`.jobs`), else the author turn's
  startup while one is open (:mod:`.lifecycle`), else harness bookkeeping.

A core's time in a sample is split between those activities by thread CPU time; what the core was
busy with beyond them (``/proc/stat``) is ``other`` -- not this experiment's processes. The history is
kept per core in fixed time buckets that halve their resolution when full, so any run length costs the
same; it starts when the console first watched this run and lives as long as the console process.
``last_cpu`` is a point sample: a thread that moved between cores is charged to the core it was last
seen on. Display telemetry only.

Section::

    {"mode": "live"|"none", "categories": [{id, code, title}], "cpus": [..],
     "history": {"t0", "dt", "n", "since", "cores": [{cpu, codes: str, util: str}]},
     "now": [{cpu, util_pct, code, top: [{activity, label, pct}]}],
     "totals": {activity: core_seconds}, "notes"}

``codes`` has one character per bucket (a category ``code``; ``" "`` not observed); ``util`` one
digit per bucket (busy tenths, ``9`` for 90 % and above).
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Any

from ... import terminal_probe as probe
from ..io import clean

#: ``(id, code, title)`` in legend order.
CATEGORIES = (
    ("author", "a", "authoring: the agent and the commands it runs"),
    ("startup", "s", "author turn startup (sandbox, overlays)"),
    ("validate", "v", "checks the agent asked for (self-check, model check, ISA tools)"),
    ("build", "b", "compiling: candidate compiler calls and compile/link stages"),
    ("test", "t", "grading: verification, model grades, capsule execution"),
    ("L2", "2", "L2 simulation (Spike)"),
    ("L3", "3", "L3 simulation (gSIM)"),
    ("calibration", "k", "calibration"),
    ("harness", "h", "run bookkeeping: leases, scheduling, waiting threads"),
    ("other", "o", "busy with processes outside this experiment"),
    ("idle", ".", "idle"),
)
CODE = {cid: code for cid, code, _ in CATEGORIES}
BUCKET_S = 2.0
MAX_BUCKETS = 720
IDLE_PCT = 3.0
DETAIL_SAMPLES = 90
#: ``classify`` action -> activity for a process's own identity.
ACTIONS = {
    "agentic": "author",
    "agent_tool_host": "author",
    "host_selfcheck_broker": "validate",
    "host_selfcheck_worker": "validate",
    "host_model_check_broker": "validate",
    "host_model_check_worker": "validate",
    "host_isa_tools_broker": "validate",
    "host_cca_broker": "validate",
    "host_simjob_broker": "sim",
    "calibration_service": "calibration",
    "calibration_supervisor": "calibration",
    "verification_lease_service": "harness",
    "L3": "L3",
    "L2_or_cycle_observer": "L2",
    "compile": "build",
}
#: Job lane (:data:`.LANES`) -> activity, for a runner thread with a running stage record.
LANES = {
    "build": "build",
    "verify": "test",
    "setup": "test",
    "L2": "L2",
    "L3": "L3",
    "calibration": "calibration",
    "queue": "harness",
}
#: Python modules (``-m``) the grader runs as workers, by activity.
MODULES = {"merlin_experiments.phase1.brokers.compiler_call_channel": ("build", "compiler call")}
_HISTORIES: dict[str, History] = {}


class History:
    """Per-core activity buckets (time-weighted) plus the latest per-sample detail for inspection."""

    def __init__(self, t0: float) -> None:
        self.t0, self.dt, self.since = t0, BUCKET_S, t0
        self.cores: dict[int, list[dict[str, float]]] = {}
        self.util: dict[int, list[list[float]]] = {}  # per bucket [busy-seconds, observed-seconds]
        self.detail: dict[int, deque] = {}
        self.totals: dict[str, float] = {}
        self.previous: float | None = None
        self.procs: dict[tuple[int, int], tuple[str | None, str | None, str]] = {}
        self.env: dict[str, str] = {}

    def _index(self, t: float) -> int:
        index = int((t - self.t0) // self.dt)
        while index >= MAX_BUCKETS:
            self._halve()
            index = int((t - self.t0) // self.dt)
        return max(0, index)

    def _halve(self) -> None:
        self.dt *= 2
        for cpu, buckets in self.cores.items():
            merged = []
            for i in range(0, len(buckets), 2):
                a = dict(buckets[i])
                for k, v in (buckets[i + 1] if i + 1 < len(buckets) else {}).items():
                    a[k] = a.get(k, 0.0) + v
                merged.append(a)
            self.cores[cpu] = merged
            util = self.util[cpu]
            self.util[cpu] = [
                [
                    util[i][0] + (util[i + 1][0] if i + 1 < len(util) else 0),
                    util[i][1] + (util[i + 1][1] if i + 1 < len(util) else 0),
                ]
                for i in range(0, len(util), 2)
            ]

    def add(self, t: float, dt: float, cpu: int, shares: dict[str, float], util: float | None) -> None:
        index = self._index(t)
        buckets = self.cores.setdefault(cpu, [])
        utils = self.util.setdefault(cpu, [])
        while len(buckets) <= index:
            buckets.append({})
            utils.append([0.0, 0.0])
        for activity, seconds in shares.items():
            buckets[index][activity] = buckets[index].get(activity, 0.0) + seconds
            if activity != "idle":
                self.totals[activity] = self.totals.get(activity, 0.0) + seconds
        if util is not None:
            utils[index][0] += util / 100 * dt
            utils[index][1] += dt

    def encode(self) -> dict[str, Any]:
        cores = []
        n = max((len(b) for b in self.cores.values()), default=0)
        for cpu in sorted(self.cores):
            codes, digits = [], []
            for bucket, (busy, seen) in zip(self.cores[cpu], self.util[cpu]):
                if not bucket and not seen:
                    codes.append(" ")
                    digits.append(" ")
                    continue
                activity = max(bucket.items(), key=lambda kv: kv[1])[0] if bucket else "idle"
                codes.append(CODE.get(activity, "o"))
                digits.append(str(min(9, int(10 * busy / seen))) if seen else " ")
            pad = n - len(codes)
            cores.append({"cpu": cpu, "codes": "".join(codes) + " " * pad, "util": "".join(digits) + " " * pad})
        return {"t0": self.t0, "dt": self.dt, "n": n, "since": self.since, "cores": cores}


def _argv(pid: int) -> list[str]:
    raw = probe.read(Path("/proc") / str(pid) / "cmdline", 65536) or b""
    return [x.decode(errors="replace") for x in raw.split(b"\0") if x]


def _environment(pid: int | None) -> dict[str, str]:
    """The runner's simulator selections (only the keys :func:`probe.classify` matches)."""
    if pid is None:
        return {}
    raw = probe.read(Path("/proc") / str(pid) / "environ", 1 << 20) or b""
    env = {}
    for item in raw.split(b"\0"):
        key, sep, value = item.decode(errors="replace").partition("=")
        if sep and (key.startswith("MERLIN_GSIM_EMU_") or key.endswith("_GSIM_EMU") or key == "MERLIN_SPIKE"):
            env[key] = value
    return env


def _name(argv: list[str]) -> str:
    """A short name for a process: a Python module or script by its own name, else the executable."""
    if not argv:
        return "process"
    exe = Path(argv[0]).name
    if exe.startswith("python"):
        module = _flag(argv, "-m")
        if module:
            return clean(module.rsplit(".", 1)[-1], 32)
        script = next((a for a in argv[1:] if not a.startswith("-")), None)
        if script and script.endswith(".py"):
            return clean(Path(script).name, 32)
    return clean(exe, 32)


def _flag(argv: list[str], name: str) -> str | None:
    for i, value in enumerate(argv[:-1]):
        if value == name:
            return argv[i + 1]
        if value.startswith(name + "="):
            return value.split("=", 1)[1]
    return None


def _own(argv: list[str], env: dict[str, str]) -> tuple[str | None, str | None]:
    """A process's own activity and label from its argv, or ``(None, None)`` when it inherits."""
    if not argv:
        return None, None
    action, _ = probe.classify(argv, {"environment": env})
    exe = Path(argv[0]).name
    module = _flag(argv, "-m")
    if module in MODULES:
        return MODULES[module]
    if "--model-grade" in argv:
        return "test", "model grade"
    tiers, sim = _flag(argv, "--tiers"), _flag(argv, "--sim")
    if action == "native_evaluator":
        return "runner", "runner"
    if action == "host_simjob_broker" or ((tiers or sim) and action not in ACTIONS):
        if "L3" in str(tiers):
            return "L3", f"{sim or 'sim'} L3"
        if "L2" in str(tiers):
            return "L2", f"{sim or 'sim'} L2"
        return ("sim", "simulation broker") if action == "host_simjob_broker" else (None, None)
    activity = ACTIONS.get(action)
    if activity is None:
        return None, None
    label = {"agentic": "agent", "L3": "gSIM", "L2_or_cycle_observer": "Spike"}.get(action, exe)
    return activity, clean(label, 32)


def _identity(history: History, pid: int) -> tuple[str | None, str | None, str, int | None]:
    """``(own activity, own label, executable, parent pid)`` of one live process, cached per lifetime."""
    stat = probe.task_stat(pid)
    key = (pid, stat["start_ticks"] if stat else 0)
    cached = history.procs.get(key)
    if cached is None:
        argv = _argv(pid)
        activity, label = _own(argv, history.env)
        cached = (activity, label, _name(argv))
        if len(history.procs) > 8192:
            history.procs.clear()
        history.procs[key] = cached
    return (*cached, stat["ppid"] if stat else None)


def _role(history: History, pid: int, runner: int | None, memo: dict) -> tuple[str | None, str]:
    """``(activity, label)`` of one process: its own identity, or the nearest classified ancestor's."""
    if pid in memo:
        return memo[pid]
    own, own_label, exe, parent = _identity(history, pid)
    result: tuple[str | None, str] = (own, own_label or exe)
    if own is None or own in ("L2", "L3", "build"):
        node = parent
        for _ in range(32):
            if node in (None, 0, 1):
                break
            activity, label, _, up = _identity(history, node)
            if activity == "author":
                result = ("author", f"agent › {exe}")  # The agent's own builds and tests are authoring.
                break
            if activity == "runner":
                result = result if own else ("runner_child", exe)
                break
            if activity is not None:
                if own is None:
                    result = ("L3", f"{exe} (sim broker)") if activity == "sim" else (activity, f"{label} › {exe}")
                break
            if node == runner:
                break
            node = up
    if result[0] == "sim":
        result = ("L3", "simulation broker")
    memo[pid] = result
    return result


def _thread_activity(history, thread, runner, stages, startup, memo) -> tuple[str, str]:
    pid, tid = thread.get("pid"), thread.get("tid")
    activity, label = _role(history, pid, runner, memo)
    if activity == "runner":
        stage = stages.get((pid, tid))
        if stage is not None:
            return LANES.get(stage["lane"], "test"), clean(stage["stage"], 40)
        if startup:
            return "startup", startup
        return "harness", "runner thread"
    if activity == "runner_child":
        return ("startup", f"{startup} › {label}") if startup else ("test", label)
    if activity is None:
        return ("startup", f"{startup} › {label}") if startup else ("harness", label)
    return activity, label


def read(ctx) -> dict[str, Any]:
    from . import jobs as jobs_source
    from . import resources as resources_source

    key = str(ctx.experiment.get("id") or ctx.experiment.get("run_dir"))
    categories = [{"id": cid, "code": code, "title": title} for cid, code, title in CATEGORIES]
    sample = resources_source.SAMPLES.get(key)
    history = _HISTORIES.get(key)
    if sample is None or not ctx.live:
        notes = ["Per-core activity is observed live only; this run is not running."] if not ctx.live else []
        if history is not None:
            notes.append("History below is what this console observed while the run was live.")
        return {
            "mode": "none" if history is None else "history",
            "categories": categories,
            "cpus": [],
            "history": history.encode() if history else None,
            "now": [],
            "totals": {k: round(v, 1) for k, v in (history.totals if history else {}).items()},
            "notes": notes or ["No live sample yet."],
        }
    if history is None:
        history = _HISTORIES[key] = History(sample["at"])
    if history.previous is not None and sample["at"] <= history.previous:
        return ctx.state.get("section") or {}
    dt = sample["at"] - history.previous if history.previous is not None else None
    history.previous = sample["at"]
    life = ctx.shared.get("lifecycle") or {}
    startup = None
    if life.get("phase") == "startup":
        turns = life.get("turns") or []
        open_stage = (turns[-1].get("open_stage") or {}) if turns else {}
        startup = clean(open_stage.get("stage"), 32) or "startup"
    stages = jobs_source.THREAD_STAGES.get(key, {})
    runner = sample.get("runner")
    if ctx.state.get("env_pid") != runner:
        history.env, history.procs = _environment(runner), {}
        ctx.state["env_pid"] = runner
    memo: dict[int, tuple] = {}
    now_rows, cpus = [], []
    for core in sample["cores"]:
        cpu = core.get("cpu")
        if type(cpu) is not int:
            continue
        cpus.append(cpu)
        util = core.get("measured_utilization_pct")
        shares: dict[str, float] = {}
        labels: dict[tuple[str, str], list] = {}
        for thread in core.get("threads") or []:
            seconds = thread.get("cpu_time_delta_s")
            if not seconds:
                continue
            activity, label = _thread_activity(history, thread, runner, stages, startup, memo)
            shares[activity] = shares.get(activity, 0.0) + seconds
            entry = labels.setdefault((activity, label), [0.0, thread.get("pid"), thread.get("tid")])
            entry[0] += seconds
        top = []
        if dt:
            attributed = sum(shares.values())
            if isinstance(util, int | float):
                other = max(0.0, util / 100 * dt - attributed)
                if other > 0.02 * dt:
                    shares["other"] = other
                    labels[("other", "outside this experiment")] = [other, None, None]
                idle = max(0.0, dt - attributed - other)
                if idle > 0:
                    shares["idle"] = idle
            if not isinstance(util, int | float) or util < IDLE_PCT and attributed < IDLE_PCT / 100 * dt:
                shares = {"idle": dt} if isinstance(util, int | float) else {}
            history.add(sample["at"] - dt / 2, dt, cpu, shares, util if isinstance(util, int | float) else None)
            top = [
                {"activity": a, "label": lab, "pct": round(100 * v[0] / dt, 1), "pid": v[1], "tid": v[2]}
                for (a, lab), v in sorted(labels.items(), key=lambda kv: -kv[1][0])[:5]
            ]
            history.detail.setdefault(cpu, deque(maxlen=DETAIL_SAMPLES)).append(
                (sample["at"], util, [dict(t) for t in top])
            )
        busy = [s for a, s in shares.items() if a != "idle"]
        code = (
            CODE["idle"]
            if not busy
            else CODE.get(max((kv for kv in shares.items() if kv[0] != "idle"), key=lambda kv: kv[1])[0], "o")
        )
        now_rows.append(
            {
                "cpu": cpu,
                "util_pct": round(util, 1) if isinstance(util, int | float) else None,
                "code": code if dt else " ",
                "top": [{k: v for k, v in t.items() if k in ("activity", "label", "pct")} for t in top[:3]],
            }
        )
    notes = [
        "Thread CPU time charged to the core it was last seen on (a point sample); other = core busy "
        "time not spent in this experiment's threads.",
        f"Observed since {time.strftime('%H:%M:%S', time.gmtime(history.since))} UTC (console start).",
    ]
    section = {
        "mode": "live",
        "categories": categories,
        "cpus": cpus,
        "history": history.encode(),
        "now": now_rows,
        "totals": {k: round(v, 1) for k, v in history.totals.items()},
        "notes": notes,
    }
    ctx.state["section"] = section
    return section


def inspect(ctx, key: str) -> dict[str, Any]:
    exp = str(ctx.experiment.get("id") or ctx.experiment.get("run_dir"))
    history = _HISTORIES.get(exp)
    if history is None or not key.isdigit() or int(key) not in history.detail:
        why = (
            "this console has not sampled the core yet (the first sample comes after one refresh)"
            if ctx.live
            else "the run is not running: per-core activity is observed live only, by a console watching it"
        )
        return {"title": f"core {clean(key, 8)}", "sections": [{"title": "no samples", "lines": [why]}], "paths": []}
    cpu = int(key)
    lines = []
    for at, util, top in reversed(history.detail[cpu]):
        what = ", ".join(
            f"{t['activity']} {t['label']} {t['pct']:.0f}%"
            + (f" (pid {t['pid']})" if ctx.operator_private and t["pid"] else "")
            for t in top
        )
        busy = f"{util:3.0f}%" if isinstance(util, int | float) else "  ?"
        lines.append(f"{time.strftime('%H:%M:%S', time.gmtime(at))} {busy}  {what or 'idle'}")
    buckets = history.cores.get(cpu, [])
    totals: dict[str, float] = {}
    for bucket in buckets:
        for activity, seconds in bucket.items():
            totals[activity] = totals.get(activity, 0.0) + seconds
    spent = [f"{a:<12}{s / 60:6.1f} min" for a, s in sorted(totals.items(), key=lambda kv: -kv[1])]
    return {
        "title": f"core {cpu} activity",
        "sections": [
            {"title": "time on this core since the console started watching", "lines": spent or ["nothing yet"]},
            {"title": "recent samples (newest first)", "lines": lines},
        ],
        "paths": [],
    }


__all__ = ["CATEGORIES", "History", "inspect", "read"]
