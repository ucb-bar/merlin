"""Jobs and queues of one experiment: the Gantt spans, stage-duration statistics and simulation queue.

Two record families feed it:

* ``merlin.compiler_setup_timing.v1`` timelines (diagnostic-only host wall time, nested stages that
  overlap) under ``_qa_work/profile_package_execution``, ``_qa_work/fruns_*/host-stage-timing``,
  ``joint_final_coverage/*/host-stage-timing`` and the workspace's ``.qa_channel/.resp_*.profile``;
* simulation broker files in ``.qa_channel``: ``simreq_<job>.json`` (request), ``simtiming_<job>.json``
  (``simjob_resource_timing_v1``: queue wait and worker runtime), ``simrun_<job>`` / ``simdone_<job>``
  (running / finished markers).

Timeline files are indexed incrementally by stat identity across refreshes (newest directories
first, a bounded number of parses per refresh), so a run with thousands of records costs a few
milliseconds once caught up. Spans are the *leaf* stages (a stage with no recorded children), so
nested intervals are never drawn twice. The lane of a stage is chosen by :data:`LANE_RULES`, the first
rule whose token occurs in the lower-cased stage name; anything unmatched is ``setup``. Simulation
jobs are drawn in the lane of their requested engine tier (``L3`` for a gSIM/L3 request, ``L2`` for
Spike/L2) with their queue wait in ``queue``. Capsule names appear only in operator-private mode.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ..io import clean, epoch, is_dir, num, obj, scan_dir
from . import LANES

#: ``(token, lane)``: first token found in the lower-cased stage name decides the lane.
LANE_RULES = (
    # The grader's own calculations (golden, L0 reference, L1 simulate, compare, hashing) and the wait
    # for the CPU lease each one takes (``grade_cpu.*`` from the capsule runner).
    ("grade_cpu.lease", "queue"),
    ("grade_cpu.", "verify"),
    ("calibr", "calibration"),
    (".l3", "L3"),
    ("gsim", "L3"),
    (".l2", "L2"),
    ("spike", "L2"),
    ("wait", "queue"),
    ("verify", "verify"),
    ("invocation_snapshot", "verify"),
    ("package_policy", "verify"),
    ("snapshot_entry", "verify"),
    ("emission", "build"),
    ("compile", "build"),
    ("link", "build"),
    ("emit", "build"),
)
TIMING_SCHEMA = "merlin.compiler_setup_timing.v1"
SIMTIMING_SCHEMA = "simjob_resource_timing_v1"
MAX_RECORDS = 5000
MAX_SPANS = 1500
PARSES_PER_REFRESH = 600
MIN_SPAN_S = 0.25
#: ``experiment id -> {(pid, native tid): {stage, lane, capsule, start}}``: the innermost running stage of
#: every grader thread with a running record, for :mod:`.activity` to name what a busy thread does.
THREAD_STAGES: dict[str, dict[tuple[int, int], dict[str, Any]]] = {}


def _ns(value: Any) -> float | None:
    """A recorded ``*_ns`` field as epoch seconds (the field name, not the magnitude, says nanoseconds)."""
    value = num(value)
    return value / 1e9 if value is not None and value > 0 else None


def lane_of(stage: str | None, tier: str | None = None) -> str:
    name = str(stage or "").lower()
    for token, lane in LANE_RULES:
        if token in name:
            return lane
    return tier if tier in ("L2", "L3") else "setup"


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def _timing_dirs(run_dir: Path, workspace: Path | None) -> list[Path]:
    qa = run_dir / "_qa_work"
    dirs = [qa / "profile_package_execution" / "setup-timing"]
    fruns = sorted((e.name for e in scan_dir(qa, 4096) if e.name.startswith("fruns_")), reverse=True)
    dirs += [qa / name / "host-stage-timing" / "setup-timing" for name in fruns[:256]]
    dirs += [
        Path(e.path) / "host-stage-timing" / "setup-timing" for e in scan_dir(run_dir / "joint_final_coverage", 64)
    ]
    if workspace is not None:
        channel = workspace / ".qa_channel"
        profiles = sorted(
            (e.name for e in scan_dir(channel, 16384) if e.name.startswith(".resp_") and e.name.endswith(".profile")),
            reverse=True,
        )
        dirs += [channel / name / "setup-timing" for name in profiles[:128]]
    return [d for d in dirs if is_dir(d)]


def _summarize(doc: dict, path: Path, private: bool) -> dict[str, Any] | None:
    if doc.get("schema") != TIMING_SCHEMA or not isinstance(doc.get("stages"), list):
        return None
    stages = [s for s in doc["stages"][:256] if isinstance(s, dict)]
    parents = {s.get("parent_stage_id") for s in stages if s.get("parent_stage_id") is not None}
    pid = doc.get("pid") if type(doc.get("pid")) is int else None
    out = []
    for stage in stages:
        start = _ns(stage.get("started_ns"))
        if start is None:
            continue
        end = _ns(stage.get("finished_ns"))
        elapsed = num(stage.get("elapsed_s"))
        out.append(
            {
                "stage": clean(stage.get("stage"), 64),
                "start": start,
                "end": end,
                "elapsed": elapsed if elapsed is not None else (end - start if end else None),
                "status": clean(stage.get("status"), 20) or "unknown",
                "leaf": stage.get("stage_id") not in parents,
                "tid": next((t for t in (stage.get("native_tid"), doc.get("native_tid")) if type(t) is int), None),
            }
        )
    if not out:
        return None
    return {
        "capsule": clean(doc.get("capsule"), 96) if private else None,
        "tier": doc.get("tier"),
        "operation": clean(doc.get("operation"), 48),
        "path": str(path),
        "pid": pid,
        "stages": out,
        "first": min(s["start"] for s in out),
    }


def _index(ctx, dirs: list[Path]) -> dict[str, dict]:
    state = ctx.state
    index: dict[str, tuple] = state.setdefault("timing", {})
    listings: dict[str, tuple] = state.setdefault("listings", {})
    parses = 0
    for directory in dirs:
        try:
            stamp = os.stat(directory).st_mtime_ns
        except OSError:
            continue
        cached = listings.get(str(directory))
        if not cached or cached[0] != stamp:
            names = [e.name for e in scan_dir(directory, 20000) if e.name.endswith(".json")]
            cached = (stamp, names)
            listings[str(directory)] = cached
        for name in cached[1]:
            path = directory / name
            known = index.get(str(path))
            # A finished record is immutable; a running one is rewritten, so re-stat only those.
            if known and not known[2]:
                continue
            ident = ctx.reader.identity(path)
            if ident is None or (known and known[0] == ident):
                continue
            if parses >= PARSES_PER_REFRESH:
                ctx.reader.notices.add("job timelines still being indexed; the Gantt fills over the next refreshes")
                break
            parses += 1
            summary = _summarize(ctx.reader.document(path, limit=1 << 20) or {}, path, ctx.operator_private)
            running = bool(summary and any(s["end"] is None and s["status"] == "running" for s in summary["stages"]))
            index[str(path)] = (ident, summary, running)
            state["version"] = state.get("version", 0) + 1
    if len(index) > MAX_RECORDS:
        keep = sorted(index.items(), key=lambda kv: (kv[1][1] or {}).get("first") or 0)[-MAX_RECORDS:]
        state["timing"] = index = dict(keep)
    return {k: v[1] for k, v in index.items() if v[1]}


def _sim_jobs(ctx, workspace: Path | None) -> list[dict[str, Any]]:
    if workspace is None:
        return []
    channel = workspace / ".qa_channel"
    try:
        stamp = os.stat(channel).st_mtime_ns
    except OSError:
        return []
    cached = ctx.state.get("channel")
    if not cached or cached[0] != stamp:
        cached = (stamp, [e.name for e in scan_dir(channel, 32768) if e.name.startswith("sim")])
        ctx.state["channel"] = cached
    names = set(cached[1])
    jobs = []
    for name in sorted(names):
        if not (name.startswith("simreq_") and name.endswith(".json")):
            continue
        job = name[len("simreq_") : -len(".json")]
        request = obj(ctx.reader.document(channel / name, limit=1 << 20))
        timing = (
            obj(ctx.reader.document(channel / f"simtiming_{job}.json", limit=1 << 20))
            if f"simtiming_{job}.json" in names
            else {}
        )
        if timing and timing.get("schema") != SIMTIMING_SCHEMA:
            timing = {}
        done, running = f"simdone_{job}" in names, f"simrun_{job}" in names
        code = timing.get("returncode")
        # simdone/simerr are the broker's own completion markers; the worker returncode is not a verdict.
        if f"simerr_{job}" in names:
            status = "error"
        elif done:
            verdict = (
                ctx.reader.document(channel / f"simresp_{job}.json", limit=8 << 20)
                if f"simresp_{job}.json" in names
                else None
            )
            # ``all_pass`` covers the whole suite even for a subset request; the request's own
            # verdict is whether every capsule it returned passed.
            verdict = obj(verdict)
            returned, passed = num(verdict.get("n_result_rows")), num(verdict.get("n_passed"))
            if returned and passed is not None:
                status = "pass" if passed >= returned else "fail"
            else:
                status = {True: "pass", False: "fail"}.get(verdict.get("all_pass"), "done")
        else:
            status = "running" if running else "queued"
        tiers, sim = str(request.get("tiers") or ""), str(request.get("sim") or "")
        lane = (
            "L3"
            if "L3" in tiers or "gsim" in sim.lower()
            else "L2"
            if "L2" in tiers or "spike" in sim.lower()
            else "L3"
        )
        capsules = request.get("capsules")
        jobs.append(
            {
                "job": job,
                "lane": lane,
                "sim": clean(sim, 16),
                "capsule": clean(capsules if isinstance(capsules, str) else ",".join(map(str, capsules or [])), 96)
                if ctx.operator_private
                else None,
                "submitted": epoch(request.get("submitted_at")),
                "started": epoch(timing.get("started_at_unix")),
                "finished": epoch(timing.get("finished_at_unix")),
                "queue_wait_s": num(timing.get("queue_wait_s")),
                "runtime_s": num(timing.get("worker_runtime_s")),
                "returncode": code,
                "status": status,
                "source": str(channel / name),
            }
        )
    return jobs


def _notes_calibration(ctx, notes: list[str]) -> None:
    launch = ctx.experiment.get("launch_dir")
    if not launch:
        return
    for name in ("calibration", "calibration-owner"):
        status = ctx.reader.document(Path(launch) / name / "status.json", limit=1 << 20)
        if status and status.get("status"):
            notes.append(f"{name} status: {clean(status['status'], 60)}")


def read(ctx) -> dict[str, Any]:
    run_dir = Path(ctx.experiment["run_dir"])
    workspace = Path(ctx.experiment["workspace"]) if ctx.experiment.get("workspace") else None
    notes = ["Setup timelines are diagnostic host wall time including waits; nested stages overlap."]
    records = _index(ctx, _timing_dirs(run_dir, workspace))
    # Unchanged records give the same section (the same object, so the server skips re-encoding it);
    # a live run with work in flight is rebuilt for its ages.
    channel = workspace / ".qa_channel" if workspace else None
    try:
        channel_stamp = os.stat(channel).st_mtime_ns if channel else None
    except OSError:
        channel_stamp = None
    key = (ctx.state.get("version", 0), channel_stamp, ctx.live, ctx.operator_private)
    memo = ctx.state.get("memo")
    if memo and memo[0] == key and not (ctx.live and memo[1]["running"]):
        return memo[1]
    spans, durations, running = [], {}, []
    threads: dict[tuple[int, int], dict[str, Any]] = {}
    for summary in records.values():
        for stage in summary["stages"]:
            durations.setdefault(stage["stage"], []).append(stage)
            is_running = stage["end"] is None and stage["status"] == "running"
            if is_running and summary["pid"] is not None and stage["tid"] is not None:
                key = (summary["pid"], stage["tid"])
                if key not in threads or stage["start"] >= threads[key]["start"]:
                    threads[key] = {
                        "stage": stage["stage"],
                        "lane": lane_of(stage["stage"], summary["tier"]),
                        "capsule": summary["capsule"],
                        "start": stage["start"],
                    }
            if is_running:
                running.append(
                    {
                        "stage": stage["stage"],
                        "capsule": summary["capsule"],
                        "age_s": max(0.0, ctx.now - stage["start"]),
                        "tier": summary["tier"],
                    }
                )
            if stage["leaf"] and (is_running or (stage["elapsed"] or 0) >= MIN_SPAN_S):
                spans.append(
                    {
                        "lane": lane_of(stage["stage"], summary["tier"]),
                        "label": stage["stage"],
                        "capsule": summary["capsule"],
                        "start": stage["start"],
                        "end": stage["end"],
                        "status": stage["status"],
                        "source": summary["path"],
                    }
                )
    stages = []
    for name, rows in durations.items():
        values = [r["elapsed"] for r in rows if isinstance(r["elapsed"], int | float)]
        stages.append(
            {
                "stage": name,
                "lane": lane_of(name),
                "count": len(rows),
                "p50_s": percentile(values, 0.5),
                "p95_s": percentile(values, 0.95),
                "max_s": max(values) if values else None,
                "total_s": sum(values) if values else None,
                "running": sum(1 for r in rows if r["end"] is None and r["status"] == "running"),
                # An envelope stage (it has child stages) contains the time of the others.
                "leaf": all(r["leaf"] for r in rows),
            }
        )
    stages.sort(key=lambda r: -(r["total_s"] or 0))
    sims = _sim_jobs(ctx, workspace)
    for job in sims:
        label = job["sim"] or job["lane"]
        if job["submitted"] is not None and (job["started"] or job["status"] == "queued"):
            spans.append(
                {
                    "lane": "queue",
                    "label": f"{label} queue",
                    "capsule": job["capsule"],
                    "start": job["submitted"],
                    "end": job["started"],
                    "status": "queued" if job["started"] is None else "waited",
                    "source": job["source"],
                }
            )
        if job["started"] is not None:
            spans.append(
                {
                    "lane": job["lane"],
                    "label": label,
                    "capsule": job["capsule"],
                    "start": job["started"],
                    "end": job["finished"],
                    "status": job["status"],
                    "source": job["source"],
                }
            )
        if job["status"] == "running":
            running.append(
                {
                    "stage": f"{label} simulation",
                    "capsule": job["capsule"],
                    "age_s": max(0.0, ctx.now - (job["started"] or job["submitted"] or ctx.now)),
                    "tier": job["lane"],
                }
            )
    spans.sort(key=lambda s: s["start"])
    if len(spans) > MAX_SPANS:
        notes.append(f"Gantt shows the newest {MAX_SPANS} of {len(spans)} spans.")
        spans = spans[-MAX_SPANS:]
    waits = [j["queue_wait_s"] for j in sims if j["queue_wait_s"] is not None]
    runtimes = [j["runtime_s"] for j in sims if j["runtime_s"] is not None]
    sim = (
        {
            "requests": len(sims),
            "queued": sum(j["status"] == "queued" for j in sims),
            "running": sum(j["status"] == "running" for j in sims),
            "done": sum(j["status"] in ("pass", "fail", "done", "error") for j in sims),
            "failed": sum(j["status"] == "fail" for j in sims),
            "errors": sum(j["status"] == "error" for j in sims),
            "queue_wait_p50": percentile(waits, 0.5),
            "queue_wait_p95": percentile(waits, 0.95),
            "runtime_p50": percentile(runtimes, 0.5),
            "runtime_p95": percentile(runtimes, 0.95),
        }
        if sims
        else {
            "requests": 0,
            "queued": None,
            "running": None,
            "done": None,
            "failed": None,
            "errors": None,
            "queue_wait_p50": None,
            "queue_wait_p95": None,
            "runtime_p50": None,
            "runtime_p95": None,
        }
    )
    if not ctx.live and running:
        notes.append("Records marked running belong to a run that is no longer live; they may be stale.")
    _notes_calibration(ctx, notes)
    running.sort(key=lambda r: -r["age_s"])
    section = {
        "spans": spans,
        "stages": stages[:48],
        "sim": sim,
        "running": running[:64],
        "records": len(records),
        "notes": notes,
    }
    ctx.state["memo"] = (key, section)
    THREAD_STAGES[str(ctx.experiment.get("id") or run_dir)] = threads if ctx.live else {}
    return section


def _inspect_lane(ctx, lane: str, records: dict, workspace: Path | None) -> dict[str, Any]:
    """One TIMELINE lane: its stages by total time, and for simulation lanes the queue and outcomes."""
    stages: dict[str, list[float]] = {}
    for record in records.values():
        for stage in record["stages"]:
            if lane_of(stage["stage"], record["tier"]) == lane and isinstance(stage["elapsed"], int | float):
                stages.setdefault(stage["stage"], []).append(stage["elapsed"])
    lines = [
        f"{sum(v):10.1f}s total  {len(v):5d}x  p50 {percentile(v, 0.5):7.2f}s  p95 {percentile(v, 0.95):7.2f}s  {name}"
        for name, v in sorted(stages.items(), key=lambda kv: -sum(kv[1]))[:30]
    ]
    sections = [{"title": f"stages in lane {lane} (by total time)", "lines": lines or ["no timed stage in this lane"]}]
    sims = [j for j in _sim_jobs(ctx, workspace) if lane == "queue" or j["lane"] == lane]
    if sims:
        waits = [j["queue_wait_s"] for j in sims if isinstance(j["queue_wait_s"], int | float)]
        runs = [j["runtime_s"] for j in sims if isinstance(j["runtime_s"], int | float)]
        outcome: dict[str, int] = {}
        for job in sims:
            outcome[job["status"]] = outcome.get(job["status"], 0) + 1
        summary = [
            f"{len(sims)} simulation request(s): " + ", ".join(f"{k} {v}" for k, v in sorted(outcome.items())),
            f"queue wait p50 {percentile(waits, 0.5) or 0:.1f}s p95 {percentile(waits, 0.95) or 0:.1f}s",
            f"runtime p50 {percentile(runs, 0.5) or 0:.1f}s p95 {percentile(runs, 0.95) or 0:.1f}s",
        ]
        slow = sorted(sims, key=lambda j: -(j["runtime_s"] or 0))[:12]
        summary += [
            f"{j['runtime_s'] or 0:8.1f}s run {j['queue_wait_s'] or 0:6.1f}s queue"
            f"  {j['status']:<7} {j['capsule'] or ''}"
            for j in slow
        ]
        sections.append({"title": "simulation jobs", "lines": summary})
    return {"title": f"lane {lane}", "sections": sections, "paths": []}


def inspect(ctx, key: str) -> dict[str, Any]:
    """A stage's duration profile, or (operator only) every recorded step of one capsule."""
    run_dir = Path(ctx.experiment["run_dir"])
    workspace = Path(ctx.experiment["workspace"]) if ctx.experiment.get("workspace") else None
    records = _index(ctx, _timing_dirs(run_dir, workspace))
    if key in LANES:
        return _inspect_lane(ctx, key, records, workspace)
    by_stage = [(s, r) for r in records.values() for s in r["stages"] if s["stage"] == key]
    if by_stage:
        values = sorted((s["elapsed"] or 0) for s, _ in by_stage)
        lines = [
            f"occurrences {len(by_stage)}; p50 {percentile(values, 0.5):.2f}s; p95 {percentile(values, 0.95):.2f}s; "
            f"max {values[-1]:.2f}s; lane {lane_of(key)}"
        ]
        slowest = sorted(by_stage, key=lambda sr: -(sr[0]["elapsed"] or 0))[:20]
        lines += [f"{s['elapsed'] or 0:9.2f}s  {r['operation'] or '-':<28} {r['capsule'] or ''}" for s, r in slowest]
        return {
            "title": f"stage {key}",
            "sections": [{"title": "slowest occurrences", "lines": lines}],
            "paths": sorted({r["path"] for _, r in slowest})[:20],
        }
    if not ctx.operator_private:
        return {
            "title": f"job {key}",
            "sections": [],
            "paths": [],
            "notice": "capsule timelines require operator-private mode",
        }
    mine = sorted((r for r in records.values() if r["capsule"] == key), key=lambda r: r["first"])
    lines = []
    for record in mine[-40:]:
        for stage in record["stages"]:
            end = f"{stage['elapsed']:.2f}s" if stage["elapsed"] is not None else "running"
            lines.append(
                f"{stage['start']:.0f}  {record['operation'] or '-':<24} {stage['stage']:<36} {end} {stage['status']}"
            )
    sims = [j for j in _sim_jobs(ctx, workspace) if key in (j["capsule"] or "")]
    sim_lines = [
        f"{j['sim'] or j['lane']}: {j['status']} queue {j['queue_wait_s'] or 0:.1f}s run {j['runtime_s'] or 0:.1f}s"
        f" rc {j['returncode']}"
        for j in sims
    ]
    invocations = []
    for fruns in sorted(
        (e.name for e in scan_dir(run_dir / "_qa_work", 4096) if e.name.startswith("fruns_")), reverse=True
    )[:16]:
        for bench in scan_dir(run_dir / "_qa_work" / fruns / "runs", 16):
            base = Path(bench.path) / key / "generated"
            for pattern in ("invocations/*/invocation.json", "*/invocations/*/invocation.json"):
                for path in list(base.glob(pattern))[:40]:
                    doc = obj(ctx.reader.document(path, limit=1 << 20))
                    start, end = _ns(doc.get("started_ns")), _ns(doc.get("finished_ns"))
                    took = f"{end - start:.2f}s" if start and end else "?"
                    invocations.append(
                        (
                            start or 0,
                            f"{clean(doc.get('stage'), 36):<36} rc {doc.get('returncode')} {took}  {fruns}",
                            str(path),
                        )
                    )
    invocations.sort()
    return {
        "title": f"capsule {key}",
        "sections": [
            {"title": "setup timeline", "lines": lines or ["no timeline records"]},
            {"title": "simulation jobs", "lines": sim_lines or ["no simulation requests"]},
            {"title": "invocations", "lines": [line for _, line, _ in invocations[-40:]] or ["none found"]},
        ],
        "paths": [r["path"] for r in mine[-10:]] + [p for _, _, p in invocations[-10:]],
    }


__all__ = ["LANE_RULES", "THREAD_STAGES", "inspect", "lane_of", "percentile", "read"]
