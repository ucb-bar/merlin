"""Where the wall time went: critical-path attribution, parallelism and phase markers.

Built from the ``agents`` and ``jobs`` sections (earlier in :data:`..model.SOURCES`) plus the run's
own completion telemetry, so it reads almost nothing itself. The lead agent drives the campaign, so its
time is the critical path and is split into four disjoint shares of the experiment window:

* ``think``: inside a turn with no tool call or wait outstanding (model reasoning and generation);
* ``tool``: a tool call outstanding (the union of overlapping calls);
* ``wait``: blocked on an await (subagents, background jobs) with no tool call outstanding;
* ``setup``: before the lead's first recorded activity (inputs, tooling readiness);
* ``idle``: no turn open after that (between sessions, relaunch, operator grading gaps).

When the run recorded its own split (``joint_campaign_summary.json`` ``telemetry``, method
``arrival_stamps``) that is reported beside the derived one as ``recorded``; the two are never mixed.
``overlap`` measures how the lead's time coincided with simulation jobs (``L2``/``L3`` lanes): time the
lead waited or ran tools while a simulation ran, and time simulations ran while the lead was thinking
(useful background work). ``concurrency`` is a per-bucket count of busy agents and running jobs.

Section::

    {"window": {start, end, seconds}, "lead": {id, think_s, tool_s, wait_s, setup_s, idle_s, basis}|None,
     "recorded": {think_s, tool_wait_s, between_session_s, measured_span_s, think_pct, source}|None,
     "overlap": {sim_busy_s, queue_s, lead_blocked_on_sim_s, sim_while_think_s},
     "parallelism": {agents_avg, agents_peak, helpers_used, helper_ceiling, jobs_avg, jobs_peak},
     "concurrency": {t0, dt, agents: [n], sim: [n], jobs: [n]},
     "markers": [{t, kind, label}], "verdict": {bound, share_pct, explanation}|None, notes}
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..io import clean, epoch, mtime, num, obj, rows, scan_dir
from . import SERIES_POINTS

SIM_LANES = ("L2", "L3")


def merge(intervals) -> list[tuple[float, float]]:
    """The union of ``(start, end)`` intervals as sorted, disjoint intervals."""
    out: list[list[float]] = []
    for a, b in sorted((a, b) for a, b in intervals if b > a):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def length(intervals) -> float:
    return sum(b - a for a, b in intervals)


def intersect(xs, ys) -> list[tuple[float, float]]:
    """Intersection of two disjoint sorted interval lists."""
    out, i, j = [], 0, 0
    while i < len(xs) and j < len(ys):
        a, b = max(xs[i][0], ys[j][0]), min(xs[i][1], ys[j][1])
        if b > a:
            out.append((a, b))
        if xs[i][1] < ys[j][1]:
            i += 1
        else:
            j += 1
    return out


def subtract(xs, ys) -> list[tuple[float, float]]:
    """``xs`` minus ``ys`` (both disjoint and sorted)."""
    out = []
    for a, b in xs:
        cur = a
        for c, d in ys:
            if d <= cur or c >= b:
                continue
            if c > cur:
                out.append((cur, c))
            cur = max(cur, d)
            if cur >= b:
                break
        if cur < b:
            out.append((cur, b))
    return out


def clip(intervals, t0: float, t1: float):
    return [(max(a, t0), min(b, t1)) for a, b in intervals if b > t0 and a < t1]


def _agent_spans(agent: dict, end: float) -> dict[str, list[tuple[float, float]]]:
    by_kind: dict[str, list[tuple[float, float]]] = {"turn": [], "tool": [], "wait": []}
    for span in agent.get("spans") or []:
        if not isinstance(span, list | tuple) or len(span) < 2:
            continue
        a, b = num(span[0]), num(span[1])
        kind = span[2] if len(span) > 2 and span[2] in by_kind else "turn"
        if a is None:
            continue
        by_kind[kind].append((a, b if b is not None else end))
    return {k: merge(v) for k, v in by_kind.items()}


def _job_spans(jobs: dict, end: float) -> dict[str, list[tuple[float, float]]]:
    lanes: dict[str, list[tuple[float, float]]] = {}
    for span in rows(jobs.get("spans")):
        a, b = num(span.get("start")), num(span.get("end"))
        if a is not None:
            lanes.setdefault(str(span.get("lane")), []).append((a, b if b is not None else end))
    return lanes


def _window(ctx, agents: list[dict], lanes: dict) -> tuple[float, float] | None:
    exp = ctx.experiment
    starts = [num(exp.get("started_at"))]
    ends = [ctx.now if ctx.live else num(exp.get("last_record_at"))]
    for agent in agents:
        for span in agent.get("spans") or []:
            if isinstance(span, list | tuple) and span:
                starts.append(num(span[0]))
                if not ctx.live and len(span) > 1:
                    ends.append(num(span[1]))
    starts = [s for s in starts if s is not None]
    ends = [e for e in ends if e is not None]
    if not starts or not ends:
        return None
    t0, t1 = min(starts), max(ends)
    return (t0, t1) if t1 > t0 else None


def _recorded(ctx, run: Path) -> dict | None:
    path = run / "joint_campaign_summary.json"
    telemetry = obj(obj(ctx.reader.document(path, limit=64 << 20)).get("telemetry"))
    think, tool = num(telemetry.get("think_generate_s")), num(telemetry.get("tool_and_wait_s"))
    if think is None or tool is None:
        return None
    return {
        "method": clean(telemetry.get("method"), 40),
        "think_s": think,
        "tool_wait_s": tool,
        "between_session_s": num(telemetry.get("between_session_s")),
        "measured_span_s": num(telemetry.get("measured_span_s")),
        "think_pct": num(telemetry.get("think_pct")),
        "tool_concurrency_overlap_s": num(telemetry.get("tool_concurrency_overlap_s")),
        "output_tokens_per_think_s": num(obj(telemetry.get("rates")).get("output_tokens_per_think_generate_s")),
        "source": str(path) if ctx.operator_private else path.name,
    }


def _markers(ctx, run: Path) -> list[dict]:
    """Round boundaries and the campaign's stop, each from its own record."""
    out = []
    rounds_dir = run / "rounds"
    names = sorted(e.name for e in scan_dir(rounds_dir, 4096) if e.name.endswith(".codex_summary.json"))
    for name in names[-64:]:
        summary = obj(ctx.reader.document(rounds_dir / name))
        label = name.split(".", 1)[0]
        wall, ended = num(summary.get("wall_s")), mtime(rounds_dir / name)
        if ended is not None:
            if wall:
                out.append({"t": ended - wall, "kind": "round", "label": f"{label} start"})
            reason = clean(summary.get("interruption_reason") or "", 60)
            out.append({"t": ended, "kind": "round", "label": f"{label} end" + (f" ({reason})" if reason else "")})
    for name, kind in (("joint_epoch_stop.json", "stop"), ("joint_campaign_summary.json", "finish")):
        record = obj(ctx.reader.document(run / name, limit=64 << 20))
        when = epoch(record.get("stopped_at") or record.get("finished_at") or record.get("generated_at"))
        if when is None and record:
            when = mtime(run / name)
        if when is not None:
            reason = clean(record.get("completion_reason") or record.get("reason") or kind, 40)
            out.append({"t": when, "kind": kind, "label": reason})
    return sorted(out, key=lambda m: m["t"])


def _counts(groups: list[list[tuple[float, float]]], t0: float, dt: float, n: int) -> list[float]:
    """Average number of busy members per bucket (fractional occupancy summed over members)."""
    out = [0.0] * n
    for spans in groups:
        for a, b in spans:
            i0, i1 = int((a - t0) / dt), min(n - 1, int((b - t0) / dt))
            for i in range(max(0, i0), i1 + 1):
                lo, hi = t0 + i * dt, t0 + (i + 1) * dt
                out[i] += max(0.0, min(b, hi) - max(a, lo)) / dt
    return [round(v, 2) for v in out]


def _verdict(lead: dict | None, recorded: dict | None, overlap: dict, window_s: float) -> dict | None:
    if recorded and (recorded["measured_span_s"] or 0) > 0:
        span = recorded["measured_span_s"]
        shares = {"agent_reasoning": recorded["think_s"] / span, "tools_and_waits": recorded["tool_wait_s"] / span}
        basis = "recorded arrival-stamp telemetry"
    elif lead:
        shares = {
            "agent_reasoning": lead["think_s"] / window_s,
            "tools": lead["tool_s"] / window_s,
            "waits": lead["wait_s"] / window_s,
            "idle": lead["idle_s"] / window_s,
            "setup": lead.get("setup_s", 0) / window_s,
        }
        basis = "rollout spans of the lead agent"
    else:
        return None
    bound = max(shares, key=shares.get)
    share = shares[bound] * 100
    words = {
        "agent_reasoning": "the lead spends most of the run reasoning/generating; faster or parallel agents help most",
        "tools_and_waits": "the lead mostly waits on tool calls; speeding up builds/simulation helps most",
        "tools": "the lead mostly waits on its tool calls; speeding up builds/selfchecks helps most",
        "waits": "the lead mostly waits on subagents or background jobs",
        "idle": "no turn is open for most of the run (relaunches, grading gaps)",
        "setup": "the run is still (or mostly) in setup before the lead agent's first turn",
    }
    blocked = overlap.get("lead_blocked_on_sim_s") or 0
    text = words[bound]
    if blocked and blocked / window_s >= 0.1:
        text += f"; {blocked / window_s * 100:.0f}% of the window the lead waited while a simulation ran"
    return {"bound": bound, "share_pct": round(share, 1), "explanation": text, "basis": basis}


def read(ctx) -> dict[str, Any]:
    agents_section = obj(ctx.shared.get("agents"))
    jobs = obj(ctx.shared.get("jobs"))
    agents = rows(agents_section.get("agents"))
    run = Path(ctx.experiment["run_dir"])
    notes: list[str] = []
    # A span still open in a run that is no longer live ends at the run's last record, not now.
    now = ctx.now if ctx.live else (num(ctx.experiment.get("last_record_at")) or ctx.now)
    lanes = _job_spans(jobs, now)
    window = _window(ctx, agents, lanes)
    recorded = _recorded(ctx, run) if run.name else None
    empty = {
        "window": None,
        "lead": None,
        "recorded": recorded,
        "overlap": {},
        "parallelism": {},
        "concurrency": None,
        "markers": [],
        "verdict": _verdict(None, recorded, {}, 1.0),
        "notes": ["no agent or job spans recorded"],
    }
    if window is None:
        return empty
    t0, t1 = window
    window_s = t1 - t0
    per_agent = {str(a.get("id")): _agent_spans(a, t1) for a in agents}
    # Prefer the index's exact unions (the section's drawn spans are bounded to the newest ones).
    from . import agents as agents_source

    threads = agents_source.index_for(ctx).threads
    for agent_id in per_agent:
        thread = threads.get(agent_id)
        if thread is not None and any(thread["intervals"].values()):
            exact = agents_source.intervals(thread, t1, ctx.live)
            per_agent[agent_id] = {k: merge(v) for k, v in exact.items()}
    lead_row = next((a for a in agents if a.get("role") == "lead"), agents[0] if agents else None)
    lead = None
    if lead_row is not None:
        spans = {k: clip(v, t0, t1) for k, v in per_agent[str(lead_row.get("id"))].items()}
        tool = spans["tool"]
        wait = subtract(spans["wait"], tool)
        busy = merge(spans["turn"] + tool + wait)
        think = subtract(subtract(busy, tool), wait)
        # Before the lead's first span the run is setting up (inputs, tooling readiness); after it,
        # time with no turn open is idle.
        setup = (busy[0][0] - t0) if busy else window_s
        lead = {
            "id": lead_row.get("id"),
            "think_s": round(length(think), 1),
            "tool_s": round(length(tool), 1),
            "wait_s": round(length(wait), 1),
            "setup_s": round(setup, 1),
            "idle_s": round(max(0.0, window_s - length(busy) - setup), 1),
            "basis": "rollout spans",
        }
        if not spans["turn"] and not tool:
            notes.append("the lead agent recorded no spans in this window")
    sim = merge(s for lane in SIM_LANES for s in clip(lanes.get(lane, []), t0, t1))
    queue = merge(clip(lanes.get("queue", []), t0, t1))
    overlap = {"sim_busy_s": round(length(sim), 1), "queue_s": round(length(queue), 1)}
    if lead is not None:
        lead_spans = {k: clip(v, t0, t1) for k, v in per_agent[str(lead_row.get("id"))].items()}
        blocked = merge(lead_spans["tool"] + lead_spans["wait"])
        busy = merge(lead_spans["turn"] + blocked)
        overlap["lead_blocked_on_sim_s"] = round(length(intersect(blocked, sim)), 1)
        overlap["sim_while_think_s"] = round(length(intersect(subtract(busy, blocked), sim)), 1)
    n = min(SERIES_POINTS, max(1, int(window_s // 5)))
    dt = window_s / n
    agent_groups = [merge(clip(sum(v.values(), []), t0, t1)) for v in per_agent.values()]
    agent_counts = _counts(agent_groups, t0, dt, n)
    sim_counts = _counts([clip(lanes.get(lane, []), t0, t1) for lane in SIM_LANES], t0, dt, n)
    job_counts = _counts([clip(v, t0, t1) for k, v in lanes.items() if k != "queue"], t0, dt, n)
    active_agent_s = sum(length(g) for g in agent_groups)
    totals = obj(agents_section.get("totals"))
    parallelism = {
        "agents_avg": round(active_agent_s / window_s, 2),
        "agents_peak": round(max(agent_counts, default=0)),
        "helpers_used": sum(1 for a in agents if a.get("role") != "lead" and per_agent[str(a.get("id"))]["turn"]),
        "helper_ceiling": totals.get("helper_ceiling"),
        "jobs_avg": round(sum(job_counts) / n, 2) if n else None,
        "jobs_peak": round(max(job_counts, default=0)),
    }
    if jobs.get("notes") and any("newest" in str(x) for x in jobs["notes"]):
        notes.append("job concurrency covers only the job spans the jobs reader kept")
    return {
        "window": {"start": t0, "end": t1, "seconds": round(window_s, 1)},
        "lead": lead,
        "recorded": recorded,
        "overlap": overlap,
        "parallelism": parallelism,
        "concurrency": {"t0": t0, "dt": dt, "agents": agent_counts, "sim": sim_counts, "jobs": job_counts},
        "markers": [m for m in _markers(ctx, run) if t0 - 60 <= m["t"] <= t1 + 60],
        "verdict": _verdict(lead, recorded, overlap, window_s),
        "notes": notes,
    }


__all__ = ["intersect", "length", "merge", "read", "subtract"]
