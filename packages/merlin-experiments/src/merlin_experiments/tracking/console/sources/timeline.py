"""Where the wall time went: critical-path attribution, parallelism and phase markers.

Built from the ``agents`` and ``jobs`` sections (earlier in :data:`..model.SOURCES`) plus the run's
own completion telemetry, so it reads almost nothing itself. Time categories describe recorded
intervals, not isolated model deliberation. Inner shell commands can
continue in the background while outer tool calls return and the lead works. ``tool_s`` and ``think_s``
are retained compatibility fields for command lifetimes and time outside those lifetimes; their names
must not be interpreted as blocked time or pure thinking. Matched outer call/return spans supply a
separate tool interval when recorded. Non-tool residual includes queueing, prefill, generation, network
and local processing. Simulation coincidence does not establish causal waiting. Agent concurrency
includes the lead; helper concurrency is reported separately, with exact and sampled peaks distinct.

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
    by_kind: dict[str, list[tuple[float, float]]] = {"turn": [], "tool": [], "wait": [], "tool_call": []}
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


def _peak(groups):
    """Exact peak of half-open interval unions, rather than a rounded bucket average."""
    changes = {}
    for spans in groups:
        for a, b in spans:
            changes[a] = changes.get(a, 0) + 1
            changes[b] = changes.get(b, 0) - 1
    active = peak = 0
    for when in sorted(changes):
        active += changes[when]
        peak = max(peak, active)
    return peak


def _verdict(lead: dict | None, recorded: dict | None, overlap: dict, window_s: float) -> dict | None:
    # The derived split covers the whole window (setup, turn startups and idle time included); the run's
    # recorded split measures only inside sessions, so it decides only when no lead spans exist.
    if not lead and recorded and (recorded["measured_span_s"] or 0) > 0:
        span = recorded["measured_span_s"]
        shares = {"unclassified_agent_time": recorded["think_s"] / span,
                  "tools_and_waits": recorded["tool_wait_s"] / span}
        basis = "recorded arrival-stamp telemetry (inside sessions only)"
    elif lead:
        shares = {
            "unclassified_agent_time": (lead.get("unclassified_non_tool_s")
                                        if lead.get("unclassified_non_tool_s") is not None
                                        else lead["think_s"]) / window_s,
            "outer_tools" if lead.get("outer_tool_call_s") is not None else "command_lifetimes":
                (lead.get("outer_tool_call_s") if lead.get("outer_tool_call_s") is not None
                 else lead["tool_s"]) / window_s,
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
        "unclassified_agent_time": "time outside recorded tools; deliberation, queueing and service are not separated",
        "outer_tools": "matched outer tool calls occupy the largest recorded share; their cause is not inferred",
        "command_lifetimes": "inner command lifetimes dominate; they can overlap continuing lead work",
        "tools_and_waits": "recorded tool/wait intervals dominate; causal simulation blocking is not established",
        "tools": "the lead mostly waits on its tool calls; speeding up builds/selfchecks helps most",
        "waits": "recorded extension/wait intervals dominate; the awaited cause is not inferred",
        "idle": "no turn is open for most of the run (relaunches, grading gaps)",
        "setup": "the author mostly waits for the host to prepare its turns (inputs, sandbox, overlays)",
    }
    text = words[bound]
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
    # A relaunched round starts a new lead thread: the lead is every lead row (one per session),
    # not the first. Counting only the first reported the later rounds' work as idle.
    lead_rows = [a for a in agents if a.get("role") == "lead"] or agents[:1]
    lead_row = lead_rows[0] if lead_rows else None
    lead_spans = {k: merge(s for a in lead_rows for s in per_agent[str(a.get("id"))].get(k, ()))
                  for k in ("turn", "tool", "wait", "tool_call")}
    lead = None
    if lead_row is not None:
        spans = {k: clip(v, t0, t1) for k, v in lead_spans.items()}
        tool = spans["tool"]
        wait = subtract(spans["wait"], tool)
        busy = merge(spans["turn"] + tool + wait)
        think = subtract(subtract(busy, tool), wait)
        # Before the lead's first span the run is setting up (inputs, tooling readiness), and so it is
        # while the host prepares a turn (author startup); other time with no turn open is idle.
        startup = merge(
            (t["startup_start"], t["startup_end"] or t1)
            for t in rows(obj(ctx.shared.get("lifecycle")).get("turns"))
            if isinstance(t.get("startup_start"), int | float)
        )
        before = [(t0, busy[0][0])] if busy else [(t0, t1)]
        setup = length(subtract(clip(merge(before + startup), t0, t1), busy))
        outer = spans["tool_call"]
        has_outer = all(a.get("outer_tool_calls_recorded") is True
                        and a.get("intervals_complete") is not False for a in lead_rows)
        residual = subtract(subtract(merge(spans["turn"] + outer + wait), outer), wait) if has_outer else None
        lead = {
            "id": lead_row.get("id"),
            "think_s": round(length(think), 1),
            "tool_s": round(length(tool), 1),
            "wait_s": round(length(wait), 1),
            "setup_s": round(setup, 1),
            "idle_s": round(max(0.0, window_s - length(busy) - setup), 1),
            "command_lifetime_s": round(length(tool), 1),
            "outer_tool_call_s": round(length(outer), 1) if has_outer else None,
            "unclassified_non_tool_s": round(length(residual), 1) if residual is not None else None,
            "pure_deliberation_s": None,
            "intervals_complete": (False if any(a.get("intervals_complete") is False for a in lead_rows)
                                   else lead_row.get("intervals_complete")),
            "sessions": len(lead_rows),
            "basis": "rollout spans; legacy think/tool fields describe command lifetime partition",
        }
        if not spans["turn"] and not tool:
            notes.append("the lead agent recorded no spans in this window")
    sim = merge(s for lane in SIM_LANES for s in clip(lanes.get(lane, []), t0, t1))
    queue = merge(clip(lanes.get("queue", []), t0, t1))
    overlap = {"sim_busy_s": round(length(sim), 1), "queue_s": round(length(queue), 1)}
    if lead is not None:
        clipped = {k: clip(v, t0, t1) for k, v in lead_spans.items()}
        overlap["command_lifetime_during_sim_s"] = round(length(intersect(clipped["tool"], sim)), 1)
        overlap["outer_tool_calls_during_sim_s"] = (
            round(length(intersect(clipped["tool_call"], sim)), 1)
            if all(a.get("outer_tool_calls_recorded") for a in lead_rows) else None)
        overlap["lead_blocked_on_sim_s"] = None  # Coincident intervals do not identify an awaited job.
        overlap["sim_while_think_s"] = None  # No producer isolates pure deliberation.
    n = min(SERIES_POINTS, max(1, int(window_s // 5)))
    dt = window_s / n
    agent_groups = [merge(clip(v["turn"], t0, t1)) for v in per_agent.values()]
    agent_counts = _counts(agent_groups, t0, dt, n)
    sim_counts = _counts([clip(lanes.get(lane, []), t0, t1) for lane in SIM_LANES], t0, dt, n)
    job_counts = _counts([clip(v, t0, t1) for k, v in lanes.items() if k != "queue"], t0, dt, n)
    active_agent_s = sum(length(g) for g in agent_groups)
    helper_groups = [merge(clip(per_agent[str(a.get("id"))]["turn"], t0, t1))
                     for a in agents if a.get("role") != "lead"]
    totals = obj(agents_section.get("totals"))
    exact_intervals = all(a.get("intervals_complete") is not False for a in agents)
    parallelism = {
        "agents_avg": round(active_agent_s / window_s, 2),
        "agents_peak": _peak(agent_groups) if exact_intervals else None,
        "agents_peak_sampled": round(max(agent_counts, default=0)),
        "helpers_avg": round(sum(length(g) for g in helper_groups) / window_s, 2),
        "helpers_peak": _peak(helper_groups) if exact_intervals else None,
        "intervals_complete": exact_intervals,
        "basis": "observed task turns; agents include lead; not CPU utilization",
        "helpers_used": sum(1 for a in agents if a.get("role") != "lead" and per_agent[str(a.get("id"))]["turn"]),
        "helper_ceiling": totals.get("helper_ceiling"),
        "jobs_avg": round(sum(job_counts) / n, 2) if n else None,
        "jobs_peak": round(max(job_counts, default=0)),
    }
    if jobs.get("notes") and any("newest" in str(x) for x in jobs["notes"]):
        notes.append("job concurrency covers only the job spans the jobs reader kept")
    notes.append("Pure thinking and causal simulator blocking are unavailable; command lifetimes may be asynchronous.")
    return {
        "window": {"start": t0, "end": t1, "seconds": round(window_s, 1)},
        "lead": lead,
        "recorded": recorded,
        "overlap": overlap,
        "parallelism": parallelism,
        "concurrency": {"t0": t0, "dt": dt, "agents": agent_counts,
                        "helpers": _counts(helper_groups, t0, dt, n), "sim": sim_counts, "jobs": job_counts},
        "markers": [m for m in _markers(ctx, run) if t0 - 60 <= m["t"] <= t1 + 60],
        "verdict": _verdict(lead, recorded, overlap, window_s),
        "notes": notes,
    }


__all__ = ["intersect", "length", "merge", "read", "subtract"]
