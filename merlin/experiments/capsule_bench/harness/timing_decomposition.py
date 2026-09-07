#!/usr/bin/env python3
"""Where an agentic run's WALL TIME went — think/generate vs tool-and-wait — derived from the
transcript's own arrival stamps, for ANY driver.

WHY THIS EXISTS IN THIS SHAPE. The first version computed the split as
``think+gen = sum(result.duration_api_ms)`` and ``tool = duration_ms - duration_api_ms``. Those two
fields are emitted only by the claude CLI's terminal ``result`` event. Point that arithmetic at a codex
(or opencode) transcript and every term is missing, so it does not fail and it does not say it cannot
measure -- it reports ``think_generate_s: 0.0, tool_and_wait_s: 0.0, think_pct: 0.0``, which reads as
"this agent spent no time thinking". A measured 5.4-hour atlas run recorded exactly that. This is the
repo's recurring "a check that could not run reported success" defect, applied to a timing axis.

WHAT IS DERIVED INSTEAD. Every event the harness normalizes carries ``arrived_at`` (ISO-8601, stamped by
the harness as the event was read off the driver's stream), so the timeline is measurable without any
driver-specific duration field:

  * a tool call OCCUPIES ``[arrived_at(tool_use), arrived_at(tool_result)]``;
  * time with NO tool call outstanding is think+generate.

Tool calls OVERLAP -- codex backgrounds a long command and keeps working (measured: one 2118 s
``atlas-opt`` invocation ran while 90 further tool calls completed under it). So wall time is the UNION
of the tool intervals, never their sum; both are reported, and their difference is the concurrency the
sum would otherwise invent. A transcript that concatenates several rounds is split into SEGMENTS at each
``system``/``init`` event, so the operator's between-round grading gap is reported separately and never
charged to the agent.

FAIL CLOSED. When a transcript carries no arrival stamps (claude and opencode transcripts do not today)
the split is recorded as ``null`` with ``unavailable_reason``, never as 0.0 -- a zero that means
"not measured" is the defect this module exists to remove. ``duration_api_ms`` is still honoured when it
is genuinely present and non-zero, labelled as such in ``method``.

CLI
    timing_decomposition.py --run-dir out/runs/<target>/<suite>/<arm>/<run-id> [--write]
    timing_decomposition.py --arms                # legacy cross-arm view (needs the experiment env)
"""
from __future__ import annotations

import json
import hashlib
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

#: Content-block / item vocabularies, matched EXACTLY (never substring-matched, never regex).
_BLOCK_TOOL_USE = "tool_use"
_BLOCK_TOOL_RESULT = "tool_result"
#: Item types that denote a tool call in a raw `codex exec --json` stream. Adding a driver is adding a
#: name here, not editing the timeline algebra.
_RAW_TOOL_ITEMS = frozenset({"command_execution", "file_change", "mcp_tool_call", "web_search"})
_RAW_ITEM_STARTED = "item.started"
_RAW_ITEM_COMPLETED = "item.completed"

OPEN, CLOSE = "open", "close"


# --- stamps ---------------------------------------------------------------------------------------

def _stamp(value) -> float | None:
    """ISO-8601 -> POSIX seconds, or None. Never raises: an unparseable stamp is a MISSING stamp."""
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None


def _event_time(evt: dict) -> float | None:
    """The moment this event was read off the driver's stream, whatever the driver calls it."""
    for key in ("arrived_at", "started_at", "timestamp"):
        t = _stamp(evt.get(key))
        if t is not None:
            return t
    return None


# --- structural extraction ------------------------------------------------------------------------

def _json_chars(value) -> int:
    """Stable serialized size used only as an index into the authoritative transcript artifact."""
    try:
        return len(json.dumps(value, sort_keys=True, ensure_ascii=False))
    except (TypeError, ValueError):
        return len(str(value))


def _boundaries(evt: dict) -> list[dict]:
    """Tool boundaries with identity, name, error state, and I/O sizes.

    Full inputs/outputs remain in the raw transcript.  Repeating them in the telemetry report would
    create a second, potentially clipped copy; sizes and artifact hashes make that copy auditable.
    """
    out: list[dict] = []
    msg = evt.get("message")
    if isinstance(msg, dict):
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                bt = block.get("type")
                if bt == _BLOCK_TOOL_USE:
                    out.append({"kind": OPEN, "call_id": str(block.get("id") or ""),
                                "name": str(block.get("name") or "unknown"),
                                "input_chars": _json_chars(block.get("input"))})
                elif bt == _BLOCK_TOOL_RESULT:
                    out.append({"kind": CLOSE, "call_id": str(block.get("tool_use_id") or ""),
                                "error": bool(block.get("is_error")),
                                "output_chars": _json_chars(block.get("content"))})
    inner = evt.get("event") if isinstance(evt.get("event"), dict) else evt
    etype, item = inner.get("type"), inner.get("item")
    if etype in (_RAW_ITEM_STARTED, _RAW_ITEM_COMPLETED) and isinstance(item, dict):
        if item.get("type") in _RAW_TOOL_ITEMS:
            rec = {"kind": OPEN if etype == _RAW_ITEM_STARTED else CLOSE,
                   "call_id": str(item.get("id") or ""),
                   "name": str(item.get("type") or "unknown")}
            if etype == _RAW_ITEM_STARTED:
                rec["input_chars"] = _json_chars(item)
            else:
                rec["error"] = bool(item.get("exit_code")) or item.get("status") == "failed"
                rec["output_chars"] = _json_chars(item.get("aggregated_output") or item)
            out.append(rec)
    return out


def _marks(evt: dict) -> list[tuple[str, str]]:
    """The tool-call boundaries this event announces, as ``(OPEN|CLOSE, call_id)``.

    Recognises the harness's normalized claude-shaped events AND a raw ``codex exec --json`` envelope,
    so the same timeline algebra reads either file. An event in neither shape yields nothing (it is a
    timeline TICK, not a tool boundary) rather than being guessed at.
    """
    return [(b["kind"], b["call_id"]) for b in _boundaries(evt)]


def _is_segment_start(evt: dict) -> bool:
    """A new agent SESSION begins here (the harness's per-round init header)."""
    return evt.get("type") == "system" and evt.get("subtype") == "init"


def read_events(paths) -> list[dict]:
    """JSONL -> dicts, in file order. Unparseable lines are skipped, not fatal."""
    evts: list[dict] = []
    for p in paths:
        p = Path(p)
        if not p.is_file():
            continue
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if isinstance(obj, dict):
                evts.append(obj)
    return evts


# --- timeline algebra -----------------------------------------------------------------------------

def _union_seconds(intervals: list[tuple[float, float]]) -> float:
    """Wall seconds covered by AT LEAST ONE interval. Overlapping tool calls occupy one clock."""
    total = 0.0
    cur_a = cur_b = None
    for a, b in sorted(intervals):
        if cur_b is None or a > cur_b:
            if cur_b is not None:
                total += cur_b - cur_a
            cur_a, cur_b = a, b
        elif b > cur_b:
            cur_b = b
    if cur_b is not None:
        total += cur_b - cur_a
    return total


def _segments(events: list[dict]) -> list[list[dict]]:
    segs: list[list[dict]] = []
    cur: list[dict] = []
    for evt in events:
        if _is_segment_start(evt) and cur:
            segs.append(cur)
            cur = []
        cur.append(evt)
    if cur:
        segs.append(cur)
    return segs or [[]]


def _token_summary(events: list[dict]) -> dict:
    """Separate every provider token bucket without pricing or double counting.

    The normalized Codex stream uses the Claude usage vocabulary.  ``input_tokens`` is fresh input,
    cache creation is a write, and cache read is a hit.  Reasoning is a labelled subset of output.
    Message-id deduplication is mandatory because streaming drivers may repeat an assistant envelope.
    """
    seen: set[str] = set()
    totals = {"fresh_input": 0, "cache_write": 0, "cache_read": 0,
              "output": 0, "reasoning": 0}
    usage_messages = 0
    for i, evt in enumerate(events):
        if evt.get("type") != "assistant":
            continue
        msg = evt.get("message") or {}
        usage = msg.get("usage") or {}
        if not isinstance(usage, dict) or not usage:
            continue
        mid = str(msg.get("id") or f"anonymous_usage_{i}")
        if mid in seen:
            continue
        seen.add(mid)
        usage_messages += 1
        totals["fresh_input"] += int(usage.get("input_tokens", 0) or 0)
        totals["cache_write"] += int(usage.get("cache_creation_input_tokens", 0) or 0)
        totals["cache_read"] += int(usage.get("cache_read_input_tokens", 0) or 0)
        totals["output"] += int(usage.get("output_tokens", 0) or 0)
        totals["reasoning"] += int(usage.get("reasoning_output_tokens", 0) or 0)
    input_traffic = totals["fresh_input"] + totals["cache_write"] + totals["cache_read"]
    total = input_traffic + totals["output"]
    return {
        "available": bool(usage_messages),
        "usage_messages": usage_messages,
        "tokens_fresh_input": totals["fresh_input"] if usage_messages else None,
        "tokens_cache_write": totals["cache_write"] if usage_messages else None,
        "tokens_cache_read": totals["cache_read"] if usage_messages else None,
        # Compatibility alias used by the existing experiment reports.
        "tokens_cached": totals["cache_read"] if usage_messages else None,
        "tokens_output": totals["output"] if usage_messages else None,
        "tokens_reasoning": totals["reasoning"] if usage_messages else None,
        "tokens_input_traffic": input_traffic if usage_messages else None,
        "tokens_total": total if usage_messages else None,
        "cache_read_share_of_input": (totals["cache_read"] / input_traffic
                                      if usage_messages and input_traffic else None),
        "cache_write_share_of_input": (totals["cache_write"] / input_traffic
                                       if usage_messages and input_traffic else None),
        "reasoning_is_subset_of_output": True,
        **({} if usage_messages else {
            "unavailable_reason": "no provider usage event has arrived; tokens are unknown, not zero"}),
    }


def _percentile(values: list[float], fraction: float) -> float | None:
    """Nearest-rank percentile; deterministic for tiny per-tool samples."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, int((len(ordered) * fraction) + 0.999999999))
    return ordered[min(rank, len(ordered)) - 1]


def _tool_rollup(calls: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for call in calls:
        grouped[call.get("name") or "unknown"].append(call)
    by_tool = {}
    for name, rows in sorted(grouped.items()):
        durations = [float(r["duration_s"]) for r in rows if r.get("duration_s") is not None]
        completed = [r for r in rows if r.get("completed")]
        by_tool[name] = {
            "calls_started": len(rows),
            "calls_completed": len(completed),
            "calls_unterminated": len(rows) - len(completed),
            "errors": sum(bool(r.get("error")) for r in completed),
            "duration_sum_s": round(sum(durations), 6),
            "duration_mean_s": round(sum(durations) / len(durations), 6) if durations else None,
            "duration_p50_s": _percentile(durations, 0.50),
            "duration_p95_s": _percentile(durations, 0.95),
            "duration_max_s": max(durations) if durations else None,
            "input_chars": sum(int(r.get("input_chars") or 0) for r in rows),
            "output_chars": sum(int(r.get("output_chars") or 0) for r in rows),
        }
    return {"used": sorted(grouped), "by_tool": by_tool, "calls": calls,
            "exact_io_source": ("the raw provider event stream; normalized tool outputs may be clipped "
                                "for prompt safety, so this report stores sizes and hashes, not a second copy")}


def _token_rates(tokens: dict, think_s: float | None, span_s: float | None) -> dict:
    """Derived throughput with denominators in every field name.

    Output/think time is useful as an activity-normalized rate, but is NOT provider decode throughput:
    arrival stamps bracket messages and tools, not first-token and last-token network events.
    """
    output = tokens.get("tokens_output")
    total = tokens.get("tokens_total")
    return {
        "output_tokens_per_think_generate_s": (
            output / think_s if output is not None and think_s and think_s > 0 else None),
        "output_tokens_per_agent_span_s": (
            output / span_s if output is not None and span_s and span_s > 0 else None),
        "total_tokens_per_agent_span_s": (
            total / span_s if total is not None and span_s and span_s > 0 else None),
        "note": ("derived from transcript arrival intervals; activity-normalized throughput, not a "
                 "provider-side first-token/decode latency measurement"),
    }


def _decompose_segment(events: list[dict]) -> dict | None:
    """One agent session -> its span, its tool-busy union, its think time. None if unstamped."""
    stamped: list[tuple[float, list[dict]]] = []
    for evt in events:
        t = _event_time(evt)
        if t is None:
            continue
        stamped.append((t, _boundaries(evt)))
    if len(stamped) < 2:
        return None
    t0, t_end = stamped[0][0], stamped[-1][0]
    open_at: dict[str, dict] = {}
    intervals: list[tuple[float, float]] = []
    calls: list[dict] = []
    unmatched_close = 0
    for t, boundaries in stamped:
        for boundary in boundaries:
            kind, call_id = boundary["kind"], boundary["call_id"]
            if kind == OPEN:
                open_at.setdefault(call_id, {"start": t,
                                             "name": boundary.get("name") or "unknown",
                                             "input_chars": boundary.get("input_chars", 0)})
            elif call_id in open_at:
                opened = open_at.pop(call_id)
                start = opened["start"]
                intervals.append((start, t))
                calls.append({"call_id": call_id, "name": opened["name"],
                              "started_at": datetime.fromtimestamp(start, timezone.utc).isoformat(),
                              "ended_at": datetime.fromtimestamp(t, timezone.utc).isoformat(),
                              "duration_s": round(max(t - start, 0.0), 6),
                              "completed": True, "error": bool(boundary.get("error")),
                              "input_chars": opened.get("input_chars", 0),
                              "output_chars": boundary.get("output_chars", 0)})
            else:
                unmatched_close += 1
    # A tool call whose result never arrived (the round was cut off mid-command) OCCUPIED the clock up
    # to the last thing we saw. Clamping to the segment end is the only reading the stamps support;
    # dropping it would silently move that wall time into "thinking".
    still_open = len(open_at)
    for call_id, opened in open_at.items():
        start = opened["start"]
        intervals.append((start, t_end))
        calls.append({"call_id": call_id, "name": opened["name"],
                      "started_at": datetime.fromtimestamp(start, timezone.utc).isoformat(),
                      "ended_at": None, "duration_s": round(max(t_end - start, 0.0), 6),
                      "completed": False, "error": None,
                      "input_chars": opened.get("input_chars", 0), "output_chars": None})
    busy = _union_seconds(intervals)
    span = t_end - t0
    return {
        "span_s": span,
        "tool_and_wait_s": busy,
        "think_generate_s": max(span - busy, 0.0),
        "tool_calls_matched": len(intervals) - still_open,
        "tool_calls_unterminated": still_open,
        "tool_results_unpaired": unmatched_close,
        "tool_call_seconds_sum": sum(b - a for a, b in intervals),
        "tool_calls": calls,
        "first_s": t0,
        "last_s": t_end,
    }


def _from_duration_fields(events: list[dict]) -> dict | None:
    """The legacy claude-only reading: ``result.duration_api_ms`` vs ``result.duration_ms``.

    Kept as a FALLBACK for stamp-free transcripts that genuinely carry it, and returns None -- not
    zero -- when they do not. Retries/resumes emit several ``result`` events per round, so the LAST
    per segment is taken rather than all of them summed.
    """
    api_ms = total_ms = 0
    seen = False
    for seg in _segments(events):
        last = None
        for evt in seg:
            if evt.get("type") == "result":
                last = evt
        if last is None:
            continue
        a = last.get("duration_api_ms") or 0
        d = last.get("duration_ms") or 0
        if a or d:
            seen = True
        api_ms += a
        total_ms += d
    if not seen or total_ms <= 0:
        return None
    return {"think_generate_s": api_ms / 1000.0,
            "tool_and_wait_s": max(0.0, total_ms - api_ms) / 1000.0,
            "span_s": total_ms / 1000.0}


UNAVAILABLE_NOTE = (
    "the transcript carries no per-event arrival stamps and no non-zero duration_api_ms/duration_ms, "
    "so the think-vs-tool split is NOT MEASURED for this run. It is recorded as null, never 0.0: a "
    "zero here would read as 'the agent spent no time thinking' and would be averaged into a study.")


def decompose(events: list[dict]) -> dict:
    """Driver-agnostic think/tool wall-time split. ``method`` says how it was obtained, and
    ``method: 'unknown'`` with null fields is the honest answer when the transcript cannot support one."""
    groups = _segments(events)
    pairs = [(group, _decompose_segment(group)) for group in groups]
    pairs = [(group, seg) for group, seg in pairs if seg]
    segs = [seg for _group, seg in pairs]
    if segs:
        think = sum(s["think_generate_s"] for s in segs)
        tool = sum(s["tool_and_wait_s"] for s in segs)
        span = sum(s["span_s"] for s in segs)
        gap = 0.0
        for prev, nxt in zip(segs, segs[1:]):
            gap += max(nxt["first_s"] - prev["last_s"], 0.0)
        call_sum = sum(s["tool_call_seconds_sum"] for s in segs)
        all_calls = [call for seg in segs for call in seg["tool_calls"]]
        tokens = _token_summary(events)
        session_rows = []
        for index, (group, seg) in enumerate(pairs):
            session_tokens = _token_summary(group)
            session_rows.append({
                "session": index,
                "started_at": datetime.fromtimestamp(seg["first_s"], timezone.utc).isoformat(),
                "ended_at": datetime.fromtimestamp(seg["last_s"], timezone.utc).isoformat(),
                "wall_s": round(seg["span_s"], 6),
                "think_generate_s": round(seg["think_generate_s"], 6),
                "tool_and_wait_s": round(seg["tool_and_wait_s"], 6),
                "activity_share": {
                    "think_generate": (seg["think_generate_s"] / seg["span_s"]
                                       if seg["span_s"] > 0 else None),
                    "tool_and_wait": (seg["tool_and_wait_s"] / seg["span_s"]
                                      if seg["span_s"] > 0 else None),
                },
                "tokens": session_tokens,
                "rates": _token_rates(session_tokens, seg["think_generate_s"], seg["span_s"]),
            })
        rec = {
            "method": "arrival_stamps",
            "think_generate_s": round(think, 1),
            "tool_and_wait_s": round(tool, 1),
            "think_pct": round(100.0 * think / span, 1) if span > 0 else None,
            "measured_span_s": round(span, 1),
            "sessions": len(segs),
            "between_session_s": round(gap, 1),
            "tool_calls_matched": sum(s["tool_calls_matched"] for s in segs),
            "tool_calls_unterminated": sum(s["tool_calls_unterminated"] for s in segs),
            "tool_results_unpaired": sum(s["tool_results_unpaired"] for s in segs),
            "tool_call_seconds_sum": round(call_sum, 1),
            "tool_concurrency_overlap_s": round(max(call_sum - tool, 0.0), 1),
            "activity_share": {
                "think_generate": (think / span if span > 0 else None),
                "tool_and_wait": (tool / span if span > 0 else None),
                "basis": "one measured agent wall clock; shares sum to one",
            },
            "tokens": tokens,
            "rates": _token_rates(tokens, think, span),
            "tools": _tool_rollup(all_calls),
            "session_details": session_rows,
            "note": ("derived from per-event arrival stamps: a tool call occupies "
                     "[tool_use, tool_result]; tool_and_wait_s is the UNION of those intervals (tool "
                     "calls overlap when the driver backgrounds one), think_generate_s is the wall "
                     "time with none outstanding. Sessions are split at each system/init event so the "
                     "operator's between-round grading gap (between_session_s) is not agent time."),
        }
        cli = _from_duration_fields(events)
        if cli:
            # An INDEPENDENT reading, recorded beside the derived split and never merged into it. The
            # claude CLI's own fields split API LATENCY from everything else -- which is not the same
            # cut as tools-vs-thinking: the non-API remainder also holds CLI and harness overhead, so
            # on a measured run the two disagree (98 s of stamped tool intervals against 649 s of
            # non-API time). Reporting both is the only way that disagreement stays visible.
            rec["cli_reported"] = {
                "api_time_s": round(cli["think_generate_s"], 1),
                "non_api_time_s": round(cli["tool_and_wait_s"], 1),
                "total_time_s": round(cli["span_s"], 1),
                "note": ("the driver CLI's own duration_api_ms / duration_ms. API latency vs "
                         "everything else -- NOT tools vs thinking. Cross-check only."),
            }
        return rec
    legacy = _from_duration_fields(events)
    if legacy:
        span = legacy["span_s"]
        return {
            "method": "duration_api_ms",
            "think_generate_s": round(legacy["think_generate_s"], 1),
            "tool_and_wait_s": round(legacy["tool_and_wait_s"], 1),
            "think_pct": round(100.0 * legacy["think_generate_s"] / span, 1) if span > 0 else None,
            "measured_span_s": round(span, 1),
            "note": ("no arrival stamps in this transcript; fell back to the claude CLI's own "
                     "result.duration_api_ms vs result.duration_ms (last result event per session)."),
        }
    return {
        "method": "unknown",
        "think_generate_s": None,
        "tool_and_wait_s": None,
        "think_pct": None,
        "measured_span_s": None,
        "unavailable_reason": UNAVAILABLE_NOTE,
    }


# --- run-directory entry points -------------------------------------------------------------------

def transcript_paths(run_dir: Path) -> list[Path]:
    """The transcripts of one run, preferring the per-round files (they cannot interleave rounds).

    Falls back to the concatenated ``transcript.jsonl``, which ``decompose`` segments anyway.
    """
    run_dir = Path(run_dir)
    per_round = sorted((run_dir / "rounds").glob("round_*.transcript.jsonl"))
    if per_round:
        return per_round
    single = run_dir / "transcript.jsonl"
    return [single] if single.is_file() else []


def circt_gate(run_dir: Path) -> dict:
    """Prescreen-gate tally from the run's own gate log (absent log -> zeros, which is what it means)."""
    skips = ran = 0
    log = Path(run_dir) / "circt_gate_log.jsonl"
    if log.is_file():
        for line in log.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if not isinstance(rec, dict):
                continue
            skips += int(bool(rec.get("sim_skipped")))
            ran += int(not rec.get("sim_skipped"))
    return {"sims_skipped": skips, "sims_run": ran}


_TIMING_FIELDS = ("build_s", "sim_active_s", "oracle_wait_s", "adapter_wall_s")


def oracle_timing(run_dir: Path) -> dict:
    """Every L-tier invocation and a lossless-by-status aggregate.

    The source record's own ``engine`` is authoritative; tier names are fidelity levels, not simulator
    names.  Missing phase timings stay missing and mark sums as lower bounds.  Repeated capsule results
    are separate paid invocations and therefore intentionally remain separate rows.
    """
    run_dir = Path(run_dir)
    rows = []
    for result_path in sorted(run_dir.rglob("capsule_result.json")):
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        capsule = result.get("capsule") or result_path.parent.name
        for tier, tier_result in sorted((result.get("tiers") or {}).items()):
            if not isinstance(tier_result, dict):
                continue
            timing = tier_result.get("timing")
            timing = dict(timing) if isinstance(timing, dict) else None
            rows.append({
                "source": str(result_path.relative_to(run_dir)),
                "capsule": capsule,
                "tier": tier,
                "status": tier_result.get("status"),
                "engine": tier_result.get("engine"),
                "measured_now": tier_result.get("measured_now"),
                "timing": timing,
                "timing_complete": bool(timing and all(
                    isinstance(timing.get(field), (int, float)) for field in _TIMING_FIELDS)),
            })
    by_tier = {}
    for tier in sorted({row["tier"] for row in rows}):
        selected = [row for row in rows if row["tier"] == tier]
        timed = [row for row in selected if row["timing"] is not None]
        field_values = {field: [float(row["timing"][field]) for row in timed
                                if isinstance(row["timing"].get(field), (int, float))]
                        for field in _TIMING_FIELDS}
        adapter = field_values["adapter_wall_s"]
        by_tier[tier] = {
            "records": len(selected),
            "timed_records": len(timed),
            "complete_timing_records": sum(row["timing_complete"] for row in selected),
            "missing_timing_records": len(selected) - len(timed),
            "statuses": dict(sorted(Counter(str(row["status"]) for row in selected).items())),
            "engines": dict(sorted(Counter(str(row["engine"]) for row in selected
                                               if row["engine"]).items())),
            "totals_s": {field: (round(sum(values), 6) if values else None)
                         for field, values in field_values.items()},
            "fields_measured": {field: len(values) for field, values in field_values.items()},
            "totals_are_lower_bounds": any(len(values) < len(selected)
                                             for values in field_values.values()),
            "adapter_wall_distribution_s": {
                "mean": round(sum(adapter) / len(adapter), 6) if adapter else None,
                "p50": _percentile(adapter, 0.50),
                "p95": _percentile(adapter, 0.95),
                "max": max(adapter) if adapter else None,
            },
        }
    return {"capsule_result_files": len({row["source"] for row in rows}),
            "tier_records": len(rows), "by_tier": by_tier, "per_invocation": rows,
            "note": ("one row per capsule_result tier; repeated grading attempts remain separate paid "
                     "invocations. Null timings are unknown/not-run, never zero.")}


def _role(path: Path) -> str:
    name = path.name
    if "agent_evidence_snapshot" in str(path):
        return "authoritative_agent_visible_broker_or_selfcheck_evidence"
    if "codex_events.raw" in name:
        return "authoritative_provider_cli_stream"
    if "codex_events.timestamped" in name:
        return "arrival_stamped_provider_cli_stream"
    if "codex_rollout" in str(path):
        return "authoritative_codex_rollout_full_io_and_incremental_usage"
    if name.endswith(".prompt.txt"):
        return "exact_agent_input_prompt"
    if name.endswith(".final.txt"):
        return "exact_agent_final_output"
    if name.endswith(".transcript.jsonl"):
        return "normalized_transcript_outputs_may_be_clipped"
    if name.endswith(".codex_summary.json"):
        return "driver_session_summary"
    if name.endswith(".stage_ledger.json"):
        return "grader_stage_artifact_ledger"
    if name.endswith("stderr.log"):
        return "driver_stderr"
    if name.endswith(".sh"):
        return "sandbox_launch_script"
    return "run_telemetry_support"


def artifact_inventory(run_dir: Path) -> dict:
    """SHA/size inventory of every process-telemetry artifact (not the bulky simulator corpus)."""
    run_dir = Path(run_dir)
    candidates = []
    rounds = run_dir / "rounds"
    if rounds.is_dir():
        for path in sorted(rounds.rglob("*")):
            if path.is_file():
                candidates.append(path)
    evidence = run_dir / "agent_evidence_snapshot"
    if evidence.is_dir():
        candidates.extend(path for path in sorted(evidence.rglob("*")) if path.is_file())
    for name in ("transcript.jsonl", "environment.yaml", "cost_time_toolcalls.yaml",
                 "qa_loop_state.yaml", "qa_loop_summary.yaml"):
        path = run_dir / name
        if path.is_file():
            candidates.append(path)
    files = []
    for path in candidates:
        try:
            data = path.read_bytes()
        except OSError:
            continue
        files.append({"path": str(path.relative_to(run_dir)), "bytes": len(data),
                      "sha256": hashlib.sha256(data).hexdigest(), "role": _role(path)})
    return {
        "files": files,
        "authoritative_io": [row["path"] for row in files if row["role"].startswith("authoritative")
                             or row["role"].startswith("exact_agent")],
        "note": ("raw provider/rollout files are authoritative for full tool I/O; normalized transcripts "
                 "are an analysis view and may clip large outputs"),
    }


def _repo_from_run(run_dir: Path) -> Path | None:
    for parent in (run_dir, *run_dir.parents):
        if (parent / "out" / "artifacts" / "cache" / "codex_home").is_dir():
            return parent
    return None


def snapshot_codex_rollouts(run_dir: Path) -> list[Path]:
    """Copy the isolated Codex rollout logs into the durable run before cache cleanup.

    These logs are the only artifact containing incremental response usage plus full apply-patch bodies;
    the CLI's top-level event stream intentionally summarizes file changes.  No auth/state database is
    copied: only session JSONL.
    """
    run_dir = Path(run_dir)
    repo = _repo_from_run(run_dir)
    if repo is None:
        return []
    cache = repo / "out" / "artifacts" / "cache" / "codex_home"
    copied = []
    for home in sorted(cache.glob(f"{run_dir.name}_r*")):
        suffix = home.name.rsplit("_r", 1)[-1]
        dest = run_dir / "rounds" / f"round_{suffix}.codex_rollout_snapshot"
        for source in sorted((home / "sessions").rglob("*.jsonl")):
            relative = source.relative_to(home / "sessions")
            target = dest / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied.append(target)
    return copied


def snapshot_agent_evidence(run_dir: Path) -> list[Path]:
    """Seal agent-visible broker/self-check evidence that otherwise lives in a mutable workspace."""
    run_dir = Path(run_dir)
    repo = _repo_from_run(run_dir)
    if repo is None:
        return []
    copied = []
    workspaces = (repo / "merlin" / "experiments" / "capsule_bench" / "targets").glob(
        f"*/_qa_ws/{run_dir.name}/workspace")
    dest_root = run_dir / "agent_evidence_snapshot"
    for workspace in workspaces:
        for name in (".qa_channel", "selfcheck_out"):
            source = workspace / name
            if not source.exists():
                continue
            dest = dest_root / name
            if source.is_dir():
                shutil.copytree(source, dest, dirs_exist_ok=True)
                copied.extend(path for path in dest.rglob("*") if path.is_file())
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dest)
                copied.append(dest)
    return copied


def rollout_telemetry(run_dir: Path) -> dict:
    """Incremental Codex response usage and client-observed response intervals.

    Codex records a completion timestamp and per-response usage, but no provider request-start/TTFT or
    token-delta stream.  The interval begins at the preceding task-start/tool-result observed by the
    client.  It is deliberately called turnaround, never API latency or decode throughput.
    """
    paths = sorted((Path(run_dir) / "rounds").glob("round_*.codex_rollout_snapshot/**/*.jsonl"))
    responses = []
    total = {"input_total": 0, "fresh_input": 0, "cache_read": 0, "cache_write": 0,
             "output": 0, "reasoning": 0}
    for path in paths:
        pending_start = None
        try:
            records = [json.loads(line) for line in path.read_text(
                encoding="utf-8", errors="ignore").splitlines() if line.startswith("{")]
        except (OSError, ValueError):
            continue
        for record in records:
            stamp = _stamp(record.get("timestamp"))
            kind = record.get("type")
            payload = record.get("payload") or {}
            payload_type = payload.get("type") if isinstance(payload, dict) else None
            if kind == "event_msg" and payload_type == "task_started" and stamp is not None:
                pending_start = stamp
            elif (kind == "response_item" and payload_type in
                  {"custom_tool_call_output", "function_call_output"} and stamp is not None):
                pending_start = stamp
            elif kind == "token_usage_record" and isinstance(payload, dict):
                usage = payload.get("usage") or {}
                if not isinstance(usage, dict):
                    continue
                input_total = int(usage.get("input_tokens", 0) or 0)
                cache_read = int(usage.get("cached_input_tokens", 0) or 0)
                cache_write = int(usage.get("cache_write_input_tokens", 0) or 0)
                output = int(usage.get("output_tokens", 0) or 0)
                reasoning = int(usage.get("reasoning_output_tokens", 0) or 0)
                fresh = max(input_total - cache_read - cache_write, 0)
                elapsed = (max(stamp - pending_start, 0.0)
                           if stamp is not None and pending_start is not None else None)
                row = {
                    "source": str(path.relative_to(run_dir)),
                    "response_id": payload.get("response_id"),
                    "turn_id": payload.get("turn_id"),
                    "completed_at": record.get("timestamp"),
                    "client_observed_turnaround_s": round(elapsed, 6) if elapsed is not None else None,
                    "tokens": {"input_total": input_total, "fresh_input": fresh,
                               "cache_read": cache_read, "cache_write": cache_write,
                               "output": output, "reasoning": reasoning},
                    "output_tokens_per_client_observed_turnaround_s": (
                        output / elapsed if elapsed and elapsed > 0 else None),
                }
                responses.append(row)
                for key, value in row["tokens"].items():
                    total[key] += value
    turnaround = [row["client_observed_turnaround_s"] for row in responses
                  if row["client_observed_turnaround_s"] is not None]
    input_traffic = total["fresh_input"] + total["cache_read"] + total["cache_write"]
    response_wall = sum(turnaround) if turnaround else None
    return {
        "available": bool(responses),
        "rollout_files": [str(path.relative_to(run_dir)) for path in paths],
        "responses": responses,
        "tokens": {
            "tokens_input_total": total["input_total"],
            "tokens_fresh_input": total["fresh_input"],
            "tokens_cache_read": total["cache_read"],
            "tokens_cache_write": total["cache_write"],
            "tokens_output": total["output"],
            "tokens_reasoning": total["reasoning"],
            "tokens_total": total["input_total"] + total["output"],
            "cache_read_share_of_input": (total["cache_read"] / input_traffic
                                          if input_traffic else None),
            "cache_write_share_of_input": (total["cache_write"] / input_traffic
                                           if input_traffic else None),
            "reasoning_is_subset_of_output": True,
        } if responses else {},
        "client_observed_turnaround_s": {
            "sum": round(sum(turnaround), 6) if turnaround else None,
            "mean": round(sum(turnaround) / len(turnaround), 6) if turnaround else None,
            "p50": _percentile(turnaround, 0.50),
            "p95": _percentile(turnaround, 0.95),
            "max": max(turnaround) if turnaround else None,
        },
        "token_rates": {
            "output_tokens_per_client_observed_turnaround_s": (
                total["output"] / response_wall if response_wall else None),
            "all_provider_tokens_per_client_observed_turnaround_s": (
                (total["input_total"] + total["output"]) / response_wall
                if response_wall else None),
            "denominator_s": round(response_wall, 6) if response_wall else None,
            "note": "client-observed response turnaround; not server decode throughput",
        },
        "latency_limit": ("Codex exposes response-completion timestamps but not request-start, TTFT, or "
                          "token-delta timestamps. Turnaround is client-observed from the prior task/tool "
                          "input; it is not server latency, TTFT, or true decode tokens/s."),
    }


def stream_reconciliation(run_dir: Path) -> dict:
    """Prove that the arrival-stamped mirror contains every raw Codex event in the same order."""
    run_dir = Path(run_dir)
    rounds = run_dir / "rounds"
    rows, failures = [], []
    for raw_path in sorted(rounds.glob("round_*.codex_events.raw.jsonl")):
        stem = raw_path.name.replace(".codex_events.raw.jsonl", "")
        stamped_path = rounds / f"{stem}.codex_events.timestamped.jsonl"
        try:
            raw_bytes = raw_path.read_bytes()
            stamped_bytes = stamped_path.read_bytes()
            raw_lines = raw_bytes.splitlines()
            stamped_lines = stamped_bytes.splitlines()
            raw_events = [json.loads(line) for line in raw_lines]
            stamped = [json.loads(line) for line in stamped_lines]
            events_equal = (len(raw_events) == len(stamped)
                            and all(row.get("event") == event
                                    for row, event in zip(stamped, raw_events)))
            seq_contiguous = [row.get("seq") for row in stamped] == list(range(1, len(stamped) + 1))
            newline_terminated = raw_bytes.endswith(b"\n") and stamped_bytes.endswith(b"\n")
            row_failures = []
            if not events_equal:
                row_failures.append("raw_and_stamped_events_differ")
            if not seq_contiguous:
                row_failures.append("stamped_sequence_not_contiguous")
            if not newline_terminated:
                row_failures.append("stream_not_newline_terminated")
            rows.append({"round": stem, "raw_events": len(raw_events),
                         "stamped_events": len(stamped), "events_equal": events_equal,
                         "seq_contiguous": seq_contiguous,
                         "newline_terminated": newline_terminated,
                         "raw_sha256": hashlib.sha256(raw_bytes).hexdigest(),
                         "stamped_sha256": hashlib.sha256(stamped_bytes).hexdigest(),
                         "failures": row_failures})
            failures.extend(f"{stem}:{failure}" for failure in row_failures)
        except Exception as exc:  # noqa: BLE001 — corruption is an integrity result, not a crash
            failure = f"{stem}:reconciliation_failed:{type(exc).__name__}"
            failures.append(failure)
            rows.append({"round": stem, "failures": [failure]})
    if not rows:
        failures.append("no_raw_stream_to_reconcile")
    return {"complete": not failures, "failures": failures, "rounds": rows}


def resource_telemetry(run_dir: Path) -> dict:
    """Summarize retained procfs samples while keeping each JSONL as the source of truth."""
    paths = sorted((Path(run_dir) / "rounds").glob("round_*.resource_samples.jsonl"))
    paths += sorted((Path(run_dir) / "rounds").glob("round_*.live_resource_samples.jsonl"))
    records = []
    for path in paths:
        try:
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                    if line.startswith("{")]
        except (OSError, ValueError):
            rows = []
        stamps = [_stamp(row.get("sampled_at")) for row in rows]
        stamps = [stamp for stamp in stamps if stamp is not None]
        records.append({
            "source": str(path.relative_to(run_dir)), "samples": len(rows),
            "observed_span_s": (round(max(stamps) - min(stamps), 6) if len(stamps) >= 2 else 0.0),
            "rss_bytes_peak": max((int(row.get("rss_bytes", 0)) for row in rows), default=None),
            "rss_bytes_mean": (round(sum(int(row.get("rss_bytes", 0)) for row in rows) / len(rows), 3)
                               if rows else None),
            "virtual_bytes_peak": max((int(row.get("virtual_bytes", 0)) for row in rows), default=None),
            "processes_peak": max((int(row.get("processes", 0)) for row in rows), default=None),
            "threads_peak": max((int(row.get("threads", 0)) for row in rows), default=None),
            "observed_cpu_seconds_peak": max((float(row.get("user_cpu_s", 0))
                                               + float(row.get("system_cpu_s", 0)) for row in rows),
                                              default=None),
            "observed_read_bytes_peak": max((int(row.get("read_bytes", 0)) for row in rows), default=None),
            "observed_write_bytes_peak": max((int(row.get("write_bytes", 0)) for row in rows), default=None),
        })
    return {"available": any(row["samples"] for row in records), "streams": records,
            "sampling_note": ("5 s procfs snapshots of the live descendant tree. Peak cumulative CPU/I/O "
                              "is a lower bound because a short-lived child may exit between samples; raw "
                              "samples are retained for alternate analyses.")}


def _codex_round_integrity(run_dir: Path, timing: dict, tokens: dict,
                           reconciliation: dict, resources: dict) -> dict:
    """Completeness gate for finished Codex rounds; an active turn is explicitly incomplete."""
    failures = []
    rounds = Path(run_dir) / "rounds"
    stems = sorted({p.name.split(".transcript.jsonl")[0]
                    for p in rounds.glob("round_[0-9][0-9].transcript.jsonl")})
    details = []
    for stem in stems:
        required = {
            "prompt": rounds / f"{stem}.prompt.txt",
            "raw": rounds / f"{stem}.codex_events.raw.jsonl",
            "timestamped": rounds / f"{stem}.codex_events.timestamped.jsonl",
            "normalized": rounds / f"{stem}.transcript.jsonl",
            "final": rounds / f"{stem}.final.txt",
            "summary": rounds / f"{stem}.codex_summary.json",
        }
        missing = [name for name, path in required.items() if not path.is_file()]
        summary = {}
        if required["summary"].is_file():
            try:
                summary = json.loads(required["summary"].read_text(encoding="utf-8"))
            except (OSError, ValueError):
                missing.append("summary_readable")
        round_failures = [f"{stem}:missing_{name}" for name in missing]
        if summary and summary.get("usage_complete") is not True:
            round_failures.append(f"{stem}:usage_incomplete")
        if summary and summary.get("unknown_types"):
            round_failures.append(f"{stem}:unknown_driver_event_types")
        rollout = list(rounds.glob(f"{stem}.codex_rollout_snapshot/**/*.jsonl"))
        if not rollout:
            round_failures.append(f"{stem}:rollout_snapshot_missing")
        failures.extend(round_failures)
        details.append({"round": stem, "complete": not round_failures,
                        "failures": round_failures,
                        "turns_started": summary.get("turns_started"),
                        "turns_usage_reported": summary.get("turns_usage_reported"),
                        "driver_wall_s": summary.get("wall_s")})
    if not stems:
        failures.append("no_completed_round_transcript")
    if timing.get("method") == "unknown":
        failures.append("agent_timing_unavailable")
    if timing.get("tool_calls_unterminated"):
        failures.append("unterminated_tool_calls")
    if timing.get("tool_results_unpaired"):
        failures.append("unpaired_tool_results")
    if not tokens.get("available"):
        failures.append("provider_token_usage_unavailable")
    if reconciliation.get("complete") is not True:
        failures.extend(reconciliation.get("failures") or ["stream_reconciliation_incomplete"])
    if not any((Path(run_dir) / "agent_evidence_snapshot").rglob("*")):
        failures.append("agent_visible_evidence_snapshot_missing")
    if resources.get("available") is not True:
        failures.append("resource_samples_missing")
    return {"complete": not failures, "failures": failures, "rounds": details,
            "policy": ("formal completion requires exact prompt/final, raw+stamped+normalized streams, "
                       "sealed rollout, complete usage, known timing, and paired tools")}


def decompose_run(run_dir: Path) -> dict:
    """The unified process/oracle telemetry record for one run dir."""
    run_dir = Path(run_dir)
    paths = transcript_paths(run_dir)
    if not paths:
        rec = {"method": "unknown", "think_generate_s": None, "tool_and_wait_s": None,
               "think_pct": None, "measured_span_s": None,
               "unavailable_reason": f"no transcript found under {run_dir}"}
    else:
        rec = decompose(read_events(paths))
    rec["transcripts"] = [p.name for p in paths]
    rec["circt_gate"] = circt_gate(run_dir)
    rec["llm"] = rollout_telemetry(run_dir)
    rec["oracle"] = oracle_timing(run_dir)
    rec["artifacts"] = artifact_inventory(run_dir)
    rec["stream_reconciliation"] = stream_reconciliation(run_dir)
    rec["resources"] = resource_telemetry(run_dir)
    normalized_tokens = rec.get("tokens") or {}
    integrity_tokens = normalized_tokens
    if not integrity_tokens.get("available") and rec["llm"].get("available"):
        integrity_tokens = {"available": True, **rec["llm"]["tokens"]}
    rec["telemetry_integrity"] = _codex_round_integrity(
        run_dir, rec, integrity_tokens, rec["stream_reconciliation"], rec["resources"])
    rec["generated_at"] = datetime.now(timezone.utc).isoformat()
    rec["measurement_limits"] = {
        "provider_server_latency": "unavailable: provider request-start/TTFT is not emitted",
        "true_decode_token_rate": "unavailable: no per-token timestamp stream",
        "historical_cpu_rss_io": "unavailable unless sampled during the run; cannot be backfilled",
        "available_substitute": ("arrival-stamped agent/tool wall, client-observed response turnaround, "
                                 "provider token buckets, and exact L-tier phase timings"),
    }
    return rec


def write_run_timing(run_dir: Path) -> Path:
    """Seal volatile evidence and write ``<run_dir>/timing_detailed.json``."""
    run_dir = Path(run_dir)
    snapshot_codex_rollouts(run_dir)
    snapshot_agent_evidence(run_dir)
    out = run_dir / "timing_detailed.json"
    out.write_text(json.dumps(decompose_run(run_dir), indent=2))
    return out


def _fmt(v, unit="s"):
    return "UNKNOWN" if v is None else f"{v:g}{unit}"


def report_run(run_dir: Path, *, write: bool = False) -> dict:
    rec = decompose_run(run_dir)
    print(f"== {run_dir}")
    print(f"  method            : {rec['method']}")
    if rec.get("unavailable_reason"):
        print(f"  UNAVAILABLE       : {rec['unavailable_reason']}")
    print(f"  think+generate    : {_fmt(rec['think_generate_s'])}")
    print(f"  tool and wait     : {_fmt(rec['tool_and_wait_s'])}")
    print(f"  think share       : {_fmt(rec['think_pct'], '%')}")
    print(f"  measured span     : {_fmt(rec.get('measured_span_s'))}"
          f"  (sessions={rec.get('sessions')}, between={_fmt(rec.get('between_session_s'))})")
    if rec["method"] == "arrival_stamps":
        print(f"  tool calls        : {rec['tool_calls_matched']} matched, "
              f"{rec['tool_calls_unterminated']} unterminated, "
              f"{rec['tool_results_unpaired']} unpaired results")
        print(f"  sum of call durs  : {_fmt(rec['tool_call_seconds_sum'])} "
              f"(overlap {_fmt(rec['tool_concurrency_overlap_s'])} — concurrent tool calls)")
    if write:
        print(f"  wrote {write_run_timing(run_dir)}")
    return rec


# --- legacy cross-arm view ------------------------------------------------------------------------
# The original purpose of this file: the per-tier SIM cost the operator paid, alongside the agent split.
# Kept, but the agent split now comes from `decompose` so it is correct for every driver.

TIER_TOOL = {"L2": "spike", "L3": "verilator/VCS", "L4": "verilator/VCS"}


def harvest_arm(run_dir: Path, target: str) -> dict:
    """Agent split + EXACT per-tier sim wall for one arm's run dir."""
    import yaml

    run_dir = Path(run_dir)
    state = run_dir / "qa_loop_state.yaml"
    st = yaml.safe_load(state.read_text()) if state.is_file() else {}
    active = ((st or {}).get("cumulative") or {}).get("active_wall_s", 0.0)
    sims = {"spike": {"runs": 0, "build_s": 0.0, "sim_s": 0.0},
            "verilator/VCS": {"runs": 0, "build_s": 0.0, "sim_s": 0.0}}
    for cr in (run_dir / "_qa_work").glob(f"runs_*/runs/{target}-capsule-bench/*/capsule_result.json"):
        try:
            r = json.loads(cr.read_text())
        except ValueError:
            continue
        for tier, tv in (r.get("tiers") or {}).items():
            tm = (tv or {}).get("timing") or {}
            tool = TIER_TOOL.get(tier)
            if tool and tm:
                sims[tool]["runs"] += 1
                sims[tool]["build_s"] += tm.get("build_s") or 0.0
                sims[tool]["sim_s"] += tm.get("sim_active_s") or 0.0
    return {"active_wall_min": round(active / 60, 1),
            "agent_session": decompose_run(run_dir),
            "tool_wall_exact": {
                tool: {"runs": v["runs"], "total_s": round(v["sim_s"] + v["build_s"], 2),
                       "per_run_s": round((v["sim_s"] + v["build_s"]) / max(v["runs"], 1), 3)}
                for tool, v in sims.items()}}


def _legacy_arms(arm_runs: dict[str, tuple[str, str]]) -> int:
    import _common as C  # noqa: PLC0415 — bootstraps sys.path; kept out of import time so this
                         # module stays unit-testable without the experiment env.
    out_dir = C.REPORTS / "timing"
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {label: harvest_arm(C.RUNS / sub / rid, C.TARGET) for rid, (sub, label) in arm_runs.items()}
    (out_dir / "timing_detailed.json").write_text(json.dumps(res, indent=2))
    for label, t in res.items():
        a = t["agent_session"]
        print(f"  {label:14s} active={t['active_wall_min']:>7} min  think={_fmt(a['think_generate_s'])} "
              f"tool={_fmt(a['tool_and_wait_s'])} ({a['method']})")
    print(f"wrote {out_dir}/timing_detailed.json")
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", action="append", default=[],
                    help="a run directory (repeatable); prints its think/tool split")
    ap.add_argument("--write", action="store_true", help="also write <run-dir>/timing_detailed.json")
    ap.add_argument("--arms", action="store_true", help="legacy cross-arm view (needs the experiment env)")
    ap.add_argument("--arm", action="append", default=[], metavar="RUN_ID=SUBDIR:LABEL",
                    help="arm to include in --arms")
    args = ap.parse_args(argv)
    if args.run_dir:
        for d in args.run_dir:
            report_run(Path(d), write=args.write)
        return 0
    if args.arms:
        arms = {}
        for spec in args.arm:
            rid, _, rest = spec.partition("=")
            sub, _, label = rest.partition(":")
            arms[rid] = (sub, label or rid)
        if not arms:
            ap.error("--arms needs at least one --arm RUN_ID=SUBDIR:LABEL")
        return _legacy_arms(arms)
    ap.error("give --run-dir DIR (or --arms with --arm specs)")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
