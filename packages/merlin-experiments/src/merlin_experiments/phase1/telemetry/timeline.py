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

Run reporting and its CLI live in ``merlin_experiments.phase1.telemetry.report``.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import UTC, datetime
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
                    out.append(
                        {
                            "kind": OPEN,
                            "call_id": str(block.get("id") or ""),
                            "name": str(block.get("name") or "unknown"),
                            "input_chars": _json_chars(block.get("input")),
                        }
                    )
                elif bt == _BLOCK_TOOL_RESULT:
                    out.append(
                        {
                            "kind": CLOSE,
                            "call_id": str(block.get("tool_use_id") or ""),
                            "error": bool(block.get("is_error")),
                            "output_chars": _json_chars(block.get("content")),
                        }
                    )
    inner = evt.get("event") if isinstance(evt.get("event"), dict) else evt
    etype, item = inner.get("type"), inner.get("item")
    if etype in (_RAW_ITEM_STARTED, _RAW_ITEM_COMPLETED) and isinstance(item, dict):
        if item.get("type") in _RAW_TOOL_ITEMS:
            rec = {
                "kind": OPEN if etype == _RAW_ITEM_STARTED else CLOSE,
                "call_id": str(item.get("id") or ""),
                "name": str(item.get("type") or "unknown"),
            }
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
    totals = {"fresh_input": 0, "cache_write": 0, "cache_read": 0, "output": 0, "reasoning": 0}
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
        "cache_read_share_of_input": (
            totals["cache_read"] / input_traffic if usage_messages and input_traffic else None
        ),
        "cache_write_share_of_input": (
            totals["cache_write"] / input_traffic if usage_messages and input_traffic else None
        ),
        "reasoning_is_subset_of_output": True,
        **(
            {}
            if usage_messages
            else {"unavailable_reason": "no provider usage event has arrived; tokens are unknown, not zero"}
        ),
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
    return {
        "used": sorted(grouped),
        "by_tool": by_tool,
        "calls": calls,
        "exact_io_source": (
            "the raw provider event stream; normalized tool outputs may be clipped "
            "for prompt safety, so this report stores sizes and hashes, not a second copy"
        ),
    }


def _token_rates(tokens: dict, think_s: float | None, span_s: float | None) -> dict:
    """Derived throughput with denominators in every field name.

    Output/think time is useful as an activity-normalized rate, but is NOT provider decode throughput:
    arrival stamps bracket messages and tools, not first-token and last-token network events.
    """
    output = tokens.get("tokens_output")
    total = tokens.get("tokens_total")
    return {
        "output_tokens_per_think_generate_s": (
            output / think_s if output is not None and think_s and think_s > 0 else None
        ),
        "output_tokens_per_agent_span_s": (output / span_s if output is not None and span_s and span_s > 0 else None),
        "total_tokens_per_agent_span_s": (total / span_s if total is not None and span_s and span_s > 0 else None),
        "note": (
            "derived from transcript arrival intervals; activity-normalized throughput, not a "
            "provider-side first-token/decode latency measurement"
        ),
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
                open_at.setdefault(
                    call_id,
                    {
                        "start": t,
                        "name": boundary.get("name") or "unknown",
                        "input_chars": boundary.get("input_chars", 0),
                    },
                )
            elif call_id in open_at:
                opened = open_at.pop(call_id)
                start = opened["start"]
                intervals.append((start, t))
                calls.append(
                    {
                        "call_id": call_id,
                        "name": opened["name"],
                        "started_at": datetime.fromtimestamp(start, UTC).isoformat(),
                        "ended_at": datetime.fromtimestamp(t, UTC).isoformat(),
                        "duration_s": round(max(t - start, 0.0), 6),
                        "completed": True,
                        "error": bool(boundary.get("error")),
                        "input_chars": opened.get("input_chars", 0),
                        "output_chars": boundary.get("output_chars", 0),
                    }
                )
            else:
                unmatched_close += 1
    # A tool call whose result never arrived (the round was cut off mid-command) OCCUPIED the clock up
    # to the last thing we saw. Clamping to the segment end is the only reading the stamps support;
    # dropping it would silently move that wall time into "thinking".
    still_open = len(open_at)
    for call_id, opened in open_at.items():
        start = opened["start"]
        intervals.append((start, t_end))
        calls.append(
            {
                "call_id": call_id,
                "name": opened["name"],
                "started_at": datetime.fromtimestamp(start, UTC).isoformat(),
                "ended_at": None,
                "duration_s": round(max(t_end - start, 0.0), 6),
                "completed": False,
                "error": None,
                "input_chars": opened.get("input_chars", 0),
                "output_chars": None,
            }
        )
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
    return {
        "think_generate_s": api_ms / 1000.0,
        "tool_and_wait_s": max(0.0, total_ms - api_ms) / 1000.0,
        "span_s": total_ms / 1000.0,
    }


UNAVAILABLE_NOTE = (
    "the transcript carries no per-event arrival stamps and no non-zero duration_api_ms/duration_ms, "
    "so the think-vs-tool split is NOT MEASURED for this run. It is recorded as null, never 0.0: a "
    "zero here would read as 'the agent spent no time thinking' and would be averaged into a study."
)


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
            session_rows.append(
                {
                    "session": index,
                    "started_at": datetime.fromtimestamp(seg["first_s"], UTC).isoformat(),
                    "ended_at": datetime.fromtimestamp(seg["last_s"], UTC).isoformat(),
                    "wall_s": round(seg["span_s"], 6),
                    "think_generate_s": round(seg["think_generate_s"], 6),
                    "tool_and_wait_s": round(seg["tool_and_wait_s"], 6),
                    "activity_share": {
                        "think_generate": (seg["think_generate_s"] / seg["span_s"] if seg["span_s"] > 0 else None),
                        "tool_and_wait": (seg["tool_and_wait_s"] / seg["span_s"] if seg["span_s"] > 0 else None),
                    },
                    "tokens": session_tokens,
                    "rates": _token_rates(session_tokens, seg["think_generate_s"], seg["span_s"]),
                }
            )
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
            "note": (
                "derived from per-event arrival stamps: a tool call occupies "
                "[tool_use, tool_result]; tool_and_wait_s is the UNION of those intervals (tool "
                "calls overlap when the driver backgrounds one), think_generate_s is the wall "
                "time with none outstanding. Sessions are split at each system/init event so the "
                "operator's between-round grading gap (between_session_s) is not agent time."
            ),
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
                "note": (
                    "the driver CLI's own duration_api_ms / duration_ms. API latency vs "
                    "everything else -- NOT tools vs thinking. Cross-check only."
                ),
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
            "note": (
                "no arrival stamps in this transcript; fell back to the claude CLI's own "
                "result.duration_api_ms vs result.duration_ms (last result event per session)."
            ),
        }
    return {
        "method": "unknown",
        "think_generate_s": None,
        "tool_and_wait_s": None,
        "think_pct": None,
        "measured_span_s": None,
        "unavailable_reason": UNAVAILABLE_NOTE,
    }
