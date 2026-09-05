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
from datetime import datetime
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

def _marks(evt: dict) -> list[tuple[str, str]]:
    """The tool-call boundaries this event announces, as ``(OPEN|CLOSE, call_id)``.

    Recognises the harness's normalized claude-shaped events AND a raw ``codex exec --json`` envelope,
    so the same timeline algebra reads either file. An event in neither shape yields nothing (it is a
    timeline TICK, not a tool boundary) rather than being guessed at.
    """
    out: list[tuple[str, str]] = []
    msg = evt.get("message")
    if isinstance(msg, dict):
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                bt = block.get("type")
                if bt == _BLOCK_TOOL_USE:
                    out.append((OPEN, str(block.get("id") or "")))
                elif bt == _BLOCK_TOOL_RESULT:
                    out.append((CLOSE, str(block.get("tool_use_id") or "")))
    inner = evt.get("event") if isinstance(evt.get("event"), dict) else evt
    etype, item = inner.get("type"), inner.get("item")
    if etype in (_RAW_ITEM_STARTED, _RAW_ITEM_COMPLETED) and isinstance(item, dict):
        if item.get("type") in _RAW_TOOL_ITEMS:
            out.append((OPEN if etype == _RAW_ITEM_STARTED else CLOSE, str(item.get("id") or "")))
    return out


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


def _decompose_segment(events: list[dict]) -> dict | None:
    """One agent session -> its span, its tool-busy union, its think time. None if unstamped."""
    stamped: list[tuple[float, list[tuple[str, str]]]] = []
    for evt in events:
        t = _event_time(evt)
        if t is None:
            continue
        stamped.append((t, _marks(evt)))
    if len(stamped) < 2:
        return None
    t0, t_end = stamped[0][0], stamped[-1][0]
    open_at: dict[str, float] = {}
    intervals: list[tuple[float, float]] = []
    unmatched_close = 0
    for t, marks in stamped:
        for kind, call_id in marks:
            if kind == OPEN:
                open_at.setdefault(call_id, t)
            elif call_id in open_at:
                intervals.append((open_at.pop(call_id), t))
            else:
                unmatched_close += 1
    # A tool call whose result never arrived (the round was cut off mid-command) OCCUPIED the clock up
    # to the last thing we saw. Clamping to the segment end is the only reading the stamps support;
    # dropping it would silently move that wall time into "thinking".
    still_open = len(open_at)
    for start in open_at.values():
        intervals.append((start, t_end))
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
    segs = [s for s in (_decompose_segment(g) for g in _segments(events)) if s]
    if segs:
        think = sum(s["think_generate_s"] for s in segs)
        tool = sum(s["tool_and_wait_s"] for s in segs)
        span = sum(s["span_s"] for s in segs)
        gap = 0.0
        for prev, nxt in zip(segs, segs[1:]):
            gap += max(nxt["first_s"] - prev["last_s"], 0.0)
        call_sum = sum(s["tool_call_seconds_sum"] for s in segs)
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


def decompose_run(run_dir: Path) -> dict:
    """The full ``timing_detailed.json`` record for one run dir."""
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
    return rec


def write_run_timing(run_dir: Path) -> Path:
    """Write ``<run_dir>/timing_detailed.json``. Call this at run finish, or to backfill."""
    run_dir = Path(run_dir)
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
