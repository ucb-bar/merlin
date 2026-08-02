"""Shared rate-limit detection for the capsule_bench_v0 QA loop + reporting.

The claude CLI, when the org five-hour session budget is exhausted, emits a `rate_limit_event` with
`rate_limit_info.status == "rejected"` and `rateLimitType == "five_hour"` (and a `result` is_error
text containing "session limit"). The agent then does ZERO tool work for that round. This module
centralizes detecting that condition and reading the `resetsAt` epoch, so:
  - reclassify_repeatability.py / gen_reports.py classify such runs as blocked (not failed), and
  - run_baseline_qa_loop.py can sleep until `resetsAt` and retry the round instead of burning it.
"""
from __future__ import annotations

import json
from pathlib import Path


def _iter_events(transcript_path: str | Path):
    p = Path(transcript_path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        try:
            yield json.loads(line)
        except Exception:
            continue


def round_rejected(transcript_path: str | Path) -> bool:
    """True if this round transcript shows a five-hour rate-limit REJECTION with no tool work."""
    rej = False
    tool_uses = 0
    for e in _iter_events(transcript_path):
        t = e.get("type")
        if t == "rate_limit_event":
            ri = e.get("rate_limit_info", {})
            if ri.get("status") == "rejected" and ri.get("rateLimitType") == "five_hour":
                rej = True
        elif t == "result":
            if e.get("is_error") and "session limit" in str(e.get("result", "")).lower():
                rej = True
        elif t == "assistant":
            for b in e.get("message", {}).get("content", []):
                if b.get("type") == "tool_use":
                    tool_uses += 1
    return rej and tool_uses == 0


def daily_limit_hit(transcript_path: str | Path) -> bool:
    """True if this round hit a provider DAILY token quota — a 429 'too many tokens per day' API error
    (e.g. Bedrock) with no tool work. Unlike the five-hour window this has no short ``resetsAt`` to sleep
    to, so the loop should abort the run early rather than burn every remaining round against the wall."""
    hit = False
    tool_uses = 0
    for e in _iter_events(transcript_path):
        t = e.get("type")
        if t in ("result", "assistant"):
            txt = str(e.get("result", "")) if t == "result" else ""
            for b in e.get("message", {}).get("content", []):
                if b.get("type") == "text":
                    txt += " " + str(b.get("text", ""))
                elif b.get("type") == "tool_use":
                    tool_uses += 1
            low = txt.lower()
            if ("per day" in low or "daily" in low) and \
                    ("429" in low or "too many" in low or "quota" in low or "limit" in low):
                hit = True
    return hit and tool_uses == 0


def rate_limit_reset_epoch(transcript_path: str | Path) -> int | None:
    """Return the `resetsAt` epoch (seconds) from a rejected five-hour event, or None."""
    latest = None
    for e in _iter_events(transcript_path):
        if e.get("type") == "rate_limit_event":
            ri = e.get("rate_limit_info", {})
            if ri.get("status") == "rejected" and ri.get("rateLimitType") == "five_hour":
                ra = ri.get("resetsAt")
                if isinstance(ra, (int, float)):
                    latest = int(ra)
    return latest


def rounds_rate_limited(run_dir: str | Path) -> tuple[int, int]:
    """(#rounds rejected by five-hour limit with zero work, #rounds that did real tool work)."""
    rejected = 0
    worked = 0
    rdir = Path(run_dir) / "rounds"
    if not rdir.exists():
        return (0, 0)
    for tp in sorted(rdir.glob("round_*.transcript.jsonl")):
        rej = False
        tu = 0
        for e in _iter_events(tp):
            t = e.get("type")
            if t == "rate_limit_event":
                ri = e.get("rate_limit_info", {})
                if ri.get("status") == "rejected" and ri.get("rateLimitType") == "five_hour":
                    rej = True
            elif t == "result":
                if e.get("is_error") and "session limit" in str(e.get("result", "")).lower():
                    rej = True
            elif t == "assistant":
                for b in e.get("message", {}).get("content", []):
                    if b.get("type") == "tool_use":
                        tu += 1
        if rej and tu == 0:
            rejected += 1
        elif tu > 0:
            worked += 1
    return (rejected, worked)
