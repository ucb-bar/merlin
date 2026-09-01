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


#: Markers a driver leaves when its turn did not run at all, as opposed to running and achieving little.
#: Kept as substrings of the ERROR text rather than as exact messages, because each provider words its own
#: refusal, and matched only alongside "no tool work" so a real turn that merely mentions one of these
#: words is never mistaken for a dead one.
_TERMINAL_MARKERS = (
    "usage limit",             # ChatGPT/codex seat credits exhausted (carries a retry DATE, not a window)
    "purchase more credits",
    "issue with the selected model",   # a model the CLI cannot serve (e.g. a Bedrock id under a seat)
    "may not exist or you may not have access",
    "authentication",
    "invalid api key",
    "unauthorized",
)


def agent_turn_dead(transcript_path: str | Path) -> tuple[bool, str]:
    """``(dead, reason)`` — did this round's agent turn fail to RUN, rather than run and accomplish little?

    This is the third failure class, and the one that had no guard. ``round_rejected`` covers the
    five-hour window and ``daily_limit_hit`` the provider's daily token quota; neither covers a turn that
    never happened because the seat is out of credits until a DATE, because the model name cannot be
    served, or because auth failed. Measured 2026-09-01, both on this bench:

      * the codex seat returned "You've hit your usage limit ... try again at Sep 6th" with
        ``content: []`` and a 4.5 s turn, for three consecutive rounds;
      * a Bedrock inference-profile id handed to a subscription CLI returned "There's an issue with the
        selected model (us.anthropic.claude-opus-4-6-v1)" in 0 ms.

    In BOTH cases the round went on to grade an unchanged submission and print
    ``NOT CONFORMANT -- failing: isa_tools_used, cca_used, ...``. That is a harness limitation reported as
    an agent defect, and roughly three and a half hours of wall-clock were spent on rounds where no agent
    ran. A verdict about an agent that did not run is not a verdict.

    Discriminated by NO TOOL WORK plus positive evidence of a terminal failure -- either a marker above,
    an explicitly failed turn, or a reply that came only from the CLI itself (``<synthetic>``). A quiet but
    real turn is left alone: it is unproductive, which the stage ledger already reports, not dead.
    """
    tool_uses = 0
    reasons: list[str] = []
    synthetic_only = True
    saw_assistant = False
    for e in _iter_events(transcript_path):
        t = e.get("type")
        msg = e.get("message") or {}
        if t == "assistant":
            saw_assistant = True
            if isinstance(msg, dict) and str(msg.get("model", "")) not in ("<synthetic>", ""):
                synthetic_only = False
            if e.get("codex_turn_failed") in (True, "True"):
                reasons.append("the driver reported the turn itself as failed")
        txt = str(e.get("result", "")) if t == "result" else ""
        if isinstance(msg, dict):
            for b in (msg.get("content") or []):
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "tool_use":
                    tool_uses += 1
                elif b.get("type") == "text":
                    txt += " " + str(b.get("text", ""))
        low = txt.lower()
        for m in _TERMINAL_MARKERS:
            if m in low:
                reasons.append(f"the driver reported {m!r}")
                break
    if tool_uses:
        return False, ""                     # the agent did work; whatever else happened, it ran
    if saw_assistant and synthetic_only:
        reasons.append("every reply came from the CLI itself (model '<synthetic>'), so no model ran")
    if not saw_assistant:
        reasons.append("the transcript carries no assistant turn at all")
    if not reasons:
        return False, ""
    # de-duplicate while keeping order, so a repeated provider message is stated once
    seen: list[str] = []
    for r in reasons:
        if r not in seen:
            seen.append(r)
    return True, "; ".join(seen)
