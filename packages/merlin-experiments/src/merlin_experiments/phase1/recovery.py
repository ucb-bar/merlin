"""Transcript-derived interruption detection and checkpoint/resume policy.

One owner serves the functional controller and reporting. Classification preserves
weekly, daily, five-hour and timeout precedence; provider failures are not grades.
No target discovery, process launch or environment mutation occurs on import.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path


def _iter_events(transcript_path: str | Path, *, strict: bool = False):
    p = Path(transcript_path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except Exception:
            if strict:
                raise
            continue
        if strict:
            if not isinstance(event, dict):
                raise ValueError("authoring transcript event must be a JSON object")
            # User prompts and system telemetry may carry plain string messages.
            # Only assistant/result messages are traversed as blocks by quota
            # predicates; the dead-turn predicate tolerates other event shapes.
            if event.get("type") in {"assistant", "result"}:
                message = event.get("message", {})
                if not isinstance(message, dict):
                    raise ValueError("authoring transcript message must be a JSON object")
                content = message.get("content", [])
                if not isinstance(content, list) or any(not isinstance(block, dict) for block in content):
                    raise ValueError("authoring transcript message.content must be a list of JSON objects")
            if event.get("type") == "rate_limit_event" and not isinstance(event.get("rate_limit_info", {}), dict):
                raise ValueError("authoring transcript rate_limit_info must be a JSON object")
        yield event


def round_rejected(transcript_path: str | Path) -> bool:
    """True if this round transcript shows a five-hour rate-limit REJECTION with no tool work."""
    return _round_rejected(_iter_events(transcript_path))


def _round_rejected(events: Iterable[dict]) -> bool:
    rej = False
    tool_uses = 0
    for e in events:
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
    return _daily_limit_hit(_iter_events(transcript_path))


def _daily_limit_hit(events: Iterable[dict]) -> bool:
    hit = False
    tool_uses = 0
    for e in events:
        t = e.get("type")
        if t in ("result", "assistant"):
            txt = str(e.get("result", "")) if t == "result" else ""
            for b in e.get("message", {}).get("content", []):
                if b.get("type") == "text":
                    txt += " " + str(b.get("text", ""))
                elif b.get("type") == "tool_use":
                    tool_uses += 1
            low = txt.lower()
            if ("per day" in low or "daily" in low) and (
                "429" in low or "too many" in low or "quota" in low or "limit" in low
            ):
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
    "usage limit",  # ChatGPT/codex seat credits exhausted (carries a retry DATE, not a window)
    "purchase more credits",
    "issue with the selected model",  # a model the CLI cannot serve (e.g. a Bedrock id under a seat)
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
    return _agent_turn_dead(_iter_events(transcript_path))


def _agent_turn_dead(events: Iterable[dict], *, assistant_tools_only: bool = False) -> tuple[bool, str]:
    tool_uses = 0
    reasons: list[str] = []
    synthetic_only = True
    saw_assistant = False
    for e in events:
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
            for b in msg.get("content") or []:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "tool_use" and (not assistant_tools_only or t == "assistant"):
                    tool_uses += 1
                elif b.get("type") == "text":
                    txt += " " + str(b.get("text", ""))
        low = txt.lower()
        for m in _TERMINAL_MARKERS:
            if m in low:
                reasons.append(f"the driver reported {m!r}")
                break
    if tool_uses:
        return False, ""  # the agent did work; whatever else happened, it ran
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


# Reasons a round can be "cut short" (a productive round that did not get to finish/grade cleanly).
REASON_TIMEOUT = "timeout"  # wall-clock (rc=124) — transcript killed mid-stream
REASON_FIVE_HOUR = "rate_limit_five_hour"  # short org window rejection (zero work)
REASON_WEEKLY = "quota_weekly"  # seven-day subscription budget exhausted
REASON_DAILY = "quota_daily"  # provider daily token quota exhausted

# Policy verbs.
RESUME_IN_BUDGET = "resume"  # keep going: the partial submission is finished/continued within budget
EXIT_WITH_STATUS = "exit"  # stop now: the quota resets far in the future; resume after reset
NO_POLICY = "none"  # the round was NOT cut short

# Machine-readable boundaries let the round brief refresh mutable operator guidance immediately before
# an agent relaunch without dropping (or duplicating) this operator-critical banner.
RESUME_NOTE_BEGIN = "<!-- merlin:resume-note:begin -->"
RESUME_NOTE_END = "<!-- merlin:resume-note:end -->"

# Distinct process exit code + status token for a weekly-quota stop (kept out of the 2..5 range the
# loop already uses for refuse-overwrite / isolation / preflight / golden-mask failures).
QUOTA_WEEKLY_EXIT_CODE = 42
STATUS_WEEKLY = "QUOTA_EXHAUSTED_WEEKLY"

_POLICY = {
    REASON_TIMEOUT: RESUME_IN_BUDGET,
    REASON_FIVE_HOUR: RESUME_IN_BUDGET,
    REASON_WEEKLY: EXIT_WITH_STATUS,
    REASON_DAILY: EXIT_WITH_STATUS,
}


def resume_policy(reason: str) -> str:
    """Map a cut-short ``reason`` to its policy verb (``resume`` / ``exit`` / ``none``)."""
    return _POLICY.get(reason, NO_POLICY)


def _latest_transcript(run_dir: str | Path) -> Path | None:
    """Highest-index ``rounds/round_*.transcript.jsonl`` under ``run_dir`` (or None)."""
    rdir = Path(run_dir) / "rounds"
    if not rdir.is_dir():
        return None
    tps = sorted(rdir.glob("round_*.transcript.jsonl"))
    return tps[-1] if tps else None


def _has_result_event(transcript_path: str | Path) -> bool:
    """True if the transcript carries a terminating ``{"type":"result"}`` event. A round the driver ran
    to completion (converged, failed, OR rate-limited) always emits one; a SIGKILLed (timeout/OOM) round
    ends mid-stream without it."""
    for e in _iter_events(transcript_path):
        if e.get("type") == "result":
            return True
    return False


def weekly_quota_hit(transcript_path: str | Path) -> bool:
    """True if this round hit the WEEKLY (seven-day) subscription budget.

    Detected structurally from either signal the driver emits:
      * a ``rate_limit_event`` with ``rate_limit_info.rateLimitType == "seven_day"`` and a rejected
        status (``status``/``overageStatus`` == "rejected"), or
      * a terminal ``result`` with ``terminal_reason == "api_error"`` + ``api_error_status == 429``
        whose message mentions a weekly limit.
    """
    return _weekly_quota_hit(_iter_events(transcript_path))


def _weekly_quota_hit(events: Iterable[dict]) -> bool:
    for e in events:
        t = e.get("type")
        if t == "rate_limit_event":
            ri = e.get("rate_limit_info", {}) or {}
            if ri.get("rateLimitType") == "seven_day" and (
                ri.get("status") == "rejected" or ri.get("overageStatus") == "rejected"
            ):
                return True
        elif t == "result":
            if e.get("terminal_reason") == "api_error" and int(e.get("api_error_status") or 0) == 429:
                if "weekly" in str(e.get("result", "")).lower():
                    return True
    return False


def weekly_reset_epoch(transcript_path: str | Path) -> int | None:
    """The ``resetsAt`` epoch (seconds) from a seven-day rejection event, or None."""
    latest = None
    for e in _iter_events(transcript_path):
        if e.get("type") == "rate_limit_event":
            ri = e.get("rate_limit_info", {}) or {}
            if ri.get("rateLimitType") == "seven_day":
                ra = ri.get("resetsAt")
                if isinstance(ra, (int, float)):
                    latest = int(ra)
    return latest


def latest_live_authoring_transcript(transcripts: Iterable[str | Path]) -> Path | None:
    """Select lexical latest authoring evidence, not a later transport-only failure.

    A live but unproductive model turn remains evidence; conformance decides its
    verdict. Tool work followed by a terminal/quota failure also remains evidence.
    No eligible transcript means no evidence, never a fabricated conformant round.
    This selection is not the operator-seal policy: that uses its checkpoint's
    explicitly completed round. Unreadable evidence propagates rather than silently
    falling back to an older transcript.
    Classification reads each candidate once; the returned path is not a frozen copy.
    """
    for path in sorted({Path(path) for path in transcripts}, reverse=True):
        events = tuple(_iter_events(path, strict=True))
        if (
            _agent_turn_dead(events, assistant_tools_only=True)[0]
            or _daily_limit_hit(events)
            or _round_rejected(events)
        ):
            continue
        if _weekly_quota_hit(events):
            # Unlike daily/five-hour guards, weekly_quota_hit also recognizes a
            # productive turn interrupted after tool work. Keep that evidence.
            tool_work = any(
                isinstance(block, dict) and block.get("type") == "tool_use"
                for event in events
                if event.get("type") == "assistant"
                for block in (event.get("message") or {}).get("content") or []
            )
            if not tool_work:
                continue
        return path
    return None


def classify(transcript_path: str | Path, rc: int | None = None) -> str:
    """Classify ONE round transcript into a cut-short reason (or "" if it finished cleanly).

    Order is most-specific-first so a rate-limited round is named by its limit, never mislabeled a
    timeout: weekly -> daily -> five-hour -> timeout.
    """
    if weekly_quota_hit(transcript_path):
        return REASON_WEEKLY
    if daily_limit_hit(transcript_path):
        return REASON_DAILY
    if round_rejected(transcript_path):
        return REASON_FIVE_HOUR
    # Timeout: the driver was SIGKILLed (rc=124) OR the transcript has no terminating result event.
    if rc == 124 or not _has_result_event(transcript_path):
        return REASON_TIMEOUT
    return ""


def round_was_cut_short(
    run_dir: str | Path, rc: int | None = None, transcript: str | Path | None = None
) -> tuple[bool, str]:
    """Was the latest round cut short before finishing? Returns ``(was_cut, reason)``.

    ``transcript`` overrides the run_dir lookup (the loop already holds the path). An empty/absent
    transcript with ``rc == 124`` still classifies as a timeout.
    """
    tp = Path(transcript) if transcript is not None else _latest_transcript(run_dir)
    if tp is None:
        return (rc == 124, REASON_TIMEOUT if rc == 124 else "")
    reason = classify(tp, rc=rc)
    return (bool(reason), reason)


def resume_note(reason: str) -> str:
    """A RESUME banner to prepend to the next round's brief when a partial submission is being continued.
    Tells the fresh session its work is preserved and to finish the incomplete pieces FIRST."""
    return (
        f"> ## RESUME — your previous round was cut short ({reason})\n"
        f"{RESUME_NOTE_BEGIN}\n"
        f"> Your previous `submission/` is PRESERVED on disk — do NOT start over or re-derive it.\n"
        f"> You were interrupted mid-work. FINISH the incomplete pieces FIRST, before any refinement:\n"
        f">   1. `submission/manifest.yaml` (the run scores 0 without it — write/repair it first),\n"
        f">   2. the 4 CLI entrypoints your manifest declares (make the tool runnable end-to-end),\n"
        f">   3. the target artifact (e.g. `kernel.S` / the emitted program) for each capsule.\n"
        f"> Only once the submission is complete and gradeable should you iterate on correctness.\n"
        f"{RESUME_NOTE_END}\n\n"
    )


def prepend_resume_note(ws: str | Path, reason: str) -> Path:
    """Prepend the RESUME banner to ``ws/qa/round_brief.md`` (written by ``round_brief.write`` just
    before), so the next fresh session reads it at round start. Best-effort; returns the brief path."""
    qa = Path(ws) / "qa"
    qa.mkdir(parents=True, exist_ok=True)
    brief = qa / "round_brief.md"
    existing = brief.read_text(encoding="utf-8") if brief.is_file() else ""
    brief.write_text(resume_note(reason) + existing, encoding="utf-8")
    return brief


def write_quota_status(run_dir: str | Path, reason: str, rnd: int, transcript: str | Path | None = None) -> Path:
    """Persist a distinct ``run_dir/quota_status.yaml`` for a quota-exhausted stop, so the run is
    recognizable as blocked-not-failed and an operator knows when to relaunch. Returns the file path.
    (No PyYAML dependency here — a tiny flat dump keeps this module import-light.)"""
    import yaml  # local import: yaml is always available in the harness runtime

    reset = weekly_reset_epoch(transcript) if (reason == REASON_WEEKLY and transcript) else None
    doc = {
        "status": STATUS_WEEKLY if reason == REASON_WEEKLY else reason.upper(),
        "reason": reason,
        "round": int(rnd),
        "message": (
            f"{STATUS_WEEKLY}: the weekly (seven-day) subscription budget is exhausted; the "
            f"partial submission was checkpointed. Resume after the reset with --resume "
            f"(same run_id) to continue this round — the workspace submission/ is preserved."
        ),
        "resets_at_epoch": reset,
        "resets_at_iso": (datetime.fromtimestamp(reset, tz=timezone.utc).isoformat() if reset else None),
        "detected_at": datetime.now(timezone.utc).isoformat(),
        "resume_command": f"--resume --run-id <this run_id>  (relaunch after the weekly reset)",
    }
    p = Path(run_dir) / "quota_status.yaml"
    p.write_text(yaml.safe_dump(doc, sort_keys=False))
    return p


def _main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python resume_on_quota.py <run_dir>", file=sys.stderr)
        return 2
    run_dir = argv[1]
    tp = _latest_transcript(run_dir)
    if tp is None:
        print(f"{run_dir}: no round transcript found")
        return 1
    was, reason = round_was_cut_short(run_dir, transcript=tp)
    policy = resume_policy(reason)
    print(f"run_dir      : {run_dir}")
    print(f"transcript   : {tp.name}")
    print(f"cut_short    : {was}")
    print(f"reason       : {reason or '(finished cleanly)'}")
    print(f"policy       : {policy}")
    if reason == REASON_WEEKLY:
        reset = weekly_reset_epoch(tp)
        iso = datetime.fromtimestamp(reset, tz=timezone.utc).isoformat() if reset else "(unknown)"
        print(f"weekly_resets: {iso}")
        print(f"exit_status  : {STATUS_WEEKLY} (exit code {QUOTA_WEEKLY_EXIT_CODE})")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
