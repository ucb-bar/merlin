"""Cut-short detection + resume/checkpoint policy for the capsule_bench QA loop.

An agentic round can die with the hard parts built but the submission unfinished. Two failure modes
lost the whole round (scored a misleading 0) even though a partial ``submission/`` was on disk:

  1. **Wall-clock timeout (rc=124).** ``launch_agent`` SIGKILLs the driver process group; its transcript
     ends mid-stream with NO terminating ``{"type":"result"}`` event. The fix is cheap: the workspace
     ``submission/`` already persists across rounds, so the NEXT round can just CONTINUE it — we only
     need to TELL the fresh session that its partial work is preserved and to finish the incomplete
     pieces (``manifest.yaml`` + the CLI entrypoints + the target artifact) first.

  2. **Weekly (seven-day) quota wall.** The subscription budget is exhausted with a
     ``rate_limit_event`` whose ``rateLimitType == "seven_day"`` (and a terminal ``result`` carrying
     ``terminal_reason == "api_error"`` / ``api_error_status == 429`` / a "weekly limit" message). This
     resets DAYS later, so a same-session sleep-and-retry (what we do for the short five-hour window) is
     pointless — it would just burn the remaining rounds against the wall. Instead we CHECKPOINT the
     partial submission and EXIT with a distinct status (``QUOTA_EXHAUSTED_WEEKLY``) so an operator
     relaunches with ``--resume`` after the reset, rather than recording a misleading converged=False/0.

This module is the single home for that policy. It reuses the five-hour / daily detectors in
``_ratelimit`` (``RL``) and adds the two cases they miss. It is target-agnostic, parses the transcript
structurally (JSON events, ``str`` membership — never regex), and leaks no goldens.

Policy per reason:
  * ``timeout`` / ``rate_limit_five_hour`` -> RESUME_IN_BUDGET (continue the partial within the round
    budget; the loop's five-hour branch already sleeps+retries, timeout just re-launches next round).
  * ``quota_weekly`` / ``quota_daily``     -> EXIT_WITH_STATUS (checkpoint + stop; resume after reset).

CLI: ``python resume_on_quota.py <run_dir>`` prints the classification of its latest round transcript.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import _ratelimit as RL

# Reasons a round can be "cut short" (a productive round that did not get to finish/grade cleanly).
REASON_TIMEOUT = "timeout"                      # wall-clock (rc=124) — transcript killed mid-stream
REASON_FIVE_HOUR = "rate_limit_five_hour"       # short org window rejection (zero work)
REASON_WEEKLY = "quota_weekly"                  # seven-day subscription budget exhausted
REASON_DAILY = "quota_daily"                    # provider daily token quota exhausted

# Policy verbs.
RESUME_IN_BUDGET = "resume"    # keep going: the partial submission is finished/continued within budget
EXIT_WITH_STATUS = "exit"      # stop now: the quota resets far in the future; resume after reset
NO_POLICY = "none"             # the round was NOT cut short

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
    for e in RL._iter_events(transcript_path):
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
    for e in RL._iter_events(transcript_path):
        t = e.get("type")
        if t == "rate_limit_event":
            ri = e.get("rate_limit_info", {}) or {}
            if ri.get("rateLimitType") == "seven_day" and (
                    ri.get("status") == "rejected" or ri.get("overageStatus") == "rejected"):
                return True
        elif t == "result":
            if e.get("terminal_reason") == "api_error" and int(e.get("api_error_status") or 0) == 429:
                if "weekly" in str(e.get("result", "")).lower():
                    return True
    return False


def weekly_reset_epoch(transcript_path: str | Path) -> int | None:
    """The ``resetsAt`` epoch (seconds) from a seven-day rejection event, or None."""
    latest = None
    for e in RL._iter_events(transcript_path):
        if e.get("type") == "rate_limit_event":
            ri = e.get("rate_limit_info", {}) or {}
            if ri.get("rateLimitType") == "seven_day":
                ra = ri.get("resetsAt")
                if isinstance(ra, (int, float)):
                    latest = int(ra)
    return latest


def classify(transcript_path: str | Path, rc: int | None = None) -> str:
    """Classify ONE round transcript into a cut-short reason (or "" if it finished cleanly).

    Order is most-specific-first so a rate-limited round is named by its limit, never mislabeled a
    timeout: weekly -> daily -> five-hour -> timeout.
    """
    if weekly_quota_hit(transcript_path):
        return REASON_WEEKLY
    if RL.daily_limit_hit(transcript_path):
        return REASON_DAILY
    if RL.round_rejected(transcript_path):
        return REASON_FIVE_HOUR
    # Timeout: the driver was SIGKILLed (rc=124) OR the transcript has no terminating result event.
    if rc == 124 or not _has_result_event(transcript_path):
        return REASON_TIMEOUT
    return ""


def round_was_cut_short(run_dir: str | Path, rc: int | None = None,
                        transcript: str | Path | None = None) -> tuple[bool, str]:
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


def write_quota_status(run_dir: str | Path, reason: str, rnd: int,
                       transcript: str | Path | None = None) -> Path:
    """Persist a distinct ``run_dir/quota_status.yaml`` for a quota-exhausted stop, so the run is
    recognizable as blocked-not-failed and an operator knows when to relaunch. Returns the file path.
    (No PyYAML dependency here — a tiny flat dump keeps this module import-light.)"""
    import yaml  # local import: yaml is always available in the harness runtime
    reset = weekly_reset_epoch(transcript) if (reason == REASON_WEEKLY and transcript) else None
    doc = {
        "status": STATUS_WEEKLY if reason == REASON_WEEKLY else reason.upper(),
        "reason": reason,
        "round": int(rnd),
        "message": (f"{STATUS_WEEKLY}: the weekly (seven-day) subscription budget is exhausted; the "
                    f"partial submission was checkpointed. Resume after the reset with --resume "
                    f"(same run_id) to continue this round — the workspace submission/ is preserved."),
        "resets_at_epoch": reset,
        "resets_at_iso": (datetime.fromtimestamp(reset, tz=timezone.utc).isoformat()
                          if reset else None),
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
