"""Cut-short detection + resume/quota policy for the agentic QA loop.

A round can die with a partial submission on disk in two ways the loop must tell apart:
  * a wall-clock timeout (rc=124) — the driver is SIGKILLed and its transcript ends with NO terminating
    `result` event; the partial submission is resumable in-budget next round, so policy = RESUME.
  * a WEEKLY (seven-day) subscription-quota wall — a `rate_limit_event` with `rateLimitType=seven_day`
    (and/or a terminal api_error 429 "weekly limit" result); it resets days later, so policy = EXIT with a
    distinct QUOTA_EXHAUSTED_WEEKLY status rather than a misleading converged=False/0.

Hermetic: synthetic transcripts on tmp dirs. Also asserts on the two REAL run dirs when present (they are
under out/runs/, gitignored + purgeable) so the fixtures stay faithful to what the driver really emits.
"""
from __future__ import annotations

import importlib.util
import json
import sys

from merlin.common.paths import merlin_dir, repo_root


def _load():
    hdir = merlin_dir() / "experiments" / "capsule_bench" / "harness"
    if str(hdir) not in sys.path:
        sys.path.insert(0, str(hdir))   # so resume_on_quota's `import _ratelimit` resolves
    p = hdir / "resume_on_quota.py"
    spec = importlib.util.spec_from_file_location("resume_on_quota", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_round(run_dir, events, rnd=0):
    d = run_dir / "rounds"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"round_{rnd:02d}.transcript.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in events))
    return run_dir


# --- synthetic transcript fragments faithful to the real driver output ---------------------------------
def _tool_use(name="Write"):
    return {"type": "assistant",
            "message": {"content": [{"type": "tool_use", "name": name, "input": {}}]}}


def _clean_result(all_pass=False):
    return {"type": "result", "is_error": False, "terminal_reason": "success",
            "result": "done", "subtype": "success"}


def _weekly_events():
    # what the radiance run really emitted: real tool work, then a seven-day rejection + a terminal
    # api_error 429 "weekly limit" result.
    return [
        _tool_use("Write"),
        {"type": "rate_limit_event",
         "rate_limit_info": {"status": "rejected", "resetsAt": 1786388400,
                             "rateLimitType": "seven_day", "overageStatus": "rejected",
                             "overageDisabledReason": "org_level_disabled"}},
        {"type": "result", "is_error": True, "terminal_reason": "api_error",
         "api_error_status": 429, "subtype": "success",
         "result": "You've hit your weekly limit · resets Aug 10, 12pm (America/Los_Angeles)"},
    ]


def _timeout_events():
    # what a rc=124 wall-clock kill leaves: real work, transcript ends mid-stream (NO result event).
    return [_tool_use("Read"), _tool_use("Write"),
            {"type": "assistant", "message": {"content": [{"type": "thinking", "thinking": "..."}]}}]


def test_weekly_quota_classifies_as_exit(tmp_path):
    R = _load()
    rd = _write_round(tmp_path / "run", _weekly_events())
    was, reason = R.round_was_cut_short(rd)
    assert was and reason == R.REASON_WEEKLY
    assert R.resume_policy(reason) == R.EXIT_WITH_STATUS
    assert R.weekly_quota_hit(rd / "rounds" / "round_00.transcript.jsonl")
    assert R.weekly_reset_epoch(rd / "rounds" / "round_00.transcript.jsonl") == 1786388400
    sp = R.write_quota_status(rd, R.REASON_WEEKLY, rnd=0,
                              transcript=rd / "rounds" / "round_00.transcript.jsonl")
    import yaml
    doc = yaml.safe_load(sp.read_text())
    assert doc["status"] == R.STATUS_WEEKLY == "QUOTA_EXHAUSTED_WEEKLY"
    assert doc["resets_at_epoch"] == 1786388400 and doc["round"] == 0


def test_timeout_classifies_as_resume(tmp_path):
    R = _load()
    rd = _write_round(tmp_path / "run", _timeout_events())
    # No result event -> timeout even without rc; rc=124 also forces it.
    was, reason = R.round_was_cut_short(rd)
    assert was and reason == R.REASON_TIMEOUT
    assert R.resume_policy(reason) == R.RESUME_IN_BUDGET
    was2, reason2 = R.round_was_cut_short(rd, rc=124)
    assert was2 and reason2 == R.REASON_TIMEOUT


def test_clean_round_is_not_cut_short(tmp_path):
    R = _load()
    rd = _write_round(tmp_path / "run", [_tool_use("Write"), _clean_result()])
    was, reason = R.round_was_cut_short(rd)
    assert not was and reason == ""
    assert R.resume_policy(reason) == R.NO_POLICY


def test_resume_note_is_prepended_to_brief(tmp_path):
    R = _load()
    ws = tmp_path / "ws"
    (ws / "qa").mkdir(parents=True)
    (ws / "qa" / "round_brief.md").write_text("# existing brief\n")
    R.prepend_resume_note(ws, R.REASON_TIMEOUT)
    txt = (ws / "qa" / "round_brief.md").read_text()
    assert txt.startswith("> ## RESUME")
    assert "manifest.yaml" in txt and "# existing brief" in txt   # banner FIRST, brief preserved


def test_real_run_dirs_when_present():
    """The two real cut-short runs on disk (gitignored/purgeable) must classify as documented."""
    import pytest
    R = _load()
    cases = [
        ("atlas/capsule-bench/merlin_assisted/merlincirct_atlas_ccval1", R.REASON_TIMEOUT,
         R.RESUME_IN_BUDGET),
        ("radiance/capsule-bench/merlin_assisted/merlincirct_rad_ccval1", R.REASON_WEEKLY,
         R.EXIT_WITH_STATUS),
    ]
    seen = 0
    for rel, want_reason, want_policy in cases:
        rd = repo_root() / "out" / "runs" / rel
        if not (rd / "rounds").is_dir():
            continue
        seen += 1
        was, reason = R.round_was_cut_short(rd)
        assert was and reason == want_reason, f"{rel}: got {reason!r}"
        assert R.resume_policy(reason) == want_policy
    if seen == 0:
        pytest.skip("real cut-short run dirs not present (purged) — synthetic tests cover the logic")
