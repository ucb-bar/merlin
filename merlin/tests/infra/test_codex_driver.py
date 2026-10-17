"""Offline proof that the Codex agent driver is faithful — BEFORE any campaign spend.

We drive the REAL ``codex_agent.run_round`` against a fake ``codex`` binary that
replays a scripted ``codex exec --json`` stream, so the actual command assembly,
streaming capture, event translation and token arithmetic are exercised rather
than mocked. What that lets us assert:

  * the command matches the flags the INSTALLED CLI (0.147.0) really has — in
    particular that ``--ask-for-approval`` is never passed, because it does not
    exist in this version and would abort the launch;
  * Codex's token subsets are translated, not copied: its ``input_tokens`` is a
    total that already contains the cache reads, while the transcript shape the
    harness consumes means the uncached remainder by that name;
  * a turn that FAILED, which carries no usage at all, is recorded as unmeasured
    rather than as zero tokens;
  * a killed/timed-out run still leaves the raw JSONL (and therefore the token
    counts) on disk;
  * instruction-file asymmetry between arms is recorded in the artifact.

Nothing here spends quota or contacts OpenAI. The live counterpart is the bwrap
canary, which is opt-in.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import stat
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from merlin.common.paths import merlin_dir

_HARNESS = merlin_dir() / "experiments/capsule_bench/harness"
if str(_HARNESS) not in sys.path:
    sys.path.insert(0, str(_HARNESS))

from merlin_experiments.phase1.providers import codex_agent as CA  # noqa: E402  (path shim above)

# The measured shape of a real 0.147.0 turn, used as the scripted reply.
_REAL_USAGE = {
    "input_tokens": 36767,
    "cached_input_tokens": 28160,
    "cache_write_input_tokens": 0,
    "output_tokens": 203,
    "reasoning_output_tokens": 90,
}


def _fake_codex(
    tmp_path: Path,
    lines: list[dict],
    *,
    final: str = "DONE",
    exit_code: int = 0,
    hang: bool = False,
    orphan_pid_path: Path | None = None,
    version: str = "codex-cli 0.153.0",
    replies: list[tuple[list[dict], int]] | None = None,
    finals: list[str | None] | None = None,
) -> Path:
    """Write an executable stand-in for the codex CLI that replays *lines*.

    It also honors ``-o <file>`` so the driver's final-message handling is real.
    """
    script = tmp_path / "fake_codex"
    # The scripted stream goes in a sibling JSON file rather than being embedded
    # in the source: a Python literal is not JSON, so an ``exit_code: None`` in
    # an ``item.started`` payload would render as ``null`` and not parse.
    stream_path = tmp_path / "fake_codex_stream.json"
    stream_path.write_text(json.dumps(replies if replies is not None else lines))
    body = [
        f"#!{sys.executable}",
        "import json, sys, time, os, subprocess",
        "from pathlib import Path",
        "argv = sys.argv[1:]",
        # The real CLI answers --version without reading stdin, and the driver asks it for the
        # provenance stamp. A stand-in that replayed its event stream here would hand back the first
        # JSON line as a version string.
        "if '--version' in argv:",
        f"    sys.stdout.write({version!r} + '\\n')",
        "    sys.exit(0)",
        "out = None",
        "for i, a in enumerate(argv):",
        "    if a in ('-o', '--output-last-message') and i + 1 < len(argv):",
        "        out = argv[i + 1]",
        "sys.stdin.read()",
        f"lines = json.load(open({str(stream_path)!r}))",
        *(
            [
                f"calls = Path({str(tmp_path / 'fake_codex_calls.jsonl')!r})",
                "attempt = len(calls.read_text().splitlines()) if calls.exists() else 0",
                "with calls.open('a') as log: log.write(json.dumps(argv) + '\\n')",
                "lines, attempt_rc = lines[min(attempt, len(lines) - 1)]",
            ]
            if replies is not None
            else [f"attempt_rc = {exit_code}"]
        ),
        *(
            [f"attempt_final = {finals!r}[min(attempt, {len(finals) - 1})]"]
            if finals is not None
            else [f"attempt_final = {final!r}"]
        ),
        "for line in lines:",
        "    sys.stdout.write(json.dumps(line) + '\\n')",
        "    sys.stdout.flush()",
        *(
            [
                "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'], "
                "                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, "
                "                         stderr=subprocess.DEVNULL, close_fds=True)",
                f"open({str(orphan_pid_path)!r}, 'w').write(str(child.pid))",
            ]
            if orphan_pid_path is not None
            else []
        ),
        *(["time.sleep(600)"] if hang else []),
        "if out and attempt_final is not None:",
        "    open(out, 'w').write(attempt_final)",
        "sys.exit(attempt_rc)",
    ]
    script.write_text("\n".join(body) + "\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return script


def _stream(usage: dict | None = _REAL_USAGE, *, failed: bool = False) -> list[dict]:
    lines: list[dict] = [
        {"type": "thread.started", "thread_id": "01a01161-dead-beef-0000-000000000001"},
        {"type": "turn.started"},
        {
            "type": "item.started",
            "item": {
                "id": "item_0",
                "type": "command_execution",
                "command": "/bin/bash -lc 'ls'",
                "aggregated_output": "",
                "exit_code": None,
                "status": "in_progress",
            },
        },
        {
            "type": "item.completed",
            "item": {
                "id": "item_0",
                "type": "command_execution",
                "command": "/bin/bash -lc 'ls'",
                "aggregated_output": "TASK.md\n",
                "exit_code": 0,
                "status": "completed",
            },
        },
        {"type": "item.completed", "item": {"id": "item_1", "type": "agent_message", "text": "DONE"}},
    ]
    if failed:
        lines.append({"type": "turn.failed", "error": {"message": "upstream 400"}})
    else:
        lines.append({"type": "turn.completed", "usage": usage})
    return lines


def _run(
    tmp_path: Path,
    script: Path,
    *,
    sandbox: str = "none",
    timeout: int = 60,
    instruction_files: tuple[str, ...] = ("TASK.md",),
    continue_session: bool = False,
):
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    for name in instruction_files:
        (ws / name).write_text("do the thing\n")
    run_dir = tmp_path / "run"
    run_dir.mkdir(exist_ok=True)
    rc, tpath = CA.run_round(
        ws,
        run_dir,
        "claude-opus-4-8",
        {},
        None,
        sandbox,
        0,
        timeout,
        effort="low",
        codex_binary=script,
        continue_session=continue_session,
    )
    records = [json.loads(line) for line in tpath.read_text().splitlines() if line.strip()]
    return rc, tpath, records


def _by_type(records: list[dict], kind: str) -> list[dict]:
    return [r for r in records if r.get("type") == kind]


# ---------------------------------------------------------------------------
# Command contract — against the flags the installed CLI actually has
# ---------------------------------------------------------------------------


def test_the_nonexistent_approval_flag_is_never_passed():
    """``--ask-for-approval`` was removed by 0.147.0; passing it aborts the run."""
    for sandbox in ("none", "bwrap"):
        cmd = CA.build_cmd(
            Path("/ws"), model="gpt-5.6-sol", effort="high", final_path=Path("/run/final.txt"), sandbox=sandbox
        )
        assert "--ask-for-approval" not in cmd
    cmd = CA.build_cmd(Path("/ws"), model="gpt-5.6-sol", effort="", final_path=Path("/run/final.txt"), sandbox="none")
    # The policy is a config override instead.
    assert "-c" in cmd and "approval_policy=never" in cmd


def test_outside_bwrap_codex_keeps_its_own_sandbox():
    cmd = CA.build_cmd(Path("/ws"), model="m", effort="", final_path=Path("/f"), sandbox="none")
    assert "--sandbox" in cmd and "workspace-write" in cmd
    assert "--dangerously-bypass-approvals-and-sandbox" not in cmd


def test_inside_bwrap_codex_enforces_the_frozen_candidate_profile():
    cmd = CA.build_cmd(Path("/ws"), model="m", effort="", final_path=Path("/f"), sandbox="bwrap")
    assert "--dangerously-bypass-approvals-and-sandbox" not in cmd
    assert "--sandbox" not in cmd
    assert "--strict-config" in cmd
    assert 'default_permissions="merlin-candidate"' in cmd
    resumed = CA.build_resume_cmd(
        Path("/ws"),
        model="m",
        effort="",
        final_path=Path("/f"),
        sandbox="bwrap",
        thread_id="session-id",
    )
    assert "--dangerously-bypass-approvals-and-sandbox" not in resumed
    assert "--sandbox" not in resumed
    assert "--strict-config" in resumed
    assert 'default_permissions="merlin-candidate"' in resumed


def test_bridged_codex_cannot_inherit_proxy_secret_into_untrusted_bwrap(tmp_path):
    with pytest.raises(RuntimeError, match="host-side proxy credential broker"):
        CA.run_round(
            tmp_path / "workspace",
            tmp_path / "run",
            "nemotron",
            {},
            None,
            "bwrap",
            0,
            1,
            effective_model="nemotron",
        )


def test_the_prompt_is_passed_on_stdin_not_as_an_argv_fragment():
    cmd = CA.build_cmd(Path("/ws"), model="m", effort="", final_path=Path("/f"), sandbox="none")
    assert cmd[-1] == "-", "prompt bytes must stay an artifact, not get mangled by quoting"


def test_effort_is_a_config_override_and_absent_when_unset():
    with_effort = CA.build_cmd(Path("/ws"), model="m", effort="high", final_path=Path("/f"), sandbox="none")
    assert 'model_reasoning_effort="high"' in with_effort
    without = CA.build_cmd(Path("/ws"), model="m", effort="", final_path=Path("/f"), sandbox="none")
    assert not any("model_reasoning_effort" in c for c in without)


@pytest.mark.parametrize(
    "alias,expected",
    [
        ("gpt-5.6-sol", "gpt-5.6-sol"),
        ("gpt-5.4", "gpt-5.4"),
        ("claude-opus-4-8", CA.DEFAULT_CODEX_MODEL),
        ("", CA.DEFAULT_CODEX_MODEL),
    ],
)
def test_model_aliases_resolve_to_a_slug_this_auth_mode_accepts(alias, expected):
    assert CA.resolve_model(alias) == expected


def test_preflighted_effective_model_bypasses_later_ambient_remapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _fake_codex(tmp_path, _stream())
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "TASK.md").write_text("do the thing\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.setenv("CODEX_BIN", str(script))
    monkeypatch.setenv("CODEX_MODEL_MAP", "declared-model=wrong-later-model")

    rc, transcript = CA.run_round(
        ws, run_dir, "declared-model", {}, None, "none", 0, 60, effort="high", effective_model="preflighted-model"
    )
    rows = [json.loads(line) for line in transcript.read_text(encoding="utf-8").splitlines()]
    init = next(row for row in rows if row.get("subtype") == "init")

    assert rc == 0
    assert init["model_requested"] == "declared-model"
    assert init["model"] == "preflighted-model"


def test_an_explicit_model_map_wins(monkeypatch):
    monkeypatch.setenv("CODEX_MODEL_MAP", "claude-opus-4-8=gpt-5.6-terra,x=y")
    assert CA.resolve_model("claude-opus-4-8") == "gpt-5.6-terra"


# ---------------------------------------------------------------------------
# Token subset translation — the arithmetic that inflates a bill if copied
# ---------------------------------------------------------------------------


def test_codex_input_total_becomes_uncached_input_not_a_copy():
    shaped, reported = CA.usage_to_claude_shape(_REAL_USAGE)
    assert reported is True
    # 36767 total - 28160 cache read - 0 cache write
    assert shaped["input_tokens"] == 8607
    assert shaped["cache_read_input_tokens"] == 28160
    assert shaped["cache_creation_input_tokens"] == 0
    # The provider's own total is kept alongside, for reconciliation.
    assert shaped["codex_input_tokens_total"] == 36767


def test_reasoning_is_kept_beside_output_never_added_to_it():
    shaped, _ = CA.usage_to_claude_shape(_REAL_USAGE)
    assert shaped["output_tokens"] == 203, "reasoning is already inside output"
    assert shaped["reasoning_output_tokens"] == 90


def test_cache_write_is_subtracted_from_input_too():
    shaped, _ = CA.usage_to_claude_shape(
        {"input_tokens": 1000, "cached_input_tokens": 600, "cache_write_input_tokens": 100, "output_tokens": 10}
    )
    assert shaped["input_tokens"] == 300
    assert shaped["cache_creation_input_tokens"] == 100


def test_inconsistent_subsets_clamp_at_zero():
    shaped, _ = CA.usage_to_claude_shape({"input_tokens": 10, "cached_input_tokens": 99})
    assert shaped["input_tokens"] == 0


def test_an_empty_usage_payload_is_unreported_rather_than_zeroes():
    shaped, reported = CA.usage_to_claude_shape({})
    assert reported is False
    assert shaped == {}, "unknown usage must not be emitted as a zero bill"


# ---------------------------------------------------------------------------
# End-to-end through the real run_round
# ---------------------------------------------------------------------------


def test_a_successful_round_translates_the_stream_into_the_harness_transcript(tmp_path):
    rc, tpath, records = _run(tmp_path, _fake_codex(tmp_path, _stream()))

    assert rc == 0
    init = _by_type(records, "system")[0]
    assert init["driver"] == "codex"
    assert init["model"] == CA.DEFAULT_CODEX_MODEL
    assert init["model_requested"] == "claude-opus-4-8"

    # The tool call became a tool_use + tool_result pair.
    tool_uses = [
        b
        for r in _by_type(records, "assistant")
        for b in r["message"].get("content", [])
        if b.get("type") == "tool_use"
    ]
    assert tool_uses and tool_uses[0]["name"] == "Bash"
    results = [b for r in _by_type(records, "user") for b in r["message"]["content"] if b.get("type") == "tool_result"]
    assert results and "TASK.md" in results[0]["content"]
    assert results[0]["is_error"] is False

    # Usage landed on an assistant record in the translated shape.
    usages = [r["message"]["usage"] for r in _by_type(records, "assistant") if "usage" in r["message"]]
    assert len(usages) == 1
    assert usages[0]["input_tokens"] == 8607
    assert _by_type(records, "result")[0]["subtype"] == "success"


def test_the_summary_records_subscription_billing_and_usage_completeness(tmp_path):
    _rc, tpath, records = _run(tmp_path, _fake_codex(tmp_path, _stream()))
    summary = _by_type(records, "codex_summary")[0]

    assert summary["billing_mode"] == "subscription_notional"
    assert summary["usage_complete"] is True
    assert summary["turns_started"] == 1 and summary["turns_usage_reported"] == 1
    assert summary["thread_id"] == "01a01161-dead-beef-0000-000000000001"
    # A sibling JSON summary exists for tooling that does not read transcripts.
    assert (tpath.parent / "round_00.codex_summary.json").is_file()


def test_a_failed_turn_is_recorded_as_unmeasured_not_as_zero_tokens(tmp_path):
    rc, _tpath, records = _run(tmp_path, _fake_codex(tmp_path, _stream(failed=True), exit_code=1))

    assert rc != 0
    failed = [r for r in _by_type(records, "assistant") if r.get("codex_turn_failed")]
    assert len(failed) == 1
    assert "usage" not in failed[0]["message"], "a failed turn's tokens are unknown, not 0"
    assert failed[0]["codex_usage_unreported"] is True

    summary = _by_type(records, "codex_summary")[0]
    assert summary["usage_complete"] is False
    assert summary["turns_usage_reported"] == 0
    assert any("upstream 400" in e for e in summary["errors"])


def test_capacity_interruption_resumes_same_thread_and_retains_failed_usage(tmp_path, monkeypatch):
    refusal = _stream(failed=True)
    refusal[-1]["error"]["message"] = "Selected model is at capacity. Please try a different model."
    script = _fake_codex(tmp_path, [], replies=[(refusal, 1), (_stream(), 0)])
    monkeypatch.setattr(CA, "_CONTINUE_MAX_TURNS", 2)
    monkeypatch.setattr(CA.time, "sleep", lambda seconds: None)
    rc, tpath, records = _run(tmp_path, script, continue_session=True, timeout=300)
    assert rc == 0, records[-2:]
    calls = [json.loads(line) for line in (tmp_path / "fake_codex_calls.jsonl").read_text().splitlines()]
    assert len(calls) == 2 and calls[1][:2] == ["exec", "resume"]
    assert "01a01161-dead-beef-0000-000000000001" in calls[1]
    assert calls[0][calls[0].index("--model") + 1] == calls[1][calls[1].index("--model") + 1]
    summary = _by_type(records, "codex_summary")[0]
    assert summary["capacity_retries"] == 1 and summary["errors"]
    assert summary["usage_complete"] is False and summary["turns_usage_reported"] == 1
    assert _by_type(records, "codex_capacity_retry")[0]["thread_id"] == summary["thread_id"]
    raw = (tpath.parent / "round_00.codex_events.raw.jsonl").read_text()
    assert '"turn.failed"' in raw and '"turn.completed"' in raw


@pytest.mark.parametrize("failure", ["capacity", "unauthorized"])
def test_capacity_retry_is_bounded_and_other_failures_are_not_retried(tmp_path, monkeypatch, failure):
    refusal = _stream(failed=True)
    refusal[-1]["error"]["message"] = "Selected model is at capacity." if failure == "capacity" else "unauthorized"
    script = _fake_codex(tmp_path, [], replies=[(refusal, 1)])
    monkeypatch.setattr(CA.time, "sleep", lambda seconds: None)
    rc, _, records = _run(tmp_path, script, continue_session=True, timeout=300)
    assert rc != 0
    calls = (tmp_path / "fake_codex_calls.jsonl").read_text().splitlines()
    assert len(calls) == (4 if failure == "capacity" else 1)
    assert _by_type(records, "result")[-1]["is_error"] is True


@pytest.mark.parametrize("boundary", ["rounds", "short_budget", "missing_thread", "mixed_error", "completed"])
def test_capacity_refusal_respects_session_and_budget_boundaries(tmp_path, boundary):
    refusal = _stream(failed=True)
    refusal[-1]["error"]["message"] = "Selected model is at capacity."
    if boundary == "missing_thread":
        refusal = refusal[1:]
    elif boundary == "mixed_error":
        refusal.insert(-1, {"type": "error", "message": "You've hit your usage limit."})
    elif boundary == "completed":
        refusal.insert(-1, {"type": "turn.completed", "usage": _REAL_USAGE})
    script = _fake_codex(tmp_path, [], replies=[(refusal, 1), (_stream(), 0)])
    rc, _, _ = _run(
        tmp_path, script, timeout=60 if boundary == "short_budget" else 300, continue_session=boundary != "rounds"
    )
    assert rc != 0 and len((tmp_path / "fake_codex_calls.jsonl").read_text().splitlines()) == 1


def test_capacity_retry_refuses_changed_thread_identity(tmp_path, monkeypatch):
    refusal = _stream(failed=True)
    refusal[-1]["error"]["message"] = "Selected model is at capacity."
    changed = _stream()
    changed[0]["thread_id"] = "different-thread"
    script = _fake_codex(tmp_path, [], replies=[(refusal, 1), (changed, 0)])
    monkeypatch.setattr(CA.time, "sleep", lambda seconds: None)
    rc, _, records = _run(tmp_path, script, continue_session=True, timeout=300)
    assert rc != 0
    calls = (tmp_path / "fake_codex_calls.jsonl").read_text().splitlines()
    assert len(calls) == 2
    summary = _by_type(records, "codex_summary")[0]
    assert summary["thread_id"] == refusal[0]["thread_id"]
    assert any("thread identity" in error for error in summary["unrecovered_errors"])


def test_capacity_retry_does_not_reuse_failed_attempt_final(tmp_path, monkeypatch):
    refusal = _stream(failed=True)
    refusal[-1]["error"]["message"] = "Selected model is at capacity."
    script = _fake_codex(tmp_path, [], replies=[(refusal, 1), (_stream(), 0)], finals=["STALE FAILED ANSWER", None])
    monkeypatch.setattr(CA, "_CONTINUE_MAX_TURNS", 2)
    monkeypatch.setattr(CA.time, "sleep", lambda seconds: None)
    rc, tpath, records = _run(tmp_path, script, continue_session=True, timeout=300)
    assert rc == 0 and _by_type(records, "result")[-1]["result"] == ""
    assert (tpath.parent / "round_00.turn00.final.txt").read_text() == "STALE FAILED ANSWER"


def test_the_raw_event_stream_is_persisted_byte_for_byte(tmp_path):
    _rc, tpath, _records = _run(tmp_path, _fake_codex(tmp_path, _stream()))
    raw = (tpath.parent / "round_00.codex_events.raw.jsonl").read_text()

    assert len(raw.splitlines()) == len(_stream())
    assert '"thread.started"' in raw
    # And a timestamped sibling, since the events carry no time of their own.
    stamped = [
        json.loads(line) for line in (tpath.parent / "round_00.codex_events.timestamped.jsonl").read_text().splitlines()
    ]
    assert [r["seq"] for r in stamped] == list(range(1, len(_stream()) + 1))
    assert all(r["arrived_at"] for r in stamped)


def test_a_hung_round_times_out_and_still_leaves_the_usage_on_disk(tmp_path):
    """The evidence must survive the kill — that is the whole point of streaming."""
    rc, tpath, records = _run(tmp_path, _fake_codex(tmp_path, _stream(), hang=True), timeout=2)

    assert rc == 124
    raw = (tpath.parent / "round_00.codex_events.raw.jsonl").read_text()
    assert '"turn.completed"' in raw, "lines emitted before the hang must be durable"
    summary = _by_type(records, "codex_summary")[0]
    assert summary["timed_out"] is True
    assert summary["turns_usage_reported"] == 1, "usage seen before the kill is still counted"


def test_a_successful_leader_cannot_leave_its_process_group_running(tmp_path):
    pid_path = tmp_path / "descendant.pid"
    rc, _tpath, _records = _run(tmp_path, _fake_codex(tmp_path, _stream(), orphan_pid_path=pid_path))

    assert rc == 0
    pid = int(pid_path.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_the_prompt_bytes_are_kept_as_an_artifact(tmp_path):
    _rc, tpath, _records = _run(tmp_path, _fake_codex(tmp_path, _stream()))
    prompt = (tpath.parent / "round_00.prompt.txt").read_text()
    assert "agent_selfcheck.py" in prompt, "the graded self-check instruction must reach the agent"
    assert "submission" in prompt


def test_the_default_prompt_requires_narrow_iteration_before_a_full_sweep(tmp_path):
    """The driver must not turn every source edit into a full-corpus grading job.

    A Gemmini clean-room run followed the former ``--capsules all after each build`` instruction
    literally: nine full 97-capsule checks in three hours, four invalidated by later edits, and only
    one focused check.  Pin the feedback-loop policy at the actual prompt seam so a relaunch cannot
    silently recreate that measurement churn.
    """
    _rc, tpath, _records = _run(tmp_path, _fake_codex(tmp_path, _stream()))
    prompt = (tpath.parent / "round_00.prompt.txt").read_text()

    assert "smallest affected capsule" in prompt
    assert "only after the focused checks improve" in prompt
    assert "You may edit submission/ while a self-check runs" in prompt
    # The recommended focused check is the certifying one: no single-engine screen in the command.
    assert "`python3 agent_selfcheck.py --submission submission --capsules <names>`" in prompt
    assert "certified pass/fail per capsule" in prompt
    assert "--sim spike --capsules" not in prompt
    assert "Build something minimal and write submission/manifest.yaml FIRST" in prompt
    assert "then use await_verdict.py rather than a poll loop" in prompt
    assert "do not launch `--capsules all` merely to refresh it" in prompt
    assert "--capsules all` with your shell after each build" not in prompt


def test_a_missing_codex_binary_fails_the_round_without_raising(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    os.environ["CODEX_BIN"] = str(tmp_path / "definitely-not-here")
    try:
        rc, tpath = CA.run_round(ws, run_dir, "m", {}, None, "none", 0, 30)
    finally:
        os.environ.pop("CODEX_BIN", None)
    assert rc == 127
    records = [json.loads(line) for line in tpath.read_text().splitlines() if line.strip()]
    assert _by_type(records, "result")[0]["is_error"] is True


# ---------------------------------------------------------------------------
# Instruction parity between arms
# ---------------------------------------------------------------------------


def test_workspace_instruction_files_are_recorded_so_asymmetry_is_visible(tmp_path):
    """Codex reads AGENTS.md where Claude reads CLAUDE.md; an arm that quietly
    got extra instructions is not the same arm."""
    _rc, _tpath, records = _run(tmp_path, _fake_codex(tmp_path, _stream()), instruction_files=("TASK.md", "AGENTS.md"))
    init = _by_type(records, "system")[0]
    assert set(init["workspace_instruction_files"]) == {"TASK.md", "AGENTS.md"}


def test_the_driver_does_not_author_instruction_files_itself(tmp_path):
    _rc, _tpath, _records = _run(tmp_path, _fake_codex(tmp_path, _stream()), instruction_files=("TASK.md",))
    assert not (tmp_path / "ws" / "AGENTS.md").exists(), "parity is recorded, never manufactured"


# ---------------------------------------------------------------------------
# The isolated CODEX_HOME — keeping prior sessions away from a graded agent
# ---------------------------------------------------------------------------


def test_the_isolated_home_holds_a_frozen_config_and_no_credential(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    info = CA.prepare_codex_home(tmp_path / "home", model="gpt-5.6-sol", effort="high", workspace=ws)

    config = (tmp_path / "home" / "config.toml").read_text()
    assert 'model = "gpt-5.6-sol"' in config
    assert 'model_reasoning_effort = "high"' in config
    assert 'default_permissions = "merlin-candidate"' in config
    assert f'{json.dumps(str(tmp_path / "home"))} = "deny"' in config
    assert '":root" = "deny"' in config
    # Only the exact prepared workspace's trust is frozen. The user's own
    # unrelated project trust and notice state are never copied.
    assert tomllib.loads(config)["projects"] == {str(ws): {"trust_level": "trusted"}}

    assert info["auth_copied"] is False
    assert not (tmp_path / "home" / "auth.json").exists(), "the credential is bind-mounted, never written into the tree"
    assert info["config_sha256"] and info["isolated_from_real_home"] is True

    bridged = CA.prepare_codex_home(tmp_path / "bridged", model="nemotron", effort="high", workspace=ws)
    bridged_config = tomllib.loads((tmp_path / "bridged/config.toml").read_text())
    assert bridged_config["model_provider"] == "merlinproxy"
    assert bridged_config["default_permissions"] == "merlin-candidate"
    assert bridged["config_sha256"]


def test_isolated_home_freezes_exact_workspace_trust_before_cli_startup(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    home = tmp_path / "isolated-codex"
    info = CA.prepare_codex_home(home, model="gpt-5.6-sol", effort="high", workspace=ws)
    config_path = home / "config.toml"
    config = config_path.read_text()
    assert tomllib.loads(config)["projects"] == {str(ws): {"trust_level": "trusted"}}
    CA._verify_frozen_config(home, info["config_sha256"])

    for changed in (
        config.replace('trust_level = "trusted"', 'trust_level = "untrusted"'),
        config + '\n[projects."/other/workspace"]\ntrust_level = "trusted"\n',
        config.replace('model_reasoning_effort = "high"', 'model_reasoning_effort = "low"'),
    ):
        config_path.write_text(changed)
        with pytest.raises(RuntimeError, match="config changed after it was frozen"):
            CA._verify_frozen_config(home, info["config_sha256"])

    alias = tmp_path / "workspace-alias"
    alias.symlink_to(ws, target_is_directory=True)
    for invalid in (Path("relative-workspace"), tmp_path / "missing-workspace", alias):
        with pytest.raises(ValueError, match="existing canonical absolute"):
            CA.prepare_codex_home(tmp_path / "refused-home", model="gpt-5.6-sol", effort="high", workspace=invalid)
    assert not (tmp_path / "refused-home").exists()


def test_selected_cli_config_load_preserves_exact_workspace_trust(tmp_path):
    """No model request: exercise the real CLI's project config parser/startup."""
    codex_bin = shutil.which("codex")
    if codex_bin is None:
        pytest.skip("selected Codex CLI is unavailable")
    ws = tmp_path / "workspace"
    ws.mkdir()
    home = tmp_path / "isolated-codex"
    info = CA.prepare_codex_home(home, model="gpt-5.6-sol", effort="high", workspace=ws)
    config_path = home / "config.toml"
    before = config_path.read_bytes()
    result = subprocess.run(
        [codex_bin, "-C", str(ws), "features", "list"],
        env={**os.environ, "CODEX_HOME": str(home)},
        capture_output=True,
        timeout=15,
    )
    assert result.returncode == 0
    assert config_path.read_bytes() == before
    CA._verify_frozen_config(home, info["config_sha256"])


def test_native_candidate_profile_blocks_synthetic_auth_inside_outer_bwrap(tmp_path, monkeypatch):
    """No model request: test the real installed CLI against a dummy auth mount."""
    if not shutil.which("bwrap") or not shutil.which("codex"):
        pytest.skip("live Codex/bwrap isolation probe requires both installed executables")
    from merlin.targetgen.sandbox import bwrap as BW

    fake_real = tmp_path / "operator-codex"
    fake_real.mkdir()
    (fake_real / "auth.json").write_text("synthetic credential, never a real token")
    monkeypatch.setattr(CA, "real_codex_home", lambda: fake_real)
    ws = tmp_path / "workspace"
    ws.mkdir()
    home = tmp_path / "isolated-codex"
    info = CA.prepare_codex_home(home, model="gpt-5.6-sol", effort="high", workspace=ws)
    assert tomllib.loads((home / "config.toml").read_text())["default_permissions"] == "merlin-candidate"
    CA._verify_frozen_config(home, info["config_sha256"])

    def sandbox_command(inner, _ws, bundle, *, extra_binds):
        argv = BW.base_argv(ws, bundle, repo=tmp_path, include_claude_home=False, inherit_environment=False)
        return shlex.join(argv + extra_binds + ["bash", "-c", inner])

    rounds = tmp_path / "rounds"
    CA._preflight_candidate_sandbox(
        ws,
        home,
        str(shutil.which("codex")),
        {},
        sandbox_command,
        rounds,
        0,
    )
    # If the exact deny is removed, the broader /scratch read grant exposes the
    # dummy file. The mandatory preflight must refuse this weaker profile.
    config = home / "config.toml"
    config.write_text(config.read_text().replace(f'{json.dumps(str(home))} = "deny"\n', ""))
    with pytest.raises(RuntimeError, match="preflight failed"):
        CA._preflight_candidate_sandbox(
            ws,
            home,
            str(shutil.which("codex")),
            {},
            sandbox_command,
            rounds,
            1,
        )


def test_native_candidate_cannot_reach_parent_open_auth_or_relogin(tmp_path, monkeypatch):
    """Probe /proc FDs, process memory, ptrace, and a nested CLI with dummy auth."""
    if not shutil.which("bwrap") or not shutil.which("codex"):
        pytest.skip("live Codex/bwrap isolation probe requires both installed executables")
    from merlin.targetgen.sandbox import bwrap as BW

    fake_real = tmp_path / "operator-codex"
    fake_real.mkdir()
    (fake_real / "auth.json").write_text(json.dumps({"OPENAI_API_KEY": "sk-synthetic-test-only"}))
    host_login = subprocess.run(
        ["codex", "login", "status"],
        env={**os.environ, "CODEX_HOME": str(fake_real)},
        capture_output=True,
        timeout=10,
    )
    assert host_login.returncode == 0, "synthetic login must be a valid negative control"
    monkeypatch.setattr(CA, "real_codex_home", lambda: fake_real)
    ws = tmp_path / "workspace"
    ws.mkdir()
    home = tmp_path / "isolated-codex"
    CA.prepare_codex_home(home, model="gpt-5.6-sol", effort="high", workspace=ws)

    candidate = """
import ctypes, json, os, subprocess
pid = int(os.environ["HOLDER_PID"])
fd = int(os.environ["HOLDER_FD"])
try:
    with open(f"/proc/{pid}/fd/{fd}", "rb") as stream:
        fd_readable = bool(stream.read(1))
except OSError:
    fd_readable = False
class IOVec(ctypes.Structure):
    _fields_ = [("base", ctypes.c_void_p), ("length", ctypes.c_size_t)]
size = int(os.environ["HOLDER_SIZE"])
local = ctypes.create_string_buffer(size)
local_vec = IOVec(ctypes.addressof(local), size)
remote_vec = IOVec(int(os.environ["HOLDER_ADDR"]), size)
libc = ctypes.CDLL(None, use_errno=True)
libc.process_vm_readv.restype = ctypes.c_ssize_t
memory_readable = libc.process_vm_readv(pid, ctypes.byref(local_vec), 1, ctypes.byref(remote_vec), 1, 0) > 0
ptrace_attached = libc.ptrace(0x4206, pid, None, None) == 0  # PTRACE_SEIZE, no stop
if ptrace_attached:
    libc.ptrace(17, pid, None, None)  # PTRACE_DETACH
login = subprocess.run(["codex", "login", "status"], capture_output=True, timeout=10)
result = {"parent_visible": os.path.exists(f"/proc/{pid}"),
          "fd_readable": fd_readable, "memory_readable": memory_readable,
          "ptrace_attached": ptrace_attached, "nested_codex_logged_in": login.returncode == 0}
print(json.dumps(result))
"""
    holder = """
import ctypes, json, os, subprocess, sys
fd = os.open(os.environ["CODEX_HOME"] + "/auth.json", os.O_RDONLY)
buffer = ctypes.create_string_buffer(b"SYNTHETIC-PARENT-MEMORY-ONLY")
env = os.environ.copy()
env.update(HOLDER_PID=str(os.getpid()), HOLDER_FD=str(fd),
           HOLDER_ADDR=str(ctypes.addressof(buffer)), HOLDER_SIZE=str(len(buffer)))
command = (["/usr/bin/python3", "-c", sys.argv[2]] if sys.argv[3] == "direct" else
           ["codex", "sandbox", "-P", "merlin-candidate", "-C", sys.argv[1],
            "--", "/usr/bin/python3", "-c", sys.argv[2]])
child = subprocess.run(command, env=env, capture_output=True, text=True, timeout=20)
sys.stdout.write(child.stdout)
sys.stderr.write(child.stderr)
sys.exit(child.returncode)
"""
    prefix = BW.base_argv(ws, {}, repo=tmp_path, include_claude_home=False, inherit_environment=False)
    prefix += CA.codex_runtime_binds(home)
    direct = subprocess.run(
        prefix + ["/usr/bin/python3", "-c", holder, str(ws), candidate, "direct"],
        cwd=ws,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert direct.returncode == 0, direct.stderr[-1200:]
    direct_result = json.loads(direct.stdout.strip().splitlines()[-1])
    assert direct_result["fd_readable"] is True, "unsandboxed shell must expose the dummy open FD"
    assert direct_result["nested_codex_logged_in"] is True, "unsandboxed CLI must see dummy auth"

    command = prefix + ["/usr/bin/python3", "-c", holder, str(ws), candidate, "sandbox"]
    proc = subprocess.run(command, cwd=ws, capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr[-1200:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result == {
        "parent_visible": True,
        "fd_readable": False,
        "memory_readable": False,
        "ptrace_attached": False,
        "nested_codex_logged_in": False,
    }


def test_the_binds_reach_the_launchers_real_target_and_redirect_the_home(tmp_path):
    """~/.local/bin/codex is a symlink into ~/.codex/packages, so binding
    ~/.local/bin alone leaves the launcher pointing at nothing."""
    home = tmp_path / "home"
    home.mkdir()
    binds = CA.codex_runtime_binds(home)

    joined = " ".join(binds)
    assert "--setenv CODEX_HOME " + str(home) in joined
    assert f"--bind {home} {home}" in joined, "codex must be able to write sessions/state"
    real = CA.real_codex_home()
    if (real / "packages").exists():
        assert f"--ro-bind {real / 'packages'} {real / 'packages'}" in joined


def test_the_real_dotcodex_directory_is_never_bound_wholesale(tmp_path):
    """Binding ~/.codex would expose sessions/ — every prior conversation on the
    host — to a graded agent. That is an answer-leak surface."""
    home = tmp_path / "home"
    home.mkdir()
    binds = CA.codex_runtime_binds(home)
    real = str(CA.real_codex_home())

    # The only permitted sources under the real home are packages/ and auth.json.
    sources = [binds[i + 1] for i, a in enumerate(binds) if a in ("--bind", "--ro-bind") and i + 1 < len(binds)]
    under_real = [s for s in sources if s.startswith(real)]
    assert all(s.startswith(f"{real}/packages") or s == f"{real}/auth.json" for s in under_real), (
        f"unexpected bind out of the real home: {under_real}"
    )
    assert real not in sources, "the real ~/.codex must never be bound as a whole"


def test_the_credential_is_bound_writable_onto_the_isolated_home(tmp_path):
    """The credential must land on the ISOLATED home, and must be writable.

    This assertion used to demand read-only, so that a refresh attempt would fail loudly rather than
    rewrite a shared credential. Measured, that inverts: OAuth refresh tokens are single-use and
    rotate, so Codex spends the old token server-side, cannot write the new pair back, and every later
    run dies `401 refresh_token_reused` -- while presenting as rounds that finish in seconds with a
    small constant score, i.e. as a bad agent rather than a dead credential.

    What still matters, and is what this test guards, is the DESTINATION: the credential is mapped onto
    `<codex_home>/auth.json`, never by exposing the real `~/.codex`. That the rest of the real home
    stays unreachable is asserted separately, just above.
    """
    home = tmp_path / "home"
    home.mkdir()
    binds = CA.codex_runtime_binds(home)
    auth = CA.real_codex_home() / "auth.json"
    if not auth.is_file():
        pytest.skip("no local codex credential to assert against")
    idx = binds.index(str(auth))
    assert binds[idx - 1] == "--bind", (
        "the credential must be writable or a rotated refresh token cannot be persisted, "
        "which kills every subsequent codex run"
    )
    assert binds[idx + 1] == str(home / "auth.json"), (
        "the credential must be mapped onto the isolated home, not the real ~/.codex"
    )


def test_a_canary_prompt_can_override_the_graded_instruction_but_is_not_the_default(tmp_path):
    """A measured arm always gets the graded text, so two arms cannot differ in
    what they were asked; only out-of-band uses override it."""
    script = _fake_codex(tmp_path, _stream())
    ws = tmp_path / "ws"
    ws.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    os.environ["CODEX_BIN"] = str(script)
    try:
        _rc, tpath = CA.run_round(ws, run_dir, "m", {}, None, "none", 0, 60, prompt="CANARY: run probe.sh")
    finally:
        os.environ.pop("CODEX_BIN", None)
    assert (tpath.parent / "round_00.prompt.txt").read_text() == "CANARY: run probe.sh"

    # Default (no override) is the graded instruction.
    os.environ["CODEX_BIN"] = str(script)
    try:
        _rc, tpath2 = CA.run_round(ws, run_dir, "m", {}, None, "none", 1, 60)
    finally:
        os.environ.pop("CODEX_BIN", None)
    assert "agent_selfcheck.py" in (tpath2.parent / "round_01.prompt.txt").read_text()


def test_an_unsupported_tiering_request_is_recorded_rather_than_silently_dropped(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "TASK.md").write_text("x")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    os.environ["CODEX_BIN"] = str(_fake_codex(tmp_path, _stream()))
    try:
        _rc, tpath = CA.run_round(ws, run_dir, "m", {}, None, "none", 0, 60, subagent_model="gpt-5.4-mini")
    finally:
        os.environ.pop("CODEX_BIN", None)
    records = [json.loads(line) for line in tpath.read_text().splitlines() if line.strip()]
    init = _by_type(records, "system")[0]
    assert init["tiering_requested_but_unsupported"] is True


# --- the last message must survive the sandbox -------------------------------------------------
# MEASURED (gemmini arm-4 calibration, 2026-08-29): every sandboxed round logged
#   Failed to write last message file ".../round_00.final.txt": No such file or directory (os error 2)
# because `-o` pointed under the run directory, which is on /scratch -- and the sandbox tmpfs-hides
# /scratch* on purpose. The read is guarded, so nothing failed: the round's `result` was just empty.


def test_the_last_message_target_is_writable_inside_the_sandbox(tmp_path):
    ws = tmp_path / "ws"
    final = tmp_path / "run" / "rounds" / "round_03.final.txt"

    inner = CA.last_message_path(ws, final, "bwrap")
    assert ws in inner.parents, "the sandbox hides /scratch*; the workspace is the writable tree"
    assert inner.name == final.name

    assert CA.last_message_path(ws, final, "none") == final


def test_an_unsandboxed_round_still_recovers_its_final_message(tmp_path):
    _rc, tpath, records = _run(tmp_path, _fake_codex(tmp_path, _stream(), final="ALL DONE"))
    assert (tpath.parent / "round_00.final.txt").read_text().strip() == "ALL DONE"
    assert _by_type(records, "result")[0]["result"].strip() == "ALL DONE"


def test_a_tool_use_block_carries_the_id_its_result_is_keyed_by():
    """Without an ``id`` on the tool_use block, the run's whole tool telemetry reads as zero.

    The transcript's ``tool_result`` has always carried ``tool_use_id``; the ``tool_use`` it pairs with
    carried no ``id`` at all, so the join key existed on one side only. aet's stream parser records a
    call only ``if tc.tool_use_id``, and per-call latency is measured as the gap between the two blocks
    -- so the atlas round of 2026-09-04 reported ``tool_call_count=0`` and ``unique_tools_used=[]`` for
    a round that made 125 tool calls, while an independent transcript audit of the same bytes counted
    93 Bash, 31 Edit and 1 web_search. Nothing failed; the number was simply zero.
    """
    from merlin_experiments.phase1.providers import codex_agent as CA

    cmd = CA._tool_block({"type": CA.ITEM_COMMAND_EXECUTION, "command": "ls"}, "codex_tool_item_7")
    assert cmd["id"] == "codex_tool_item_7", "the tool_use block must carry its pairing id"
    assert cmd["name"] == "Bash"

    edit = CA._tool_block({"type": CA.ITEM_FILE_CHANGE, "changes": []}, "codex_tool_item_8")
    assert edit["id"] == "codex_tool_item_8"

    other = CA._tool_block({"type": "web_search", "query": "x"}, "codex_tool_item_9")
    assert other["id"] == "codex_tool_item_9", "the fallback branch must carry an id too"


def test_aet_counts_the_tool_calls_the_transcript_contains():
    """End-to-end over the parser that actually consumes the transcript, not just the block shape."""
    import json

    from merlin_experiments.phase1.providers import codex_agent as CA

    parse_stream = pytest.importorskip("aet.tracking.claude_stream").parse_stream

    items = [{"type": CA.ITEM_COMMAND_EXECUTION, "command": "ls", "id": f"item_{i}"} for i in range(3)]
    lines = []
    for it in items:
        tid = f"codex_tool_{it['id']}"
        lines.append(
            json.dumps(
                {"type": "assistant", "message": {"id": tid, "model": "m", "content": [CA._tool_block(it, tid)]}}
            )
        )
        lines.append(
            json.dumps(
                {
                    "type": "user",
                    "message": {
                        "content": [{"type": "tool_result", "tool_use_id": tid, "content": "ok", "is_error": False}]
                    },
                }
            )
        )
    result = parse_stream("\n".join(lines))
    assert result.tool_call_count == 3, "every emitted tool_use must reach the telemetry store"
    assert result.unique_tools_used == ["Bash"]


def test_the_cli_version_is_recorded_in_the_init_record_and_the_summary(tmp_path):
    """Which CLI parsed this stream is part of the run's provenance.

    ``unknown_types`` DETECTS contract drift; it cannot say what to compare a drifted run against.
    Every event name this driver reads belongs to a particular CLI, so a run that does not record
    which one produced its stream cannot be attributed to a contract afterwards -- the same reason a
    hardware verdict records its RTL revision rather than just its number.
    """
    script = _fake_codex(tmp_path, _stream(), version="codex-cli 9.9.9")
    _rc, tpath, records = _run(tmp_path, script)

    init = _by_type(records, "system")[0]
    assert init["cli_version"] == "codex-cli 9.9.9"

    summary = _by_type(records, "codex_summary")[0]
    assert summary["cli_version"] == "codex-cli 9.9.9"
    on_disk = json.loads((tpath.parent / "round_00.codex_summary.json").read_text())
    assert on_disk["cli_version"] == "codex-cli 9.9.9", "the stamp must survive to the artifact"


def test_an_unaskable_binary_records_none_rather_than_a_guess(tmp_path):
    """A fabricated version stamp on every run is worse than an absent one."""
    assert CA.cli_version(str(tmp_path / "no-such-codex-binary")) is None


def test_the_version_is_asked_once_per_binary(tmp_path):
    """The stamp must not cost a subprocess per round."""
    script = _fake_codex(tmp_path, _stream(), version="codex-cli 1.2.3")
    calls = tmp_path / "version_calls"
    assert CA.cli_version(str(script)) == "codex-cli 1.2.3"
    assert CA.cli_version(str(script)) == "codex-cli 1.2.3"
    assert str(script) in CA._CLI_VERSION
    assert not calls.exists()
