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
import stat
import sys
from pathlib import Path

import pytest

from merlin.common.paths import merlin_dir

_HARNESS = merlin_dir() / "experiments/capsule_bench/harness"
if str(_HARNESS) not in sys.path:
    sys.path.insert(0, str(_HARNESS))

import codex_agent as CA  # noqa: E402  (path shim above)


# The measured shape of a real 0.147.0 turn, used as the scripted reply.
_REAL_USAGE = {
    "input_tokens": 36767,
    "cached_input_tokens": 28160,
    "cache_write_input_tokens": 0,
    "output_tokens": 203,
    "reasoning_output_tokens": 90,
}


def _fake_codex(tmp_path: Path, lines: list[dict], *, final: str = "DONE",
                exit_code: int = 0, hang: bool = False) -> Path:
    """Write an executable stand-in for the codex CLI that replays *lines*.

    It also honors ``-o <file>`` so the driver's final-message handling is real.
    """
    script = tmp_path / "fake_codex"
    # The scripted stream goes in a sibling JSON file rather than being embedded
    # in the source: a Python literal is not JSON, so an ``exit_code: None`` in
    # an ``item.started`` payload would render as ``null`` and not parse.
    stream_path = tmp_path / "fake_codex_stream.json"
    stream_path.write_text(json.dumps(lines))
    body = [
        f"#!{sys.executable}",
        "import json, sys, time, os",
        "argv = sys.argv[1:]",
        "out = None",
        "for i, a in enumerate(argv):",
        "    if a in ('-o', '--output-last-message') and i + 1 < len(argv):",
        "        out = argv[i + 1]",
        "sys.stdin.read()",
        f"lines = json.load(open({str(stream_path)!r}))",
        "for line in lines:",
        "    sys.stdout.write(json.dumps(line) + '\\n')",
        "    sys.stdout.flush()",
        *(["time.sleep(600)"] if hang else []),
        "if out:",
        f"    open(out, 'w').write({final!r})",
        f"sys.exit({exit_code})",
    ]
    script.write_text("\n".join(body) + "\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return script


def _stream(usage: dict | None = _REAL_USAGE, *, failed: bool = False) -> list[dict]:
    lines: list[dict] = [
        {"type": "thread.started", "thread_id": "01a01161-dead-beef-0000-000000000001"},
        {"type": "turn.started"},
        {"type": "item.started", "item": {"id": "item_0", "type": "command_execution",
                                          "command": "/bin/bash -lc 'ls'",
                                          "aggregated_output": "", "exit_code": None,
                                          "status": "in_progress"}},
        {"type": "item.completed", "item": {"id": "item_0", "type": "command_execution",
                                            "command": "/bin/bash -lc 'ls'",
                                            "aggregated_output": "TASK.md\n", "exit_code": 0,
                                            "status": "completed"}},
        {"type": "item.completed", "item": {"id": "item_1", "type": "agent_message",
                                            "text": "DONE"}},
    ]
    if failed:
        lines.append({"type": "turn.failed", "error": {"message": "upstream 400"}})
    else:
        lines.append({"type": "turn.completed", "usage": usage})
    return lines


def _run(tmp_path: Path, script: Path, *, sandbox: str = "none", timeout: int = 60,
         instruction_files: tuple[str, ...] = ("TASK.md",)):
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    for name in instruction_files:
        (ws / name).write_text("do the thing\n")
    run_dir = tmp_path / "run"
    run_dir.mkdir(exist_ok=True)
    os.environ["CODEX_BIN"] = str(script)
    try:
        rc, tpath = CA.run_round(ws, run_dir, "claude-opus-4-8", {}, None, sandbox, 0, timeout,
                                 effort="low")
    finally:
        os.environ.pop("CODEX_BIN", None)
    records = [json.loads(l) for l in tpath.read_text().splitlines() if l.strip()]
    return rc, tpath, records


def _by_type(records: list[dict], kind: str) -> list[dict]:
    return [r for r in records if r.get("type") == kind]


# ---------------------------------------------------------------------------
# Command contract — against the flags the installed CLI actually has
# ---------------------------------------------------------------------------


def test_the_nonexistent_approval_flag_is_never_passed():
    """``--ask-for-approval`` was removed by 0.147.0; passing it aborts the run."""
    for sandbox in ("none", "bwrap"):
        cmd = CA.build_cmd(Path("/ws"), model="gpt-5.6-sol", effort="high",
                           final_path=Path("/run/final.txt"), sandbox=sandbox)
        assert "--ask-for-approval" not in cmd
    cmd = CA.build_cmd(Path("/ws"), model="gpt-5.6-sol", effort="",
                       final_path=Path("/run/final.txt"), sandbox="none")
    # The policy is a config override instead.
    assert "-c" in cmd and "approval_policy=never" in cmd


def test_outside_bwrap_codex_keeps_its_own_sandbox():
    cmd = CA.build_cmd(Path("/ws"), model="m", effort="", final_path=Path("/f"), sandbox="none")
    assert "--sandbox" in cmd and "workspace-write" in cmd
    assert "--dangerously-bypass-approvals-and-sandbox" not in cmd


def test_inside_bwrap_the_outer_boundary_is_the_proof_so_codex_bypasses_its_own():
    cmd = CA.build_cmd(Path("/ws"), model="m", effort="", final_path=Path("/f"), sandbox="bwrap")
    assert "--dangerously-bypass-approvals-and-sandbox" in cmd
    assert "--sandbox" not in cmd


def test_the_prompt_is_passed_on_stdin_not_as_an_argv_fragment():
    cmd = CA.build_cmd(Path("/ws"), model="m", effort="", final_path=Path("/f"), sandbox="none")
    assert cmd[-1] == "-", "prompt bytes must stay an artifact, not get mangled by quoting"


def test_effort_is_a_config_override_and_absent_when_unset():
    with_effort = CA.build_cmd(Path("/ws"), model="m", effort="high",
                               final_path=Path("/f"), sandbox="none")
    assert 'model_reasoning_effort="high"' in with_effort
    without = CA.build_cmd(Path("/ws"), model="m", effort="", final_path=Path("/f"), sandbox="none")
    assert not any("model_reasoning_effort" in c for c in without)


@pytest.mark.parametrize("alias,expected", [
    ("gpt-5.6-sol", "gpt-5.6-sol"),
    ("gpt-5.4", "gpt-5.4"),
    ("claude-opus-4-8", CA.DEFAULT_CODEX_MODEL),
    ("", CA.DEFAULT_CODEX_MODEL),
])
def test_model_aliases_resolve_to_a_slug_this_auth_mode_accepts(alias, expected):
    assert CA.resolve_model(alias) == expected


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
        {"input_tokens": 1000, "cached_input_tokens": 600,
         "cache_write_input_tokens": 100, "output_tokens": 10})
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
    tool_uses = [b for r in _by_type(records, "assistant")
                 for b in r["message"].get("content", []) if b.get("type") == "tool_use"]
    assert tool_uses and tool_uses[0]["name"] == "Bash"
    results = [b for r in _by_type(records, "user")
               for b in r["message"]["content"] if b.get("type") == "tool_result"]
    assert results and "TASK.md" in results[0]["content"]
    assert results[0]["is_error"] is False

    # Usage landed on an assistant record in the translated shape.
    usages = [r["message"]["usage"] for r in _by_type(records, "assistant")
              if "usage" in r["message"]]
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


def test_the_raw_event_stream_is_persisted_byte_for_byte(tmp_path):
    _rc, tpath, _records = _run(tmp_path, _fake_codex(tmp_path, _stream()))
    raw = (tpath.parent / "round_00.codex_events.raw.jsonl").read_text()

    assert len(raw.splitlines()) == len(_stream())
    assert '"thread.started"' in raw
    # And a timestamped sibling, since the events carry no time of their own.
    stamped = [json.loads(l) for l in
               (tpath.parent / "round_00.codex_events.timestamped.jsonl").read_text().splitlines()]
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


def test_the_prompt_bytes_are_kept_as_an_artifact(tmp_path):
    _rc, tpath, _records = _run(tmp_path, _fake_codex(tmp_path, _stream()))
    prompt = (tpath.parent / "round_00.prompt.txt").read_text()
    assert "agent_selfcheck.py" in prompt, "the graded self-check instruction must reach the agent"
    assert "submission" in prompt


def test_a_missing_codex_binary_fails_the_round_without_raising(tmp_path):
    ws = tmp_path / "ws"; ws.mkdir()
    run_dir = tmp_path / "run"; run_dir.mkdir()
    os.environ["CODEX_BIN"] = str(tmp_path / "definitely-not-here")
    try:
        rc, tpath = CA.run_round(ws, run_dir, "m", {}, None, "none", 0, 30)
    finally:
        os.environ.pop("CODEX_BIN", None)
    assert rc == 127
    records = [json.loads(l) for l in tpath.read_text().splitlines() if l.strip()]
    assert _by_type(records, "result")[0]["is_error"] is True


# ---------------------------------------------------------------------------
# Instruction parity between arms
# ---------------------------------------------------------------------------


def test_workspace_instruction_files_are_recorded_so_asymmetry_is_visible(tmp_path):
    """Codex reads AGENTS.md where Claude reads CLAUDE.md; an arm that quietly
    got extra instructions is not the same arm."""
    _rc, _tpath, records = _run(tmp_path, _fake_codex(tmp_path, _stream()),
                                instruction_files=("TASK.md", "AGENTS.md"))
    init = _by_type(records, "system")[0]
    assert set(init["workspace_instruction_files"]) == {"TASK.md", "AGENTS.md"}


def test_the_driver_does_not_author_instruction_files_itself(tmp_path):
    _rc, _tpath, _records = _run(tmp_path, _fake_codex(tmp_path, _stream()),
                                 instruction_files=("TASK.md",))
    assert not (tmp_path / "ws" / "AGENTS.md").exists(), "parity is recorded, never manufactured"


# ---------------------------------------------------------------------------
# The isolated CODEX_HOME — keeping prior sessions away from a graded agent
# ---------------------------------------------------------------------------


def test_the_isolated_home_holds_a_frozen_config_and_no_credential(tmp_path):
    info = CA.prepare_codex_home(tmp_path / "home", model="gpt-5.6-sol", effort="high")

    config = (tmp_path / "home" / "config.toml").read_text()
    assert 'model = "gpt-5.6-sol"' in config
    assert 'model_reasoning_effort = "high"' in config
    # The user's own config carries per-project trust levels and notice state;
    # none of it belongs in a measured run.
    assert "trust_level" not in config and "[projects" not in config

    assert info["auth_copied"] is False
    assert not (tmp_path / "home" / "auth.json").exists(), \
        "the credential is bind-mounted read-only, never written into the tree"
    assert info["config_sha256"] and info["isolated_from_real_home"] is True


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
    sources = [binds[i + 1] for i, a in enumerate(binds)
               if a in ("--bind", "--ro-bind") and i + 1 < len(binds)]
    under_real = [s for s in sources if s.startswith(real)]
    assert all(s.startswith(f"{real}/packages") or s == f"{real}/auth.json"
               for s in under_real), f"unexpected bind out of the real home: {under_real}"
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
    ws = tmp_path / "ws"; ws.mkdir()
    run_dir = tmp_path / "run"; run_dir.mkdir()
    os.environ["CODEX_BIN"] = str(script)
    try:
        _rc, tpath = CA.run_round(ws, run_dir, "m", {}, None, "none", 0, 60,
                                  prompt="CANARY: run probe.sh")
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
    ws = tmp_path / "ws"; ws.mkdir()
    (ws / "TASK.md").write_text("x")
    run_dir = tmp_path / "run"; run_dir.mkdir()
    os.environ["CODEX_BIN"] = str(_fake_codex(tmp_path, _stream()))
    try:
        _rc, tpath = CA.run_round(ws, run_dir, "m", {}, None, "none", 0, 60,
                                  subagent_model="gpt-5.4-mini")
    finally:
        os.environ.pop("CODEX_BIN", None)
    records = [json.loads(l) for l in tpath.read_text().splitlines() if l.strip()]
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
