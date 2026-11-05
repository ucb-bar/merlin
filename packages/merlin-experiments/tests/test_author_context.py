"""The Codex author context selection is recorded once and reproduced exactly: ceiling, prompt, skills."""

from __future__ import annotations

import dataclasses
import json
import os
import select
import shutil
import subprocess
import time
import tomllib
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase1 import launch_config as lc
from merlin_experiments.phase1 import session, task_disclosure
from merlin_experiments.phase1.options import parse_options
from merlin_experiments.phase1.providers import codex_agent, execution
from merlin_experiments.phase1.providers import codex_context as CTX

CODEX = ["--run-id", "test", "--driver", "codex", "--sandbox", "bwrap"]


def _record(options) -> dict:
    request = SimpleNamespace(options=options, launcher_argv=(), context=SimpleNamespace(readback_policy=None))
    return session.RunRequest.run_config.fget(request)


def _environment(run_dir, record: dict | None) -> None:
    run_config = {"schedule": "continuous", **({CTX.KEY: record} if record is not None else {})}
    (run_dir / "environment.yaml").write_text(yaml.safe_dump({"run_id": run_dir.name, "run_config": run_config}))


def test_an_isolated_codex_run_records_the_measured_default_and_its_prompt():
    record = _record(parse_options(CODEX))[CTX.KEY]
    assert record[CTX.LIMIT_KEY] == CTX.DEFAULT_COMPACT_LIMIT and record["compact_limit_source"] == "default"
    assert record[CTX.PROMPT_KEY] == CTX.compact_prompt() and record["task_skills"] is True
    # The prompt keeps the CLI's default handoff items and names where feedback and state live.
    for text in ("CONTEXT CHECKPOINT COMPACTION", "What remains to be done", "qa/verdict.json",
                 "qa/latest_grade_summary.txt", "failure_clusters", "task-*", "helper agent"):
        assert text in record[CTX.PROMPT_KEY]
    explicit = CTX.selection(parse_options([*CODEX, "--author-compact-limit", "150_000", "--author-task-skills",
                                            "off"]))
    assert explicit[CTX.LIMIT_KEY] == 150_000 and explicit["task_skills"] is False
    off = CTX.selection(parse_options([*CODEX, "--author-compact-limit", "off"]))
    assert off[CTX.LIMIT_KEY] is None and CTX.PROMPT_KEY not in off and CTX.home_config(off) == ""


def test_other_drivers_record_nothing_and_refuse_the_options():
    assert CTX.KEY not in _record(parse_options(["--run-id", "test"]))
    assert CTX.selection(parse_options(["--run-id", "t", "--driver", "codex", "--sandbox", "none"])) is None
    for argv in (["--run-id", "t", "--author-compact-limit", "150000"],
                 ["--run-id", "t", "--driver", "codex", "--sandbox", "none", "--author-task-skills", "on"],
                 [*CODEX, "--author-compact-limit", "5000"], [*CODEX, "--author-compact-limit", "lots"]):
        with pytest.raises(SystemExit):
            parse_options(argv)


def test_resume_keeps_the_record_and_refuses_an_explicit_change():
    options = parse_options([*CODEX, "--author-compact-limit", "150000"])
    record = _record(options)
    session._verify_client_selection(options, {"run_config": record})
    session._verify_client_selection(dataclasses.replace(options, author_compact_limit=None), {"run_config": record})
    with pytest.raises(RuntimeError, match="author context selection changed"):
        session._verify_client_selection(dataclasses.replace(options, author_compact_limit="200000"),
                                         {"run_config": record})
    with pytest.raises(RuntimeError, match="task_skills"):
        session._verify_client_selection(dataclasses.replace(options, author_task_skills="off"),
                                         {"run_config": record})
    legacy = {key: value for key, value in record.items() if key != CTX.KEY}
    session._verify_client_selection(dataclasses.replace(options, author_compact_limit=None), {"run_config": legacy})
    with pytest.raises(RuntimeError, match="author context selection changed"):
        session._verify_client_selection(options, {"run_config": legacy})


def test_the_recorded_selection_reaches_the_provider_and_fails_closed_when_altered(tmp_path, monkeypatch):
    from merlin_experiments.phase1.context import InvocationContext
    from merlin_experiments.phase1.providers import agent_bridge

    from merlin.targetgen import target_experiment

    record = CTX.selection(parse_options(CODEX))
    _environment(tmp_path, record)
    assert CTX.launch_kwargs(tmp_path, "bwrap") == {CTX.KEY: record} and CTX.launch_kwargs(tmp_path, "none") == {}
    context = InvocationContext(tmp_path, tmp_path / "descriptor", tmp_path, "fixture", tmp_path, tmp_path, tmp_path,
                                ())
    (tmp_path / "TASK.md").write_text("sealed")
    config = execution.ExecutionConfig(context, execution.ProviderConfig("codex"), lambda: (), tmp_path / "timing")
    monkeypatch.setattr(execution.FL, "start_brokers", lambda *a: [])
    monkeypatch.setattr(execution.FL, "stop_brokers", lambda *a, **k: None)
    monkeypatch.setattr(target_experiment, "load_target_experiment", lambda *a: object())
    monkeypatch.setattr(agent_bridge, "bridged_name", lambda *a: None)
    received = []
    monkeypatch.setattr(codex_agent, "run_round", lambda *a, **k: (received.append(k), (0, tmp_path / "t"))[1])
    execution.launch(tmp_path, tmp_path, "gpt-6.1-sol", "high", "bwrap", {}, 0, 10, config=config)
    assert received[0][CTX.KEY] == record
    _environment(tmp_path, {**record, CTX.PROMPT_KEY: record[CTX.PROMPT_KEY] + " edited"})
    with pytest.raises(ValueError, match="does not match its digest"):
        CTX.launch_kwargs(tmp_path, "bwrap")
    _environment(tmp_path, None)
    assert CTX.launch_kwargs(tmp_path, "bwrap") == {}  # a run launched before the setting keeps its home


def test_the_frozen_home_carries_the_recorded_keys_and_is_unchanged_without_them(tmp_path):
    workspace = tmp_path / "compiler"
    workspace.mkdir()
    plain = codex_agent.prepare_codex_home(tmp_path / "plain", model="gpt-6.1-sol", effort="high",
                                           workspace=workspace, max_concurrent_subagents=16)
    text = (tmp_path / "plain" / "config.toml").read_text()
    assert CTX.LIMIT_KEY not in text and CTX.PROMPT_KEY not in text
    record = CTX.selection(parse_options(CODEX))
    selected = codex_agent.prepare_codex_home(tmp_path / "ctx", model="gpt-6.1-sol", effort="high",
                                              workspace=workspace, max_concurrent_subagents=16,
                                              author_context=record)
    written = (tmp_path / "ctx" / "config.toml").read_text()
    capacity_line = "agents.max_concurrent_threads_per_session = 16\n"
    # Exactly the historical bytes plus the two root-level keys after the capacity lines.
    assert written.replace(str(tmp_path / "ctx"), str(tmp_path / "plain")) == text.replace(
        capacity_line, capacity_line + CTX.home_config(record))
    parsed = tomllib.loads(written)
    assert parsed[CTX.LIMIT_KEY] == CTX.DEFAULT_COMPACT_LIMIT and parsed[CTX.PROMPT_KEY] == record[CTX.PROMPT_KEY]
    assert parsed["agents"]["max_concurrent_threads_per_session"] == 16 and parsed["model_reasoning_effort"] == "high"
    assert selected["config_sha256"] != plain["config_sha256"]
    codex_agent._verify_frozen_config(tmp_path / "ctx", selected["config_sha256"])


def test_the_lead_is_told_about_the_deliveries_in_both_delegation_paths():
    record = CTX.selection(parse_options(CODEX))
    note = CTX.round_note(record, 16)
    assert 'fork_turns "none"' in note and 'fork_turns omitted or "all"' in note and "The choice is yours" in note
    assert "task-*" in note and "qa/latest_grade_summary.txt" in note and "175,000 tokens" in note
    # The staged author skills add one optional inventory step (no gate).
    assert (CTX.LIBRARY_INVENTORY_NOTE in note) == (record.get("feedback_skill") is not None)
    assert "fork_turns" not in CTX.round_note(record, None) and CTX.round_note(None, 16) == ""
    no_skills = CTX.selection(parse_options([*CODEX, "--author-task-skills", "off", "--author-compact-limit", "off"]))
    assert CTX.round_note(no_skills, None) == "" and CTX.continuation_note(no_skills) == ""
    assert "task-*" in CTX.continuation_note(record)
    # The legacy explicit delegation request keeps its exact bytes; the guidance is appended after it.
    assert codex_agent.delegation_user_prompt is CTX.delegation_user_prompt
    assert codex_agent.delegation_user_prompt("P", 4).startswith("P\n\nI explicitly request native sub-agent")


def test_launch_config_renders_the_controller_options_only_when_declared(tmp_path):
    from test_phase1_launch_config import _checkout

    plain = lc.build(lc.load(_checkout(tmp_path / "plain")), task_root=tmp_path, prep=tmp_path, environ={"HOME": "/h"})
    assert "--author-compact-limit" not in plain.argv and "--author-task-skills" not in plain.argv
    config = _checkout(tmp_path / "set", lambda doc: doc["author"].update(compact_limit=150000, task_skills=False))
    launch = lc.build(lc.load(config), task_root=tmp_path, prep=tmp_path, environ={"HOME": "/h"})
    assert launch.argv[launch.argv.index("--author-compact-limit") + 1] == "150000"
    assert launch.argv[launch.argv.index("--author-task-skills") + 1] == "off"
    with pytest.raises(lc.ConfigError, match="compact_limit"):
        lc.load(_checkout(tmp_path / "small", lambda doc: doc["author"].update(compact_limit=10)))


def _codex() -> str:
    codex_bin = shutil.which("codex")
    if codex_bin is None:
        pytest.skip("selected Codex CLI is unavailable")
    return codex_bin


def _config_read(codex_bin: str, home, workspace) -> dict:
    """The app-server's effective configuration over stdio JSON-RPC; no model request is made."""
    proc = subprocess.Popen([codex_bin, "app-server"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, env={**os.environ, "CODEX_HOME": str(home)}, cwd=workspace,
                            text=True)
    try:
        replies = {}
        for message in ({"id": 1, "method": "initialize", "params": {"clientInfo": {"name": "probe", "version": "0"}}},
                        {"method": "initialized"},
                        {"id": 2, "method": "config/read", "params": {"includeLayers": False, "cwd": str(workspace)}}):
            proc.stdin.write(json.dumps(message) + "\n")
            proc.stdin.flush()
        deadline = time.monotonic() + 60
        while 2 not in replies and time.monotonic() < deadline:
            if select.select([proc.stdout], [], [], 1)[0]:
                line = proc.stdout.readline()
                if not line:
                    break
                reply = json.loads(line)
                replies[reply.get("id")] = reply
        return replies.get(2) or {}
    finally:
        proc.kill()
        proc.wait()


def test_the_installed_cli_loads_the_frozen_keys_and_lists_staged_skills(tmp_path):
    """The real CLI parses the frozen home, types both keys, and discovers workspace skills by name only."""
    codex_bin = _codex()
    workspace = tmp_path / "compiler"
    workspace.mkdir()
    (workspace / "TASK.md").write_text("# Task\n\nShort.\n\n## Reference section\n\nIntro sentence.\n\n"
                                       + "Detail line.\n" * 120)
    (tmp_path / "run").mkdir()
    task_disclosure.stage(workspace, tmp_path / "run")
    record = CTX.selection(parse_options(CODEX))
    home = tmp_path / "home"
    home.mkdir()
    # The frozen home's root-level keys, without its permission profile (whose sandbox helper only exists
    # inside the launch sandbox); the full file's TOML is checked above.
    from merlin_experiments.phase1 import author_subagents

    (home / "config.toml").write_text(author_subagents.home_config(4) + CTX.home_config(record)
                                      + 'model = "gpt-6.1-sol"\nmodel_reasoning_effort = "high"\n')
    rendered = subprocess.run([codex_bin, "-C", str(workspace), "debug", "prompt-input", "probe"],
                              env={**os.environ, "CODEX_HOME": str(home)}, capture_output=True, text=True, timeout=120)
    if rendered.returncode != 0:
        pytest.skip(f"installed CLI cannot render prompt input: {rendered.stderr[-300:]}")
    assert "task-reference-section" in rendered.stdout and task_disclosure.FEEDBACK_SKILL in rendered.stdout
    assert "Detail line." not in rendered.stdout  # bodies are read on demand, not preloaded
    reply = _config_read(codex_bin, home, workspace)
    if "result" not in reply:
        pytest.skip(f"installed app-server did not answer config/read: {reply}")
    effective = reply["result"]["config"]
    assert effective[CTX.LIMIT_KEY] == CTX.DEFAULT_COMPACT_LIMIT
    assert effective[CTX.PROMPT_KEY] == record[CTX.PROMPT_KEY]
    for key, value, expected in ((CTX.LIMIT_KEY, '"many"', "expected i64"), (CTX.PROMPT_KEY, "5", "expected a string")):
        bad = tmp_path / f"bad-{key}"
        bad.mkdir()
        (bad / "config.toml").write_text(f'{key} = {value}\nmodel = "gpt-6.1-sol"\n')
        refused = subprocess.run([codex_bin, "-C", str(workspace), "debug", "prompt-input", "probe"],
                                 env={**os.environ, "CODEX_HOME": str(bad)}, capture_output=True, text=True,
                                 timeout=120)
        assert refused.returncode != 0 and expected in refused.stderr


def _fake_cli(tmp_path):
    """A stand-in CLI that answers --version, replays one completed turn and honours ``-o``."""
    import stat
    import sys

    events = [{"type": "thread.started", "thread_id": "01a01161-dead-beef-0000-000000000001"},
              {"type": "turn.started"},
              {"type": "item.completed", "item": {"id": "i", "type": "agent_message", "text": "DONE"}},
              {"type": "turn.completed", "usage": {"input_tokens": 10, "cached_input_tokens": 0, "output_tokens": 1}}]
    script, stream = tmp_path / "fake_codex", tmp_path / "fake_codex_stream.json"
    stream.write_text(json.dumps(events))
    script.write_text("\n".join([
        f"#!{sys.executable}", "import json, sys", "argv = sys.argv[1:]",
        "if '--version' in argv:", "    print('codex-cli 0.162.0'); sys.exit(0)",
        "out = argv[argv.index('-o') + 1] if '-o' in argv else None", "sys.stdin.read()",
        f"for line in json.load(open({str(stream)!r})):", "    print(json.dumps(line), flush=True)",
        "if out: open(out, 'w').write('DONE')", ""]))
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script


def test_a_round_states_the_deliveries_and_records_the_selection(tmp_path):
    ws, run_dir = tmp_path / "ws", tmp_path / "run"
    ws.mkdir()
    run_dir.mkdir()
    (ws / "TASK.md").write_text("# T\n\n## Long section\n\n" + "x\n" * 600)
    task_disclosure.stage(ws, run_dir)
    record = CTX.selection(parse_options(CODEX))
    rc, tpath = codex_agent.run_round(ws, run_dir, "gpt-6.1-sol", {}, None, "none", 0, 60, effort="low",
                                      codex_binary=_fake_cli(tmp_path), author_context=record)
    assert rc == 0
    prompt = (run_dir / "rounds" / "round_00.prompt.txt").read_text()
    assert prompt.endswith(CTX.round_note(record, None)) and "task-*" in prompt
    records = [json.loads(line) for line in tpath.read_text().splitlines() if line.strip()]
    init = next(r for r in records if r.get("subtype") == "init")
    assert set(init["workspace_skills"]) == {"task-long-section", task_disclosure.FEEDBACK_SKILL}
    summary = json.loads((run_dir / "rounds" / "round_00.codex_summary.json").read_text())
    assert summary["author_context"] == CTX.summary_record(record) and summary["author_context"]["task_skills"]


def test_delegation_joins_only_changes_included_in_the_candidate():
    prompt = CTX.delegation_user_prompt("implement", 4)
    assert "all helper changes" not in prompt
    assert "Unfinished or unrelated helper work does not" in prompt
    assert "keep editing while snapshots grade" in prompt
