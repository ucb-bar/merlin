"""A helper reasoning effort is an explicit, recorded launch setting; unset leaves helpers at the lead's."""

from __future__ import annotations

import dataclasses
import os
import shutil
import subprocess
import tomllib
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase1 import author_subagents as AS
from merlin_experiments.phase1 import session
from merlin_experiments.phase1.options import parse_options
from merlin_experiments.phase1.providers import codex_agent, execution

CODEX = ["--run-id", "test", "--driver", "codex", "--sandbox", "bwrap"]


def _record(options) -> dict:
    request = SimpleNamespace(options=options, launcher_argv=(), context=SimpleNamespace(readback_policy=None))
    return session.RunRequest.run_config.fget(request)


def _environment(run_dir, record: dict | None) -> None:
    run_config = {"schedule": "continuous", **({"author_subagents": record} if record is not None else {})}
    (run_dir / "environment.yaml").write_text(yaml.safe_dump({"run_id": run_dir.name, "run_config": run_config}))


def test_unset_records_nothing_and_an_explicit_effort_is_recorded_with_its_scope():
    assert parse_options(CODEX).author_helper_effort is None
    assert AS.HELPER_EFFORT_KEY not in AS.selection(parse_options(CODEX), {})
    record = AS.selection(parse_options([*CODEX, "--author-helper-effort", "low"]), {})
    assert record[AS.HELPER_EFFORT_KEY] == "low"
    assert "full-history forks inherit" in record["helper_effort_scope"]
    for bad in ("", "Low", "low effort", "l0w", "низкий"):
        with pytest.raises(ValueError, match="lowercase reasoning-effort word"):
            AS.selection(parse_options([*CODEX, "--author-helper-effort", bad]), {})


def test_only_an_isolated_codex_run_accepts_a_helper_effort():
    with pytest.raises(ValueError, match="requires the Codex driver"):
        AS.selection(parse_options(["--run-id", "test", "--author-helper-effort", "low"]), {})
    unsandboxed = ["--run-id", "test", "--driver", "codex", "--sandbox", "none", "--author-helper-effort", "low"]
    with pytest.raises(ValueError, match="isolated bwrap"):
        AS.selection(parse_options(unsandboxed), {})


def test_resume_refuses_a_changed_explicit_effort_and_keeps_the_recorded_one(tmp_path, monkeypatch):
    for key in (AS.ENV, "MERLIN_COMPUTE_CPUS", "MERLIN_COMPUTE_CPU_SLOTS", "MERLIN_GRADE_CPUS"):
        monkeypatch.delenv(key, raising=False)
    options = parse_options([*CODEX, "--author-helper-effort", "low"])
    record = _record(options)
    assert record["author_subagents"][AS.HELPER_EFFORT_KEY] == "low"
    session._verify_client_selection(options, {"run_config": record})
    # Omitting the option on resume keeps the recorded selection (the provider reads the record).
    session._verify_client_selection(dataclasses.replace(options, author_helper_effort=None), {"run_config": record})
    with pytest.raises(RuntimeError, match="helper reasoning effort selection changed"):
        session._verify_client_selection(dataclasses.replace(options, author_helper_effort="medium"),
                                         {"run_config": record})
    # A run launched without a helper effort cannot gain one on resume.
    plain = _record(parse_options(CODEX))
    with pytest.raises(RuntimeError, match="helper reasoning effort selection changed"):
        session._verify_client_selection(options, {"run_config": plain})
    legacy = {k: v for k, v in plain.items() if k != "author_subagents"}
    with pytest.raises(RuntimeError, match="helper reasoning effort selection changed"):
        session._verify_client_selection(options, {"run_config": legacy})


def test_the_recorded_effort_reaches_the_provider_call(tmp_path, monkeypatch):
    from merlin_experiments.phase1.context import InvocationContext
    from merlin_experiments.phase1.providers import agent_bridge

    from merlin.targetgen import target_experiment

    record = AS.selection(parse_options([*CODEX, "--author-max-subagents", "5", "--author-helper-effort", "medium"]),
                          {})
    _environment(tmp_path, record)
    assert AS.recorded_helper_effort(tmp_path) == "medium"
    assert AS.launch_kwargs(tmp_path, "bwrap") == {"subagent_capacity": 5, "helper_effort": "medium"}
    assert AS.launch_kwargs(tmp_path, "none") == {}
    _environment(tmp_path, {k: v for k, v in record.items() if k != AS.HELPER_EFFORT_KEY})
    assert AS.recorded_helper_effort(tmp_path) is None
    assert AS.launch_kwargs(tmp_path, "bwrap") == {"subagent_capacity": 5}
    _environment(tmp_path, {**record, AS.HELPER_EFFORT_KEY: 3})
    with pytest.raises(ValueError, match="reasoning-effort word"):
        AS.recorded_helper_effort(tmp_path)

    context = InvocationContext(
        tmp_path, tmp_path / "descriptor", tmp_path, "fixture", tmp_path, tmp_path, tmp_path, ()
    )
    (tmp_path / "TASK.md").write_text("sealed")
    _environment(tmp_path, record)
    config = execution.ExecutionConfig(context, execution.ProviderConfig("codex"), lambda: (), tmp_path / "timing")
    monkeypatch.setattr(execution.FL, "start_brokers", lambda *a: [])
    monkeypatch.setattr(execution.FL, "stop_brokers", lambda *a, **k: None)
    monkeypatch.setattr(target_experiment, "load_target_experiment", lambda *a: object())
    monkeypatch.setattr(agent_bridge, "bridged_name", lambda *a: None)
    received = []
    monkeypatch.setattr(codex_agent, "run_round", lambda *a, **k: (received.append(k), (0, tmp_path / "t"))[1])
    execution.launch(tmp_path, tmp_path, "gpt-6.1-sol", "high", "bwrap", {}, 0, 10, config=config)
    assert received[0]["helper_effort"] == "medium" and received[0]["subagent_capacity"] == 5


def test_the_frozen_home_writes_the_key_only_when_selected(tmp_path):
    workspace = tmp_path / "compiler"
    workspace.mkdir()
    plain = codex_agent.prepare_codex_home(tmp_path / "plain", model="gpt-6.1-sol", effort="high",
                                           workspace=workspace, max_concurrent_subagents=16)
    text = (tmp_path / "plain" / "config.toml").read_text()
    assert AS.HELPER_EFFORT_KEY not in text
    # Unchanged bytes for runs without the option: the historical capacity lines, in order.
    assert text.startswith("features.multi_agent = true\nagents.enabled = true\n"
                           "agents.max_concurrent_threads_per_session = 16\nmodel = ")
    selected = codex_agent.prepare_codex_home(tmp_path / "low", model="gpt-6.1-sol", effort="high",
                                              workspace=workspace, max_concurrent_subagents=16, helper_effort="low")
    parsed = tomllib.loads((tmp_path / "low" / "config.toml").read_text())
    assert parsed["agents"][AS.HELPER_EFFORT_KEY] == "low" and parsed["model_reasoning_effort"] == "high"
    assert selected["config_sha256"] != plain["config_sha256"]
    codex_agent._verify_frozen_config(tmp_path / "low", selected["config_sha256"])
    with pytest.raises(ValueError, match="requires the recorded subagent capacity"):
        codex_agent.prepare_codex_home(tmp_path / "bad", model="gpt-6.1-sol", effort="high", workspace=workspace,
                                       helper_effort="low")


def test_summary_and_lead_note_state_the_setting_and_its_scope():
    assert AS.HELPER_EFFORT_KEY not in AS.summary_record(16, None)
    summary = AS.summary_record(16, 16, "low")
    assert summary[AS.HELPER_EFFORT_KEY] == "low" and "fork_turns none" in summary["helper_effort_scope"]
    assert AS.helper_effort_note(None) == ""
    note = AS.helper_effort_note("low")
    assert 'fork_turns "none"' in note and "effort low" in note and "full-history forks keep your own" in note


def test_the_installed_cli_accepts_the_key_as_a_typed_string(tmp_path):
    """The key is the CLI's own: a string loads; a non-string is refused at that key, not read as a role."""
    codex_bin = shutil.which("codex")
    if codex_bin is None:
        pytest.skip("selected Codex CLI is unavailable")
    workspace = tmp_path / "compiler"
    workspace.mkdir()
    results = {}
    for name, value in (("string", '"low"'), ("integer", "5")):
        home = tmp_path / f"home-{name}"
        home.mkdir()
        (home / "config.toml").write_text(
            f'agents.max_concurrent_threads_per_session = 4\nagents.{AS.HELPER_EFFORT_KEY} = {value}\n'
            'model = "gpt-6.1-sol"\n')
        results[name] = subprocess.run(
            [codex_bin, "-C", str(workspace), "debug", "prompt-input", "probe"],
            env={**os.environ, "CODEX_HOME": str(home)}, capture_output=True, text=True, timeout=60)
    if results["string"].returncode != 0:
        pytest.skip(f"installed CLI cannot render prompt input: {results['string'].stderr[-300:]}")
    refused = results["integer"]
    assert refused.returncode != 0 and f"agents.{AS.HELPER_EFFORT_KEY}" in refused.stderr
    assert "AgentRoleToml" not in refused.stderr
