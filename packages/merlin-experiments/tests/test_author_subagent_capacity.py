"""The author's helper-thread ceiling is a recorded launch setting that reaches the frozen Codex home."""

from __future__ import annotations

import dataclasses
import json
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


def _pool(tmp_path, cpus: str) -> dict:
    slots = tmp_path / "slots"
    slots.mkdir(exist_ok=True)
    return {"MERLIN_COMPUTE_CPUS": cpus, "MERLIN_COMPUTE_CPU_SLOTS": str(slots)}


def test_default_is_derived_from_the_compute_pool_and_capped(tmp_path):
    options = parse_options(CODEX)
    assert options.author_max_subagents is None
    small = AS.selection(options, _pool(tmp_path, "4,5,6,7"))
    assert small["max_concurrent_threads_per_session"] == 4
    assert small["source"] == "compute_pool" and small["compute_pool_cpus"] == 4
    large = AS.selection(options, _pool(tmp_path, ",".join(str(cpu) for cpu in range(30))))
    assert large["max_concurrent_threads_per_session"] == AS.DEFAULT_CEILING == 16
    unpooled = AS.selection(options, {})
    assert (unpooled["max_concurrent_threads_per_session"], unpooled["source"]) == (16, "default")
    assert unpooled["delegation_prompt"] is False and unpooled["multi_agent"] is True
    assert unpooled["provider_default_when_unset"] == 3


def test_explicit_selections_take_precedence_in_order(tmp_path):
    environ = {**_pool(tmp_path, "0,1"), AS.ENV: "9"}
    assert AS.selection(parse_options(CODEX), environ)["source"] == "environment"
    assert AS.selection(parse_options(CODEX), environ)["max_concurrent_threads_per_session"] == 9
    option = AS.selection(parse_options([*CODEX, "--author-max-subagents", "12"]), environ)
    assert (option["max_concurrent_threads_per_session"], option["source"]) == (12, "option")
    delegated = AS.selection(parse_options([*CODEX, "--codex-max-subagents", "2"]), environ)
    assert (delegated["max_concurrent_threads_per_session"], delegated["source"]) == (2, "codex_max_subagents")
    assert delegated["delegation_prompt"] is True
    agreed = parse_options([*CODEX, "--codex-max-subagents", "2", "--author-max-subagents", "2"])
    assert AS.selection(agreed, environ)["max_concurrent_threads_per_session"] == 2
    with pytest.raises(ValueError, match="conflicts with the --codex-max-subagents"):
        AS.selection(parse_options([*CODEX, "--codex-max-subagents", "2", "--author-max-subagents", "3"]), {})
    for raw in ("0", "-1", "two", "1.5"):
        with pytest.raises(ValueError, match="positive integer"):
            AS.selection(parse_options(CODEX), {AS.ENV: raw})
    with pytest.raises(SystemExit):
        parse_options([*CODEX, "--author-max-subagents", "0"])


def test_only_isolated_codex_runs_record_a_capacity():
    assert AS.selection(parse_options(["--run-id", "test"]), {AS.ENV: "4"}) is None
    with pytest.raises(ValueError, match="requires the Codex driver"):
        AS.selection(parse_options(["--run-id", "test", "--author-max-subagents", "4"]), {})
    unsandboxed = ["--run-id", "test", "--driver", "codex", "--sandbox", "none"]
    assert AS.selection(parse_options(unsandboxed), {}) is None
    with pytest.raises(ValueError, match="isolated bwrap"):
        AS.selection(parse_options(unsandboxed), {AS.ENV: "4"})


def _record(options) -> dict:
    request = SimpleNamespace(options=options, launcher_argv=(), context=SimpleNamespace(readback_policy=None))
    return session.RunRequest.run_config.fget(request)


def test_the_run_config_records_the_effective_value_and_resume_keeps_it(tmp_path, monkeypatch):
    for key in (AS.ENV, "MERLIN_COMPUTE_CPUS", "MERLIN_COMPUTE_CPU_SLOTS", "MERLIN_GRADE_CPUS"):
        monkeypatch.delenv(key, raising=False)
    options = parse_options(CODEX)
    record = _record(options)
    assert record["author_subagents"]["max_concurrent_threads_per_session"] == 16
    assert record["author_subagents"]["source"] == "default"
    assert "author_subagents" not in _record(parse_options(["--run-id", "test"]))

    # A derived default adopts the recorded value even when the pool changed since launch.
    monkeypatch.setenv("MERLIN_COMPUTE_CPUS", "0,1")
    monkeypatch.setenv("MERLIN_COMPUTE_CPU_SLOTS", str(tmp_path))
    session._verify_client_selection(options, {"run_config": record})
    # An explicit different selection on resume is refused; the same one is accepted.
    explicit = dataclasses.replace(options, author_max_subagents=4)
    with pytest.raises(RuntimeError, match="author subagent capacity selection changed"):
        session._verify_client_selection(explicit, {"run_config": record})
    session._verify_client_selection(dataclasses.replace(options, author_max_subagents=16), {"run_config": record})
    # A run recorded before the setting existed keeps its provider default unless explicitly reselected.
    legacy = {k: v for k, v in record.items() if k != "author_subagents"}
    session._verify_client_selection(options, {"run_config": legacy})
    with pytest.raises(RuntimeError, match="author subagent capacity selection changed"):
        session._verify_client_selection(explicit, {"run_config": legacy})


def _environment(run_dir, record: dict | None) -> None:
    run_config = {"schedule": "continuous", **({"author_subagents": record} if record is not None else {})}
    (run_dir / "environment.yaml").write_text(yaml.safe_dump({"run_id": run_dir.name, "run_config": run_config}))


def test_the_provider_applies_exactly_the_recorded_value(tmp_path):
    record = AS.selection(parse_options([*CODEX, "--author-max-subagents", "7"]), {})
    assert AS.recorded_capacity(tmp_path) is None
    _environment(tmp_path, None)
    assert AS.recorded_capacity(tmp_path) is None
    _environment(tmp_path, record)
    assert AS.recorded_capacity(tmp_path) == 7
    assert AS.launch_kwargs(tmp_path, "bwrap") == {"subagent_capacity": 7}
    assert AS.launch_kwargs(tmp_path, "none") == {}
    _environment(tmp_path, {**record, "max_concurrent_threads_per_session": 0})
    with pytest.raises(ValueError, match="positive integer"):
        AS.recorded_capacity(tmp_path)
    _environment(tmp_path, {"max_concurrent_threads_per_session": 7})
    with pytest.raises(ValueError, match="malformed"):
        AS.recorded_capacity(tmp_path)


def test_the_recorded_capacity_reaches_the_actual_provider_call(tmp_path, monkeypatch):
    from merlin_experiments.phase1.context import InvocationContext
    from merlin_experiments.phase1.providers import agent_bridge

    from merlin.targetgen import target_experiment

    context = InvocationContext(
        tmp_path, tmp_path / "descriptor", tmp_path, "fixture", tmp_path, tmp_path, tmp_path, ()
    )
    (tmp_path / "TASK.md").write_text("sealed")
    _environment(tmp_path, AS.selection(parse_options([*CODEX, "--author-max-subagents", "5"]), {}))
    config = execution.ExecutionConfig(context, execution.ProviderConfig("codex"), lambda: (), tmp_path / "timing")
    monkeypatch.setattr(execution.FL, "start_brokers", lambda *a: [])
    monkeypatch.setattr(execution.FL, "stop_brokers", lambda *a, **k: None)
    monkeypatch.setattr(target_experiment, "load_target_experiment", lambda *a: object())
    monkeypatch.setattr(agent_bridge, "bridged_name", lambda *a: None)
    received = []
    monkeypatch.setattr(codex_agent, "run_round", lambda *a, **k: (received.append(k), (0, tmp_path / "t"))[1])
    execution.launch(tmp_path, tmp_path, "gpt-6.1-sol", "high", "bwrap", {}, 0, 10, config=config)
    assert received[0]["subagent_capacity"] == 5
    assert "max_concurrent_subagents" not in received[0]


def test_round_capacity_reconciles_the_delegation_bound():
    assert AS.effective_capacity(None, None, "bwrap") is None
    assert AS.effective_capacity(None, 16, "bwrap") == 16
    assert AS.effective_capacity(2, None, "bwrap") == 2
    assert AS.effective_capacity(2, 2, "bwrap") == 2
    with pytest.raises(ValueError, match="conflicts with the recorded"):
        AS.effective_capacity(2, 16, "bwrap")
    for bad in ((0, None, "bwrap"), (None, True, "bwrap"), (None, 4, "none")):
        with pytest.raises(ValueError, match="positive isolated selection"):
            AS.effective_capacity(*bad)
    assert AS.summary_record(None, None) == {
        "max_concurrent_threads_per_session": None,
        "configured": False,
        "delegation_prompt": False,
        "provider_default": 3,
    }
    assert AS.summary_record(16, None)["configured"] is True


def test_the_frozen_home_carries_the_capacity_keys(tmp_path):
    workspace = tmp_path / "compiler"
    workspace.mkdir()
    home = tmp_path / "home"
    record = codex_agent.prepare_codex_home(
        home, model="gpt-6.1-sol", effort="high", workspace=workspace, max_concurrent_subagents=16
    )
    parsed = tomllib.loads((home / "config.toml").read_text())
    assert parsed["features"]["multi_agent"] is True
    assert parsed["agents"]["max_concurrent_threads_per_session"] == 16
    codex_agent._verify_frozen_config(home, record["config_sha256"])


def test_the_installed_cli_reads_the_frozen_capacity_without_a_model_call(tmp_path):
    """Config semantics against the real binary: N helpers render as N + 1 slots including the lead."""
    codex_bin = shutil.which("codex")
    if codex_bin is None:
        pytest.skip("selected Codex CLI is unavailable")
    workspace = tmp_path / "compiler"
    workspace.mkdir()
    rendered = {}
    for count in (None, 16):
        frozen = tmp_path / f"frozen-{count}"
        options = {} if count is None else {"max_concurrent_subagents": count}
        codex_agent.prepare_codex_home(frozen, model="gpt-6.1-sol", effort="low", workspace=workspace, **options)
        # The exact root-level capacity lines of the frozen config, without the candidate permission
        # profile (whose filesystem sandbox needs the outer bwrap mounts to start the helper).
        lines = (frozen / "config.toml").read_text().splitlines()
        fragment = [line for line in lines if line.startswith(("features.", "agents."))]
        assert bool(fragment) is (count is not None)
        home = tmp_path / f"home-{count}"
        home.mkdir()
        (home / "config.toml").write_text("\n".join([*fragment, 'model = "gpt-6.1-sol"', ""]))
        result = subprocess.run(
            [codex_bin, "-C", str(workspace), "debug", "prompt-input", "probe"],
            env={**os.environ, "CODEX_HOME": str(home)},
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            pytest.skip(f"installed CLI cannot render prompt input: {result.stderr[-300:]}")
        text = json.dumps(json.loads(result.stdout))
        rendered[count] = text
    assert "There are 4 available concurrency slots" in rendered[None]
    assert "There are 17 available concurrency slots" in rendered[16]
