"""The team-wide Phase 1 agent selection (Codex, gpt-6.1-sol, effort high) reaches codex-cli unchanged."""

import tomllib

import pytest
from merlin_experiments.adapters import ADAPTERS
from merlin_experiments.phase1.providers import agent_bridge, codex_agent

TEAM_MODEL = "gpt-6.1-sol"
TEAM_EFFORT = "high"


@pytest.fixture(autouse=True)
def _no_ambient_model_overrides(monkeypatch):
    for name in ("CODEX_MODEL_MAP", "CODEX_MODEL", "MERLIN_FORCE_BRIDGE", "CODEX_SERVICE_TIER"):
        monkeypatch.delenv(name, raising=False)


def test_team_model_is_native_and_passes_through_unchanged():
    assert codex_agent.resolve_model(TEAM_MODEL) == TEAM_MODEL
    assert agent_bridge.bridged_name(TEAM_MODEL, "codex") is None
    assert agent_bridge.codex_config_fragment(TEAM_MODEL) == ""
    assert agent_bridge.codex_model_name(TEAM_MODEL) == TEAM_MODEL


def test_exec_and_resume_argv_carry_exact_model_and_effort(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    effort = ["-c", f'model_reasoning_effort="{TEAM_EFFORT}"']
    first = codex_agent.build_cmd(
        workspace, model=TEAM_MODEL, effort=TEAM_EFFORT, final_path=tmp_path / "final.md", sandbox="bwrap"
    )
    assert first[first.index("--model") + 1] == TEAM_MODEL
    assert any(first[i : i + 2] == effort for i in range(len(first) - 1))
    assert "--strict-config" in first
    resumed = codex_agent.build_resume_cmd(
        workspace,
        model=TEAM_MODEL,
        effort=TEAM_EFFORT,
        final_path=tmp_path / "final.md",
        sandbox="bwrap",
        thread_id="thread-fixture",
    )
    assert resumed[resumed.index("--model") + 1] == TEAM_MODEL
    assert any(resumed[i : i + 2] == effort for i in range(len(resumed) - 1))


def test_frozen_codex_home_records_exact_model_and_effort(tmp_path, monkeypatch):
    real = tmp_path / "real"
    real.mkdir()
    (real / "auth.json").write_text('{"fixture": "never-copied"}')
    monkeypatch.setenv("CODEX_HOME", str(real))
    workspace = tmp_path / "ws"
    workspace.mkdir()
    report = codex_agent.prepare_codex_home(
        tmp_path / "home", model=TEAM_MODEL, effort=TEAM_EFFORT, workspace=workspace
    )
    config = tomllib.loads((tmp_path / "home" / "config.toml").read_text())
    assert config["model"] == TEAM_MODEL
    assert config["model_reasoning_effort"] == TEAM_EFFORT
    assert config["default_permissions"] == "merlin-candidate"
    assert report["auth_copied"] is False and not (tmp_path / "home" / "auth.json").exists()


def test_capsule_bench_accepts_team_agent_selection(tmp_path):
    config = {
        "require_reviewed_corpus": True,
        "descriptor": str(tmp_path / "target_experiment.yaml"),
        "level": "EL4",
        "bundle": "merlin_assisted_rtlchecks_public_v0",
        "bundle_manifest": str(tmp_path / "input_bundle_manifest.yaml"),
        "oracle_timing": str(tmp_path / ".oracle_timing.gemmini.json"),
        "model": TEAM_MODEL,
        "effort": TEAM_EFFORT,
        "driver": "codex",
        "provider": "subscription",
        "max_wall_s": 43200,
        "round_timeout": 43200,
        "grade_interval": 900,
        "sim_max_jobs": 5,
    }
    ADAPTERS["capsule_bench"].validate(config)
    assert config["arm"] == "merlin_assisted" and config["treatment"] == "rtlchecks"
    assert (config["driver"], config["model"], config["effort"]) == ("codex", TEAM_MODEL, TEAM_EFFORT)
