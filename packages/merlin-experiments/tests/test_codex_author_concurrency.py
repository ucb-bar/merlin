"""Frozen native author settings only; no provider call or credential material."""

import tomllib

import pytest
from merlin_experiments.phase1.providers import codex_agent as CA


def test_explicit_three_authors_are_in_the_actual_frozen_home(tmp_path):
    workspace = tmp_path / "compiler"
    workspace.mkdir()
    auth = tmp_path / "auth.json"
    auth.write_text("{}")
    home = tmp_path / "home"
    record = CA.prepare_codex_home(
        home, model="gpt-6.1-sol", effort="xhigh", workspace=workspace,
        candidate_read_paths=("/public",), auth_source=auth, max_concurrent_subagents=3,
    )
    parsed = tomllib.loads((home / "config.toml").read_text())
    assert parsed["agents"]["max_concurrent_threads_per_session"] == 3
    assert parsed["agents"]["enabled"] is True
    assert parsed["features"]["multi_agent"] is True
    assert parsed["model"] == "gpt-6.1-sol" and parsed["model_reasoning_effort"] == "xhigh"
    assert "agents" not in parsed.get("permissions", {})
    CA._verify_frozen_config(home, record["config_sha256"])
    assert not (home / "auth.json").exists()


@pytest.mark.parametrize("count", [0, -1, True, 3.0, "3"])
def test_malformed_concurrency_stops_before_home_creation(tmp_path, count):
    home = tmp_path / "home"
    with pytest.raises(ValueError, match="positive integer"):
        CA.prepare_codex_home(
            home, model="gpt-6.1-sol", effort="xhigh", workspace=tmp_path, max_concurrent_subagents=count,
        )
    assert not home.exists()
