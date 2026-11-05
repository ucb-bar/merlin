"""A failed round retry retains its compiler but never reuses provider history."""
import hashlib
import json
from pathlib import Path

import pytest

from merlin_experiments.phase1.providers.codex_runtime import failed_round_retry


def previous(tmp_path, *, code=75):
    root, run = tmp_path / "homes", tmp_path / "run"
    home = root / "run_r00"
    home.mkdir(parents=True)
    (home / "sessions").mkdir()
    (home / "sessions/private-history").write_text("old author only")
    (home / "config.toml").write_text("old pinned config")
    rounds = run / "rounds"
    rounds.mkdir(parents=True)
    summary = rounds / "round_00.codex_summary.json"
    summary.write_text(json.dumps({"exit_code": code, "codex_home": {
        "codex_home": str(home),
        "config_sha256": hashlib.sha256((home / "config.toml").read_bytes()).hexdigest(),
    }}))
    return root, run, home, summary


def test_retry_allocates_empty_unique_home_and_pins_failed_record(tmp_path):
    root, run, old, summary = previous(tmp_path)
    original = summary.read_bytes()
    home, record = failed_round_retry(root, run, 0)
    other, second = failed_round_retry(root, run, 0)
    assert home != other and home.parent == root and home != old
    assert list(home.iterdir()) == [] and list(other.iterdir()) == []
    assert home.stat().st_mode & 0o777 == 0o700
    assert record["previous_summary_sha256"] == hashlib.sha256(original).hexdigest()
    assert record["logical_round"] == 0 and not record["reused_session"]
    assert summary.read_bytes() == original
    assert (old / "sessions/private-history").read_text() == "old author only"


@pytest.mark.parametrize("damage", ["successful", "missing", "foreign", "config", "symlink"])
def test_unbound_or_successful_home_cannot_authorize_retry(tmp_path, damage):
    root, run, home, summary = previous(tmp_path)
    if damage == "missing":
        summary.unlink()
    elif damage == "config":
        (home / "config.toml").write_text("changed")
    elif damage == "symlink":
        original = summary.with_suffix(".saved")
        summary.rename(original)
        summary.symlink_to(original)
    else:
        data = json.loads(summary.read_text())
        if damage == "successful": data["exit_code"] = 0
        else: data["codex_home"]["codex_home"] = str(tmp_path / "foreign")
        summary.write_text(json.dumps(data))
    with pytest.raises(ValueError): failed_round_retry(root, run, 0)
    assert list(root.iterdir()) == [home]


def test_real_driver_retry_uses_new_home_and_preserves_old_artifacts(tmp_path, monkeypatch):
    from test_codex_driver import _fake_codex, _stream
    from merlin_experiments.phase1.providers import codex_agent as CA
    from merlin_experiments.phase1.telemetry import author_agents

    root, run, old, summary = previous(tmp_path)
    original = summary.read_bytes()
    (run / "rounds/round_00.codex_events.raw.jsonl").write_text(
        json.dumps({"type": "thread.started", "thread_id": "old-lead"}) + "\n")
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "TASK.md").write_text("synthetic driver test")
    script = _fake_codex(tmp_path, _stream())

    def prepare(home, **kwargs):
        assert home != old and not list(home.iterdir())
        (home / "sessions").mkdir()
        config = home / "config.toml"
        config.write_text("synthetic frozen config")
        return {"codex_home": str(home), "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest()}

    monkeypatch.setattr(CA, "prepare_codex_home", prepare)
    monkeypatch.setattr(CA, "_preflight_candidate_sandbox", lambda *args, **kwargs: None)
    monkeypatch.setattr(CA, "authoring_prompt", lambda: "test")
    rc, transcript = CA.run_round(
        ws, run, "gpt-6.1-sol", {}, None, "bwrap", 0, 30,
        codex_binary=script, codex_home_root=root, require_fresh_home=True,
        allow_failed_home_retry=True, runtime_binds=lambda home: [],
        sandbox_command=lambda inner, *args, **kwargs: inner,
    )
    assert rc == 0 and ".attempt_" in transcript.name
    assert summary.read_bytes() == original
    records = [json.loads(line) for line in transcript.read_text().splitlines()]
    init = next(row for row in records if row.get("subtype") == "init")
    result = next(row for row in records if row.get("type") == "codex_summary")
    assert result["codex_home"]["attempt"] == init["home_attempt"]
    assert result["codex_home"]["codex_home"] != str(old)
    leads = author_agents.lead_thread_ids(run)
    assert "old-lead" in leads and len(leads) == 2
    roots = [path.parent for _, path in author_agents.session_roots(run)]
    assert old in roots and Path(result["codex_home"]["codex_home"]) in roots
    # Success now becomes latest authority: a further retry cannot conceal it.
    with pytest.raises(ValueError, match="successful"):
        failed_round_retry(root, run, 0)
