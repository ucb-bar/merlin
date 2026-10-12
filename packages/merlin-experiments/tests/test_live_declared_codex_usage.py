"""Live provider counters belong only to explicitly selected run/thread identities."""
import json
from pathlib import Path

from merlin_experiments.phase1.telemetry.evidence import declared_home_usage
from merlin_experiments.tracking.phase1_views import cost_section


def _line(value):
    return json.dumps(value) + "\n"


def _usage(total=100, output=20, reasoning=8, cached=70, stamp="2026-10-12T00:00:10Z"):
    return {"type": "event_msg", "timestamp": stamp, "payload": {"type": "token_count", "info": {
        "total_token_usage": {"input_tokens": total, "cached_input_tokens": cached,
            "output_tokens": output, "reasoning_output_tokens": reasoning, "total_tokens": total + output},
        "last_token_usage": {"input_tokens": 999999, "output_tokens": 999999},
    }}}


def _run(tmp_path):
    run = tmp_path / "run-one"
    rounds = run / "rounds"
    rounds.mkdir(parents=True)
    workspace = tmp_path / "workspace"
    home = tmp_path / "declared-homes"
    (run / "environment.yaml").write_text(json.dumps({"run_id": run.name, "workspace_path": str(workspace),
        "codex_runtime_policy": {"home_root": str(home)}}))
    (rounds / "round_00.codex_events.raw.jsonl").write_text(_line({"type": "thread.started", "thread_id": "a"}))
    session = home / "run-one_r00/sessions/log.jsonl"
    session.parent.mkdir(parents=True)
    meta = {"type": "session_meta", "timestamp": "2026-10-12T00:00:00Z",
        "payload": {"id": "a", "cwd": str(workspace)}}
    session.write_text(_line(meta) + _line(_usage()) + _line(_usage()) + _line(
        _usage(total=150, cached=100, output=30, reasoning=10, stamp="2026-10-12T00:00:20Z")))
    return run, session, meta


def test_latest_cumulative_not_sum_of_snapshots_or_last_usage(tmp_path):
    run, session, _ = _run(tmp_path)
    snapshot = run / "rounds/round_00.codex_rollout_snapshot/log.jsonl"
    snapshot.parent.mkdir()
    snapshot.write_bytes(session.read_bytes())
    with session.open("a") as stream:
        stream.write('{"type":')  # live partial event is not a schema or zero-usage event
    result = declared_home_usage(run)
    assert result["available"] and len(result["threads"]) == 1
    assert result["tokens"]["tokens_total"] == 180
    assert result["tokens"]["tokens_fresh_input"] == 50
    assert result["tokens"]["tokens_output"] == 30
    assert result["tokens"]["tokens_reasoning"] == 10
    assert result["threads"][0]["input_tokens_per_client_observed_s"] == 7.5
    assert result["threads"][0]["output_tokens_per_client_observed_s"] == 1.5
    assert result["cost_usd"] is None and result["usage_complete"] is False
    html = cost_section({"cost": {"live_usage": result}}, None)
    assert "Observed agent threads" in html and "not server decode" in html and "USD charge" in html


def test_thread_workspace_and_declared_home_are_required(tmp_path):
    run, session, meta = _run(tmp_path)
    for changed in ({**meta["payload"], "id": "unrelated"}, {**meta["payload"], "cwd": "/another/workspace"}):
        session.write_text(_line({**meta, "payload": changed}) + _line(_usage()))
        assert declared_home_usage(run)["available"] is False
    (run / "environment.yaml").write_text(json.dumps({"workspace_path": meta["payload"]["cwd"]}))
    session.write_text(_line(meta) + _line(_usage()))
    assert declared_home_usage(run)["available"] is False


def test_invalid_bucket_relations_cannot_mint_live_usage(tmp_path):
    run, session, meta = _run(tmp_path)
    for usage in (_usage(cached=101), _usage(reasoning=21), _usage(total=True)):
        session.write_text(_line(meta) + _line(usage))
        assert declared_home_usage(run)["available"] is False
