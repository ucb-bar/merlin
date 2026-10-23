"""Managed provider admission preserves all-token accounting and active work."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase1.providers.token_admission import PREFIX, ManagedTokens
from merlin_experiments.phase1.telemetry import author_agents, token_budget


def selected(**changes):
    record = token_budget.selection(
        SimpleNamespace(
            author_token_admission=True,
            author_token_target=1000,
            author_token_reserve=300,
            author_token_turn_allowance=100,
        )
    )
    record["admission"].update(changes)
    return token_budget.validate(record)


def setup(run, policy=None):
    run.mkdir(exist_ok=True)
    (run / "environment.yaml").write_text(yaml.safe_dump({"run_config": {"author_token_budget": policy or selected()}}))


def usage(total, *, active=(), available=True):
    return dict(
        available=available,
        complete=available,
        tokens={"total": total},
        open_threads=list(active),
        missing_leads=[],
        missing_subagents=[],
    )


def test_active_helpers_consume_allowance_before_their_turns_complete(tmp_path):
    setup(tmp_path)
    guard = ManagedTokens(tmp_path, 0, observe=lambda: usage(450, active=("helper1", "helper2")))
    decision = guard.admit(0)
    assert decision["admitted"] and decision["phase"] == "integration"
    assert decision["projected_total"] == 750 and decision["observed_total"] == 450
    assert decision["outstanding_allowance"] == 200
    assert decision["hard_token_cap"] is False and decision["provider_upper_bound"] is None
    guard.finish()
    denied = ManagedTokens(tmp_path, 1, observe=lambda: usage(460)).admit(0)
    assert not denied["admitted"] and "already admitted" in denied["reason"]


def test_missing_usage_is_not_zero_and_outstanding_other_owner_is_not_freed(tmp_path):
    setup(tmp_path)
    unknown = ManagedTokens(tmp_path, 0, observe=lambda: usage(None, available=False)).admit(0)
    assert not unknown["admitted"] and unknown["observed_total"] is None
    one = ManagedTokens(tmp_path, 0, observe=lambda: usage(500))
    assert one.admit(0)["admitted"]
    two = ManagedTokens(tmp_path, 1, observe=lambda: usage(500))
    decision = two.admit(0)
    assert decision["outstanding_allowance"] == 100 and decision["projected_total"] == 700
    with pytest.raises(ValueError, match="outstanding reservation"):
        one.admit(0)


def test_live_checkpoint_transition_and_single_bounded_integration(tmp_path):
    setup(tmp_path, selected(poll_s=1, integration_slice_s=2))
    now = [0.0]
    observed = [usage(50)]
    guard = ManagedTokens(tmp_path, 0, clock=lambda: now[0], observe=lambda: observed[0])
    assert guard.admit(0)["phase"] == "exploration"
    observed[0] = usage(620, active=("lead",))
    now[0] = 1
    reason = guard.poll()
    assert reason == PREFIX + "development boundary"
    assert guard.can_resume_checkpoint(reason, thread_id="lead", continuous=True, remaining_s=100)
    guard.finish()
    observed[0] = usage(620)
    assert guard.admit(1)["phase"] == "integration"
    assert "Do not start new exploration or helpers" in guard.prompt()
    now[0] = 3
    reason = guard.poll()
    assert reason == PREFIX + "integration wall slice"
    assert not guard.can_resume_checkpoint(reason, thread_id="lead", continuous=True, remaining_s=100)
    guard.finish()
    assert not guard.admit(2)["admitted"]


def test_declining_total_retains_high_water_and_unknown_grace_is_bounded(tmp_path):
    setup(tmp_path, selected(poll_s=1, missing_usage_grace_s=2))
    now = [0.0]
    observed = [usage(200)]
    guard = ManagedTokens(tmp_path, 0, clock=lambda: now[0], observe=lambda: observed[0])
    guard.admit(0)
    guard.finish()
    observed[0] = usage(100)
    assert guard.admit(1)["observed_total"] == 200
    observed[0] = usage(None, available=False)
    now[0] = 1
    assert guard.poll() is None
    now[0] = 2
    assert guard.poll() is None
    now[0] = 3
    assert "unavailable after grace" in guard.poll()


def test_new_policy_is_explicit_and_resume_cannot_silently_enable_it():
    policy = selected()
    assert policy["schema"] == token_budget.ADMISSION_SCHEMA
    old = token_budget.selection(SimpleNamespace(joint_campaign="x"))
    with pytest.raises(ValueError, match="changed on resume"):
        token_budget.verify_resume(SimpleNamespace(author_token_admission=True), {"author_token_budget": old})
    token_budget.verify_resume(SimpleNamespace(), {"author_token_budget": old})
    from merlin_experiments.phase1.options import parse_options

    opts = parse_options(
        ["--run-id", "test", "--driver", "codex", "--schedule", "continuous", "--author-token-admission"]
    )
    assert token_budget.selection(opts)["admission"]["turn_allowance"] == 1_000_000
    assert "Read each catalog index once" in token_budget.guidance(opts)


def test_live_snapshot_uses_same_canonical_descendant_accounting(tmp_path):
    from test_author_agents import _captured

    run, home, _ = _captured(tmp_path)
    live = author_agents.usage_snapshot(run, codex_homes=(home,))
    report = author_agents.account(run, codex_homes=(home,))
    assert live["available"] and live["tokens"]["total"] == report["totals"]["tokens_all"]["total"]
    assert len(live["threads"]) == 3 and live["provider_upper_bound"] is None
    assert sum(row["role"] == "subagent" for row in live["threads"]) == 2


def test_real_provider_interrupts_inside_process_then_limits_resume(tmp_path, monkeypatch):
    from test_codex_driver import _run

    run = tmp_path / "run"
    setup(run, selected(poll_s=1, integration_slice_s=10))
    calls = tmp_path / "calls"
    script = tmp_path / "fake_codex"
    script.write_text(f"""#!{sys.executable}
import sys,json,time
from pathlib import Path
if '--version' in sys.argv: print('fake-budget-client');sys.exit(0)
p=Path({str(calls)!r});n=int(p.read_text())+1 if p.exists() else 1;p.write_text(str(n))
print(json.dumps({{'type':'thread.started','thread_id':'lead'}}),flush=True)
print(json.dumps({{'type':'turn.started'}}),flush=True)
if n==1: time.sleep(60)
print(json.dumps({{'type':'turn.completed','usage':{{'input_tokens':20,'cached_input_tokens':15,'output_tokens':2}}}}),flush=True)
""")
    script.chmod(0o755)

    def observe(*args, **kwargs):
        if not calls.exists():
            return {"available": False, "pristine": True}
        return usage(710 if calls.read_text() == "1" else 750)

    monkeypatch.setattr(author_agents, "usage_snapshot", observe)
    rc, _, records = _run(tmp_path, script, continue_session=True, timeout=600)
    assert rc == 75 and calls.read_text() == "2"
    interrupted = [r for r in records if r.get("type") == "codex_operator_interrupt"]
    assert interrupted and interrupted[0]["reason"] == PREFIX + "development boundary"
    assert interrupted[0]["turn_incomplete"] is True
    assert any(r.get("type") == "codex_token_checkpoint" for r in records)
    summary = next(r for r in records if r.get("type") == "codex_summary")
    assert summary["usage_complete"] is False and summary["token_admission"]["admitted"] is False
    # The ordinary continuation filename preserves the first prompt's stem.
    prompt = (run / "rounds" / "round_00.prompt.cont01.txt").read_text()
    assert "integration/checkpoint only" in prompt and "Keep delegating" not in prompt


def test_fresh_under_budget_work_is_not_interrupted_by_elapsed_wall_time(tmp_path):
    setup(tmp_path)
    now = [0.0]
    guard = ManagedTokens(tmp_path, 0, clock=lambda: now[0], observe=lambda: usage(20, active=("lead",)))
    assert guard.admit(0)["phase"] == "exploration"
    now[0] = 3600
    assert guard.poll() is None


def test_incremental_usage_matches_full_scan_without_rereading_history(tmp_path):
    import json

    from test_author_agents import _captured

    run, home, _ = _captured(tmp_path)
    reader = author_agents.UsageRows()
    expected = author_agents.usage_snapshot(run, codex_homes=(home,))
    actual = author_agents.usage_snapshot(run, codex_homes=(home,), reader=reader)
    assert actual == expected
    before = reader.bytes_read
    assert author_agents.usage_snapshot(run, codex_homes=(home,), reader=reader) == expected
    assert reader.bytes_read == before
    path = tmp_path / "partial.jsonl"
    line = json.dumps({"type": "token_usage_record", "payload": {"usage": {"input_tokens": 3}}}).encode()
    path.write_bytes(line[:20])
    assert reader(path) == []
    with path.open("ab") as stream:
        stream.write(line[20:] + b"\n")
    assert len(reader(path)) == 1
    assert len(reader(path)) == 1
    path.write_text('{"type":"thread.started","thread_id":"replacement"}\n')
    assert reader(path)[0]["thread_id"] == "replacement"


def test_resume_checks_allowance_even_without_repeated_enable_flag():
    policy = selected()
    with pytest.raises(ValueError, match="changed on resume"):
        token_budget.verify_resume(SimpleNamespace(author_token_turn_allowance=200), {"author_token_budget": policy})
    token_budget.verify_resume(SimpleNamespace(author_token_turn_allowance=100), {"author_token_budget": policy})
    assert policy["admission"]["turn_allowance"] == 100


def test_transient_missing_usage_after_long_healthy_turn_gets_full_continuous_grace(tmp_path):
    setup(tmp_path, selected(poll_s=1, missing_usage_grace_s=60))
    now = [0.0]
    observed = [usage(20, active=("lead",))]
    guard = ManagedTokens(tmp_path, 0, clock=lambda: now[0], observe=lambda: observed[0])
    guard.admit(0)
    now[0] = 180
    assert guard.poll() is None
    observed[0] = usage(None, available=False)
    now[0] = 185
    assert guard.poll() is None
    assert guard.last["observed_total"] is None
    assert guard.last["missing_usage_duration_s"] == 0
    now[0] = 244
    assert guard.poll() is None
    observed[0] = usage(30, active=("lead",))
    now[0] = 245
    assert guard.poll() is None
    assert guard.last["missing_usage_duration_s"] == 0
    observed[0] = usage(None, available=False)
    now[0] = 246
    assert guard.poll() is None
    now[0] = 305
    assert guard.poll() is None
    now[0] = 306
    assert guard.poll() == PREFIX + "live all-author usage unavailable after grace; checkpoint required"
    assert guard.last["missing_usage_duration_s"] == 60
    # Missing samples never become measured zeros or release outstanding work.
    import json

    state = json.loads(guard.path.read_text())
    assert state["observed_high_water"] == 30 and state["outstanding"]


def test_new_helper_before_first_usage_does_not_end_long_healthy_turn(tmp_path):
    from test_author_agents import _Rollout, _spawn

    run, home, ws = tmp_path / "run", tmp_path / "home", tmp_path / "workspace"
    setup(run, selected(poll_s=1, missing_usage_grace_s=60))
    lead = _Rollout("lead").turn(0, None, "lead-turn").usage(1, "first-response", 100)
    lead.write(home / "sessions", ws)
    now = [0.0]
    guard = ManagedTokens(run, 0, homes=(home,), clock=lambda: now[0])
    assert guard.admit(0)["admitted"]
    now[0] = 180
    assert guard.poll() is None and guard.last["observed_total"] == 110
    _spawn(lead, 181, "helper", "/root/helper")
    lead.write(home / "sessions", ws)
    child = _Rollout("helper", parent="lead", path="/root/helper").turn(181, None, "helper-turn")
    child.write(home / "sessions", ws)
    now[0] = 185
    assert guard.poll() is None
    snapshot = author_agents.usage_snapshot(run, codex_homes=(home,), reader=guard.reader)
    assert snapshot["complete"] and not snapshot["available"]
    assert snapshot["missing_leads"] == snapshot["missing_subagents"] == []
    assert guard.last["observed_total"] is None and guard.last["missing_usage_duration_s"] == 0
    child.usage(190, "helper-first-response", 20).write(home / "sessions", ws)
    now[0] = 190
    assert guard.poll() is None and guard.last["observed_total"] == 132
    assert guard.last["missing_usage_duration_s"] == 0
