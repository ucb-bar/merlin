"""Console agent, token and event sources read the agents' own streams faithfully and read-only."""

import json

from merlin_experiments.tracking.console.model import Collector

SOURCES = (("agents", "fast"), ("tokens", "fast"), ("events", "fast"))
LEAD, HELPER, OTHER = "01a-lead", "01a-helper-one", "01a-helper-two"


def _usage(inp, cached, out, reasoning=0):
    return {
        "input_tokens": inp,
        "cached_input_tokens": cached,
        "cache_write_input_tokens": 0,
        "output_tokens": out,
        "reasoning_output_tokens": reasoning,
        "total_tokens": inp + out,
    }


def _meta(thread, ordinal=0, parent=None, nickname=None, path=None, start=None):
    payload = {"id": thread, "timestamp": "2026-10-22T18:00:00Z"}
    if parent:
        payload.update(
            parent_thread_id=parent,
            agent_nickname=nickname,
            agent_path=path,
            subagent_history_start_ordinal=start,
            source={
                "subagent": {
                    "thread_spawn": {
                        "parent_thread_id": parent,
                        "depth": 1,
                        "agent_path": path,
                        "agent_nickname": nickname,
                    }
                }
            },
        )
    return {"timestamp": "2026-10-22T18:00:00Z", "ordinal": ordinal, "type": "session_meta", "payload": payload}


def _event(thread, ordinal, stamp, payload):
    return {"timestamp": stamp, "ordinal": ordinal, "type": "event_msg", "payload": {"thread_id": thread, **payload}}


def _command(thread, ordinal, stamp, command, code, start_ms, end_ms):
    item = {
        "type": "CommandExecution",
        "command": ["/bin/bash", "-lc", command],
        "status": "failed" if code else "completed",
        "exit_code": code,
        "duration": {"secs": 1, "nanos": 0},
    }
    return _event(
        thread,
        ordinal,
        stamp,
        {"type": "item_completed", "item": item, "started_at_ms": start_ms, "completed_at_ms": end_ms},
    )


def _tokens(thread, ordinal, stamp, usage):
    return {
        "timestamp": stamp,
        "ordinal": ordinal,
        "type": "token_usage_record",
        "payload": {"thread_id": thread, "usage": usage, "thread_token_usage": usage},
    }


def _write(path, rows, tail=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows) + tail)


def _run(tmp_path, *, budget=True):
    run = tmp_path / "out" / "runs" / "t" / "phase1" / "run-a"
    sessions = run / "rounds" / "round_00.codex_rollout_snapshot" / "2026" / "10" / "22"
    base = 1792692000
    lead = [
        _meta(LEAD),
        _event(LEAD, 1, "2026-10-22T18:00:01Z", {"type": "task_started", "started_at": base}),
        _command(LEAD, 2, "2026-10-22T18:00:05Z", "make check", 2, (base + 2) * 1000, (base + 5) * 1000),
        _tokens(LEAD, 3, "2026-10-22T18:00:06Z", _usage(1000, 900, 50, 10)),
        _event(
            LEAD,
            4,
            "2026-10-22T18:00:07Z",
            {
                "type": "item_completed",
                "item": {"type": "AgentMessage", "content": [{"type": "Text", "text": "done\x1b[2J here"}]},
            },
        ),
        _tokens(LEAD, 5, "2026-10-22T18:00:20Z", _usage(3000, 2500, 150, 30)),
        {
            "timestamp": "2026-10-22T18:00:21Z",
            "ordinal": 6,
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {"total_token_usage": _usage(3000, 2500, 150, 30), "model_context_window": 475000},
                "rate_limits": {"primary": {"used_percent": 21.0, "window_minutes": 10080, "resets_at": base + 9}},
            },
        },
    ]
    # The helper's stream begins with the forked parent history (ordinals before 5): never its own work.
    helper = [
        _meta(HELPER, 0, LEAD, "Curie", "/root/frontend", start=5),
        _command(HELPER, 2, "2026-10-22T18:00:02Z", "parent history", 1, (base + 1) * 1000, (base + 2) * 1000),
        _event(HELPER, 6, "2026-10-22T18:00:08Z", {"type": "task_started", "started_at": base + 8}),
        _command(HELPER, 7, "2026-10-22T18:00:10Z", "pytest", 0, (base + 9) * 1000, (base + 10) * 1000),
        _tokens(HELPER, 8, "2026-10-22T18:00:11Z", _usage(500, 400, 20)),
        _event(
            HELPER,
            9,
            "2026-10-22T18:00:12Z",
            {"type": "task_complete", "started_at": base + 8, "completed_at": base + 12},
        ),
    ]
    other = [_meta(OTHER, 0, LEAD, "Kuhn", "/root/device")]
    _write(sessions / "rollout-a-lead.jsonl", lead, tail='{"timestamp": "2026-10-22T18:00:30Z", "ordinal": 7, "ty')
    _write(sessions / "rollout-b-helper.jsonl", helper)
    _write(sessions / "rollout-c-helper.jsonl", other)
    (run / "rounds" / "round_00.codex_summary.json").write_text(
        json.dumps({"subagent_capacity": {"max_concurrent_threads_per_session": 16}, "exit_code": 0, "wall_s": 3600})
    )
    config = (
        {
            "author_token_budget": {
                "schema": "merlin.author_token_budget.v1",
                "target": 40000,
                "reserve": 8000,
                "mode": "soft",
            }
        }
        if budget
        else {}
    )
    import yaml

    (run / "environment.yaml").write_text(yaml.safe_dump({"run_config": config}))
    workspace = tmp_path / "workspace"
    (workspace / ".qa_channel").mkdir(parents=True)
    (workspace / ".qa_channel" / "resp_1.json").write_text(
        json.dumps({"all_pass": False, "n_passed": 3, "n_capsules": 5, "graded_at_unix_ns": (base + 15) * 10**9})
    )
    experiment = {
        "id": str(run),
        "run_id": run.name,
        "run_dir": str(run),
        "workspace": str(workspace),
        "state": "LIVE",
        "launch_dir": None,
    }
    return run, experiment, sessions


def test_tree_tokens_and_events_from_rollouts(tmp_path):
    run, experiment, sessions = _run(tmp_path)
    collector = Collector(experiment, operator_private=True, sources=SOURCES, clock=lambda: 1792692100.0)
    snapshot = collector.refresh()
    agents = snapshot["sections"]["agents"]
    assert "error" not in agents, agents
    by_id = {a["id"]: a for a in agents["agents"]}
    assert set(by_id) == {LEAD, HELPER, OTHER}
    assert by_id[LEAD]["role"] == "lead" and by_id[LEAD]["parent_id"] is None
    assert by_id[HELPER]["parent_id"] == LEAD and by_id[HELPER]["name"] == "Curie"
    assert by_id[HELPER]["path"] == "/root/frontend"
    # Forked parent history is skipped: one command of its own, not two.
    assert by_id[HELPER]["commands"] == 1 and by_id[HELPER]["failed_commands"] == 0
    assert by_id[HELPER]["state"] == "completed" and by_id[HELPER]["active_s"] == 4
    assert by_id[LEAD]["state"] == "working" and by_id[LEAD]["failed_commands"] == 1
    assert "turn" in [s[2] for s in by_id[HELPER]["spans"]] and any(s[2] == "tool" for s in by_id[LEAD]["spans"])
    assert "\x1b" not in by_id[LEAD]["last_message"] and "done" in by_id[LEAD]["last_message"]
    lead_tokens = by_id[LEAD]["tokens"]
    assert lead_tokens == {
        "input": 3000,
        "cache_read": 2500,
        "cache_write": 0,
        "fresh_input": 500,
        "output": 150,
        "reasoning": 30,
        "total": 3150,
    }
    # An agent that recorded no usage has no tokens, not zero tokens.
    assert by_id[OTHER]["tokens"]["total"] is None
    assert agents["totals"] == {"agents": 3, "working": 1, "helpers_working": 0,
                                "subagents": 2, "helper_ceiling": 16}
    assert agents["usage_completeness"]["threads_with_usage"] == 2
    assert not agents["usage_completeness"]["all_observed_threads_have_usage"]
    assert agents["usage_completeness"]["totals_are_lower_bounds"]

    tokens = snapshot["sections"]["tokens"]
    assert tokens["total"]["input"] == 3500 and tokens["total"]["output"] == 170
    series = tokens["series"]
    assert series["output"] == sorted(series["output"]) and series["output"][-1] == 170
    assert series["cache_read"][-1] == 2900
    assert tokens["budget"]["total"] == 40000 and tokens["budget"]["working"] == 32000
    assert abs(tokens["budget"]["used_pct"] - 100 * 3670 / 40000) < 1e-9
    assert tokens["budget"]["enforced"] is False  # a v1 target has no managed admission
    assert tokens["rate_limit"]["used_percent"] == 21.0 and tokens["context_window"] == 475000
    assert abs(tokens["cache_hit_ratio"] - 2900 / 3500) < 1e-9

    events = snapshot["sections"]["events"]["events"]
    kinds = [e["kind"] for e in events]
    assert {"check", "agent", "stop"} <= set(kinds)
    # Messages appear in operator-private mode; failed commands are grouped per agent and minute.
    assert any(e["kind"] == "agent" and "failed" in e["text"] for e in events)
    timed = [e["t"] for e in events if e["t"] is not None]
    assert timed == sorted(timed, reverse=True)
    assert any(e["text"] == "self-check fail 3/5" for e in events)
    assert all("\x1b" not in e["text"] for e in events)

    detail = collector.inspect("agents", LEAD)
    assert detail["paths"] and any(s["title"] == "Recent commands" for s in detail["sections"])


def test_truncated_line_waits_and_public_mode_hides_text(tmp_path):
    run, experiment, sessions = _run(tmp_path)
    collector = Collector(experiment, operator_private=False, sources=SOURCES, clock=lambda: 1792692100.0)
    first = collector.refresh()["sections"]
    lead = next(a for a in first["agents"]["agents"] if a["id"] == LEAD)
    assert lead["current"] is None and lead["last_message"] is None
    assert lead["tokens"]["total"] == 3150
    # Public mode drops agent messages and command text: agent rows are only lifecycle and failure counts.
    agent_rows = [e for e in first["events"]["events"] if e["kind"] == "agent"]
    assert all(
        e["text"].startswith(("helper", "command failed (exit", "commands failed")) or " commands failed (" in e["text"]
        for e in agent_rows
    )
    # Completing the partial line makes it visible on the next refresh, not before.
    with open(sessions / "rollout-a-lead.jsonl", "a") as stream:
        stream.write(
            f'pe": "token_usage_record", "payload": {{"thread_id": "{LEAD}", '
            f'"thread_token_usage": {json.dumps(_usage(4000, 3000, 200))}}}}}\n'
        )
    second = collector.refresh()["sections"]
    lead = next(a for a in second["agents"]["agents"] if a["id"] == LEAD)
    assert lead["tokens"]["total"] == 4200


def test_missing_records_are_not_zero(tmp_path):
    run = tmp_path / "empty-run"
    (run / "rounds").mkdir(parents=True)
    experiment = {"id": str(run), "run_id": run.name, "run_dir": str(run), "workspace": None, "state": "ENDED"}
    sections = Collector(experiment, operator_private=True, sources=SOURCES).refresh()["sections"]
    assert sections["agents"]["agents"] == [] and sections["agents"]["notes"]
    assert all(v is None for v in sections["tokens"]["total"].values())
    assert sections["tokens"]["budget"] is None and sections["tokens"]["series"]["t"] == []
    assert sections["events"]["events"] == []


def test_stopped_run_does_not_report_a_working_agent(tmp_path):
    run, experiment, _ = _run(tmp_path, budget=False)
    experiment["state"] = "INTERRUPTED"
    agents = Collector(experiment, operator_private=True, sources=SOURCES, clock=lambda: 1792699999.0).refresh()[
        "sections"
    ]["agents"]["agents"]
    lead = next(a for a in agents if a["id"] == LEAD)
    assert lead["state"] == "stopped"
    # The open turn ends at the last recorded activity, not at the viewer's clock.
    assert lead["active_s"] < 60


def test_outer_tool_spans_match_call_ids_without_using_background_duration():
    from merlin_experiments.tracking.console.sources import agents
    index = agents.RolloutIndex()
    index._row("a", _meta(LEAD))
    for stamp, kind, call in [("18:00:01", "custom_tool_call", "a"),
                              ("18:00:02", "function_call", "b"),
                              ("18:00:04", "custom_tool_call_output", "a"),
                              ("18:00:05", "function_call_output", "b")]:
        index._row("a", {"timestamp": f"2026-10-22T{stamp}Z", "type": "response_item",
                         "payload": {"type": kind, "call_id": call}})
    spans = agents.intervals(index.threads[LEAD], 0, False)
    assert sum(b-a for a, b in spans["tool_call"]) == 4
    assert spans["tool"] == []
    assert agents.tokens_of({"input_tokens": 100})["fresh_input"] is None
