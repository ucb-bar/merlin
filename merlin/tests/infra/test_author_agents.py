"""Per-thread author accounting: the lead and every Codex sub-agent, from their own rollouts.

Two kinds of input. A sanitized REAL capture (codex-cli 0.162.0, one lead spawning two helpers that
each ran a command and added a file) pins the event schema the accounting relies on, including the
two facts that make the rollouts necessary: the exec stream carries the lead only, and its usage
excludes the helpers. Synthetic streams then exercise interleaved helpers, request attribution,
simulation joins and the live/cost outputs without any provider call.
"""

from __future__ import annotations

import json
import os
import shlex
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml
from merlin_experiments.phase1 import convergence_report as CR
from merlin_experiments.phase1.telemetry import agent_commands as AC
from merlin_experiments.phase1.telemetry import author_agents as AA

from merlin.common.paths import merlin_dir

CAPTURE = merlin_dir() / "tests" / "fixtures" / "codex_multi_agent.json"
LEAD = "01a14a7f-9a32-7d50-9a5f-e1c96b57a926"
ONE = "01a14a7f-b0d1-7121-b977-83225f4fba52"
TWO = "01a14a7f-bb58-7853-80bb-49b375f616ec"
T0 = datetime(2026, 10, 17, 12, 0, 0, tzinfo=UTC).timestamp()


def _captured(tmp_path: Path) -> tuple[Path, Path, list[dict]]:
    """Materialize the capture as the files Codex wrote: the exec stream and one rollout per thread."""
    capture = json.loads(CAPTURE.read_text())
    run, home = tmp_path / "run", tmp_path / "codex-home"
    (run / "rounds").mkdir(parents=True)
    stream = capture["exec_stream"]
    (run / "rounds" / "round_00.codex_events.raw.jsonl").write_text("".join(json.dumps(e) + "\n" for e in stream))
    for relative, rows in capture["rollouts"].items():
        path = home / "sessions" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return run, home, stream


def _rollout_file(home: Path, thread: str) -> Path:
    return next((home / "sessions").rglob(f"*{thread}.jsonl"))


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


# --- the measured schema ------------------------------------------------------------------------


def test_the_exec_stream_reports_the_lead_only_and_omits_helper_usage(tmp_path):
    _run, home, stream = _captured(tmp_path)
    assert json.loads(CAPTURE.read_text())["provenance"]["cli_version"] == "codex-cli 0.162.0"
    assert [e["thread_id"] for e in stream if e["type"] == "thread.started"] == [LEAD]
    collab = [e["item"] for e in stream if e.get("item", {}).get("type") == "collab_tool_call"]
    # Only the waits surface, and they name no receiver: no spawn event, no helper id.
    assert collab and {i["tool"] for i in collab} == {"wait"} and all(not i["receiver_thread_ids"] for i in collab)
    commands = [e["item"] for e in stream if e["type"] == "item.completed" and e["item"]["type"] == "command_execution"]
    assert [c["aggregated_output"].strip() for c in commands] == [LEAD]
    assert not [e for e in stream if e.get("item", {}).get("type") == "file_change"]
    (usage,) = [e["usage"] for e in stream if e["type"] == "turn.completed"]
    lead_rollout = AA.read_rollout(_rollout_file(home, LEAD))
    lead_input = sum(
        r["payload"]["usage"]["input_tokens"] for r in lead_rollout["rows"] if r["type"] == "token_usage_record"
    )
    assert usage["input_tokens"] == lead_input == 90231


def test_a_forked_helper_rollout_excludes_its_copied_parent_history(tmp_path):
    _run, home, _stream = _captured(tmp_path)
    helper = AA.read_rollout(_rollout_file(home, ONE))
    raw = _rows(_rollout_file(home, ONE))
    metas = [r["payload"] for r in raw if r["type"] == "session_meta"]
    assert [m["id"] for m in metas] == [ONE, LEAD]  # own meta first, then the copied parent's
    assert helper["parent_thread_id"] == LEAD and helper["agent_path"] == "/root/one" and helper["depth"] == 1
    start = metas[0]["subagent_history_start_ordinal"]
    assert all(r["ordinal"] >= start for r in helper["rows"])
    assert len(helper["rows"]) < len(raw) - 1
    commands = [r for r in helper["rows"] if r.get("payload", {}).get("item", {}).get("type") == "CommandExecution"]
    # Every thread's commands see their OWN id in CODEX_THREAD_ID.
    assert [c["payload"]["item"]["aggregated_output"].strip() for c in commands] == [ONE]


def test_the_captured_run_is_accounted_per_thread(tmp_path):
    run, home, _stream = _captured(tmp_path)
    report = AA.account(run, codex_homes=[home], now=T0 * 2)
    assert report["available"] and report["complete"] and report["missing_subagents"] == []
    rows = {row["thread_id"]: row for row in report["agents"]}
    assert set(rows) == {LEAD, ONE, TWO}
    assert rows[LEAD]["role"] == "lead" and rows[LEAD]["spawned"] == [ONE, TWO]
    assert {rows[ONE]["agent_path"], rows[TWO]["agent_path"]} == {"/root/one", "/root/two"}
    assert rows[ONE]["parent_thread_id"] == rows[TWO]["parent_thread_id"] == LEAD
    assert rows[ONE]["tokens"]["input_total"] == 44124 and rows[TWO]["tokens"]["input_total"] == 44058
    assert rows[LEAD]["tokens"]["input_total"] == 90231
    totals = report["totals"]
    assert totals["tokens_subagents"]["input_total"] == 44124 + 44058
    assert totals["tokens_all"]["total"] == sum(r["tokens"]["total"] for r in rows.values())
    assert [rows[t]["commands"]["n"] for t in (LEAD, ONE, TWO)] == [1, 1, 1]
    assert [rows[t]["edits"]["items"] for t in (LEAD, ONE, TWO)] == [0, 1, 1]
    assert rows[ONE]["edits"]["first_paths"] == ["/fixture/workspace/one.txt"]
    assert all(row["state"] == "finished" for row in rows.values())
    assert report["concurrency"]["max_active"] == 3
    assert rows[LEAD]["wait_s"] > 0 and rows[LEAD]["active_s"] < rows[LEAD]["turn_wall_s"]
    shares = sum(row["share_of_agent_activity"] for row in rows.values())
    assert shares == pytest.approx(1.0, abs=1e-3)


# --- command parsing ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("python3 agent_selfcheck.py --submission submission --capsules A,B", [("check", "A,B", "submission")]),
        ("timeout 600 env X=1 python3 -u ./agent_selfcheck.py --capsules=A > log 2>&1", [("check", "A", "submission")]),
        ("python3 agent_selfcheck.py --submission variants/fast --capsules A", [("check", "A", "variants/fast")]),
        ("python3 agent_selfcheck.py --attach 1_2_3 > resume.log", [("attach", "all", "submission")]),
        ("python3 agent_selfcheck.py --help | tail -5", [("help", "all", "submission")]),
        ("python3 agent_selfcheck.py --compile-only --capsules A", [("compile-only", "A", "submission")]),
        ("rg --files -g 'agent_selfcheck.py'; sed -n '1,9p' agent_selfcheck.py", []),
        ("python3 - <<'PY'\nimport subprocess\nprint('agent_selfcheck.py --capsules X')\nPY\nls", []),
        (
            "cat > t.py <<'PY'\npython3 agent_selfcheck.py --capsules X\nPY\npython3 agent_selfcheck.py --capsules Y",
            [("check", "Y", "submission")],
        ),
        (
            "cd sub && PYTHONPATH=. python3 ../agent_selfcheck.py --capsules A; echo done",
            [("check", "A", "submission")],
        ),
    ],
)
def test_only_real_client_invocations_count(command, expected):
    for form in (["/bin/bash", "-lc", command], f"/bin/bash -lc {shlex.quote(command)}"):
        calls = [c for c in AC.client_calls(form) if c["client"] == AC.SELFCHECK]
        assert [(c["mode"], c["capsules"], c["submission"]) for c in calls] == expected
    variant = AC.client_calls(["/bin/bash", "-lc", "python3 agent_selfcheck.py --submission variants/fast/"])
    assert variant[0]["variant"] == "fast"
    (job,) = AC.client_calls(["/bin/bash", "-lc", "python3 simjob.py submit --sim gsim --capsules A"])
    assert (job["client"], job["action"], job["submits_request"], job["sim"]) == ("simjob.py", "submit", True, "gsim")


# --- synthetic interleaved helpers --------------------------------------------------------------


def _iso(offset: float) -> str:
    return datetime.fromtimestamp(T0 + offset, UTC).isoformat().replace("+00:00", "Z")


class _Rollout:
    def __init__(self, tid: str, parent: str | None = None, path: str = "/root", start: int | None = None):
        self.tid, self.rows, self.ordinal = tid, [], 0
        source = (
            "exec"
            if parent is None
            else {
                "subagent": {
                    "thread_spawn": {
                        "parent_thread_id": parent,
                        "depth": path.count("/") - 1,
                        "agent_path": path,
                        "agent_nickname": path[6:],
                        "agent_role": None,
                    }
                }
            }
        )
        meta = {"id": tid, "parent_thread_id": parent, "cwd": "WS", "source": source, "timestamp": _iso(0)}
        if start is not None:
            meta["subagent_history_start_ordinal"] = start
        self.rows.append({"timestamp": _iso(0), "ordinal": 0, "type": "session_meta", "payload": meta})

    def add(self, at: float, kind: str, payload: dict, ordinal: int | None = None) -> _Rollout:
        self.ordinal = self.ordinal + 1 if ordinal is None else ordinal
        self.rows.append({"timestamp": _iso(at), "ordinal": self.ordinal, "type": kind, "payload": payload})
        return self

    def turn(self, begin: float, end: float | None, turn: str) -> _Rollout:
        self.add(begin, "event_msg", {"type": "task_started", "turn_id": turn})
        if end is not None:
            self.add(end, "event_msg", {"type": "task_complete", "turn_id": turn})
        return self

    def item(self, begin: float, end: float, item: dict, thread: str | None = None) -> _Rollout:
        payload = {
            "type": "item_completed",
            "thread_id": thread or self.tid,
            "item": item,
            "started_at_ms": int((T0 + begin) * 1000),
            "completed_at_ms": int((T0 + end) * 1000),
        }
        return self.add(end, "event_msg", payload)

    def command(self, begin: float, end: float, script: str, status: str = "completed") -> _Rollout:
        item = {
            "type": "CommandExecution",
            "command": ["/bin/bash", "-lc", script],
            "status": status,
            "exit_code": 0 if status == "completed" else 1,
            "aggregated_output": "",
        }
        return self.item(begin, end, item)

    def usage(self, at: float, response: str, tokens: int, thread: str | None = None) -> _Rollout:
        payload = {
            "thread_id": thread or self.tid,
            "response_id": response,
            "usage": {
                "input_tokens": tokens,
                "cached_input_tokens": tokens // 2,
                "cache_write_input_tokens": 0,
                "output_tokens": tokens // 10,
                "reasoning_output_tokens": tokens // 20,
            },
        }
        return self.add(at, "token_usage_record", payload)

    def write(self, sessions: Path, workspace: Path) -> None:
        path = sessions / "2026" / "10" / "17" / f"rollout-{self.tid}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        text = "".join(json.dumps(r).replace('"WS"', json.dumps(str(workspace))) + "\n" for r in self.rows)
        path.write_text(text)


def _spawn(lead: _Rollout, at: float, child: str, path: str, kind: str = "started") -> None:
    lead.item(at, at, {"type": "SubAgentActivity", "kind": kind, "agent_thread_id": child, "agent_path": path})


def _write_json(path: Path, value, mtime: float | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    if mtime is not None:
        os.utime(path, (T0 + mtime, T0 + mtime))


def _request(
    channel: Path,
    rid: str,
    at: float,
    capsules: str,
    *,
    start: float,
    graded: float,
    passed=True,
    caller: str | None = None,
    engines=None,
) -> None:
    body = {"protocol": 3, "request_id": rid, "requested_at_unix_ns": int((T0 + at) * 1e9), "capsules": capsules}
    if caller:
        body["caller_thread_id"] = caller
    _write_json(channel / f"req_{rid}.json", body)
    names = capsules.split(",")
    tiers = {"L2": "pass", "L3": "pass" if passed else "fail"}
    _write_json(
        channel / f"resp_{rid}.json",
        {
            "selfcheck_protocol": 3,
            "selfcheck_request_id": rid,
            "submission_sha256": "f" * 64,
            "all_pass": passed,
            "n_passed": len(names) if passed else 0,
            "tier_engines": engines or {"L2": "spike", "L3": "gsim"},
            "per_capsule": [{"capsule": n, "pass": passed, "tiers": tiers} for n in names],
            "grade_started_at_unix_ns": int((T0 + start) * 1e9),
            "graded_at_unix_ns": int((T0 + graded) * 1e9),
        },
    )


def _raw(run: Path, name: str, capsules: list[str], at: float, *, carried_l3: bool = False) -> None:
    for capsule in capsules:
        l3 = {"status": "pass", "engine": "gsim", "measured_now": not carried_l3}
        if carried_l3:
            l3["carried"] = {"carried": True}
        tiers = {
            "L1": {"status": "pass", "measured_now": True},
            "L2": {"status": "pass", "engine": "spike", "measured_now": True},
            "L3": l3,
        }
        _write_json(
            run / "_qa_work" / "selfchecks" / name / "runs" / "suite" / capsule / "capsule_result.json",
            {"capsule": capsule, "tiers": tiers},
            mtime=at,
        )


def _synthetic_run(tmp_path: Path, *, drop_helper: bool = False) -> tuple[Path, Path, dict]:
    run, workspace, homes = tmp_path / "out" / "run7", tmp_path / "agent" / "workspace", tmp_path / "homes"
    for path in (run / "rounds", workspace / ".qa_channel"):
        path.mkdir(parents=True)
    ids = {"lead": "lead-0000", "a": "help-aaaa", "b": "help-bbbb", "c": "help-cccc"}
    (run / "environment.yaml").write_text(
        yaml.safe_dump(
            {
                "run_id": run.name,
                "workspace_path": str(workspace),
                "codex_runtime_policy": {"home_root": str(homes)},
            }
        )
    )
    stream = [{"type": "thread.started", "thread_id": ids["lead"]}, {"type": "turn.started"}]
    (run / "rounds" / "round_00.codex_events.raw.jsonl").write_text("".join(json.dumps(e) + "\n" for e in stream))

    lead = _Rollout(ids["lead"]).turn(0, None, "t-lead")
    lead.usage(5, "r-l1", 10000).command(10, 20, "ls")
    for child, at, path in ((ids["a"], 30, "/root/a"), (ids["b"], 31, "/root/b"), (ids["c"], 32, "/root/a/c")):
        _spawn(lead, at, child, path)
    # The lead and helper A submit the SAME capsule list concurrently; B submits another list via a
    # shell variable the parser cannot read literally; the lead also runs a simjob.
    lead.command(100, 400, "python3 agent_selfcheck.py --submission submission --capsules X,Y > lead.log")
    lead.command(500, 503, "python3 simjob.py submit --sim gsim --capsules Z")
    lead.item(600, 900, {"type": "CollabAgentToolCall", "tool": "wait", "receiver_thread_ids": []})
    _spawn(lead, 950, ids["a"], "/root/a", kind="completed")
    lead.usage(1000, "r-l2", 20000)

    a = _Rollout(ids["a"], ids["lead"], "/root/a", start=3)
    # Copied parent history below the start ordinal must not be counted for A.
    a.add(0, "event_msg", {"type": "task_started", "turn_id": "t-lead"}, ordinal=1)
    a.add(
        1,
        "token_usage_record",
        {"thread_id": ids["lead"], "response_id": "r-l1", "usage": {"input_tokens": 10000, "output_tokens": 1000}},
        ordinal=2,
    )
    a.ordinal = 2
    a.turn(30, 940, "t-a").usage(40, "r-a1", 4000)
    a.command(100.5, 380, "timeout 900 python3 agent_selfcheck.py --capsules X,Y --submission variants/fast")
    a.item(381, 382, {"type": "FileChange", "changes": {str(workspace / "submission/lower.py"): {"type": "update"}}})
    a.usage(939, "r-a2", 4000)

    b = _Rollout(ids["b"], ids["lead"], "/root/b").turn(31, None, "t-b").usage(50, "r-b1", 3000)
    b.command(200, 260, 'CAPS=W; python3 agent_selfcheck.py --capsules "$CAPS"')
    b.add(990, "event_msg", {"type": "token_count", "info": {"total_token_usage": {"input_tokens": 1}}})

    c = _Rollout(ids["c"], ids["a"], "/root/a/c").turn(32, 70, "t-c").usage(60, "r-c1", 1000)

    channel = workspace / ".qa_channel"
    _request(channel, "lead_req", 100.2, "X,Y", start=101, graded=200)
    _request(channel, "a_req", 100.7, "X,Y", start=101.5, graded=210)
    _request(channel, "b_req", 200.1, "W", start=201, graded=250, passed=False)
    _request(channel, "orphan", 2000, "Q", start=2001, graded=2002)
    _raw(run, "selfcheck_lead", ["X", "Y"], 150)
    _raw(run, "selfcheck_a", ["X", "Y"], 160, carried_l3=True)
    _write_json(
        channel / "simreq_promo0_abc_X.json",
        {"sim": "gsim", "capsules": "X", "tiers": "L3", "promoted": True, "submitted_at": T0 + 205},
    )
    _write_json(
        channel / "simreq_promo1_abc_W.json",
        {"sim": "gsim", "capsules": "W", "tiers": "L3", "promoted": True, "submitted_at": T0 + 255},
    )
    _write_json(channel / "simreq_7_1.json", {"sim": "gsim", "capsules": "Z", "submitted_at": int(T0 + 501)})

    sessions = homes / f"{run.name}_r00" / "sessions"
    for rollout in (lead, a, b) if drop_helper else (lead, a, b, c):
        rollout.write(sessions, workspace)
    return run, workspace, ids


def test_interleaved_helpers_are_each_accounted_and_attributed(tmp_path):
    run, workspace, ids = _synthetic_run(tmp_path)
    report = AA.account(run, now=T0 + 1000)
    assert report["complete"] and report["totals"]["threads"] == 4 and report["totals"]["subagents"] == 3
    rows = {row["thread_id"]: row for row in report["agents"]}
    lead, a, b, c = (rows[ids[k]] for k in ("lead", "a", "b", "c"))
    assert c["parent_thread_id"] == ids["a"] and c["depth"] == 2 and c["round"] == 0
    # Usage: per-response records, deduplicated; the copied parent response is not A's.
    assert lead["tokens"]["input_total"] == 30000 and a["tokens"]["input_total"] == 8000
    assert b["tokens"]["source"] == "token_usage_record" and b["tokens"]["input_total"] == 3000
    assert report["totals"]["tokens_subagents"]["input_total"] == 12000
    # States: A finished per its own turn; B's turn never closed but is recent; C finished.
    assert (lead["state"], a["state"], b["state"], c["state"]) == ("active", "finished", "active", "finished")
    assert lead["wait_s"] == pytest.approx(300.0, abs=0.01)
    # Same capsule list at the same moment: each request goes to the command that started nearest it.
    by_id = {q["request_id"]: q for row in report["agents"] for q in row["selfcheck_requests"]}
    assert [q["request_id"] for q in lead["selfcheck_requests"]] == ["lead_req"]
    assert [q["request_id"] for q in a["selfcheck_requests"]] == ["a_req"]
    assert by_id["lead_req"]["basis"] == by_id["a_req"]["basis"] == "command_window"
    assert by_id["a_req"]["candidates"] == 2 and by_id["a_req"]["variant"] == "fast"
    assert by_id["b_req"]["basis"] == "command_window_unmatched_capsules"
    assert [q["request_id"] for q in report["unattributed_requests"]] == ["orphan"]
    # Simulations: raw results joined by capsule set and grade window; a carried L3 is not a new run.
    assert by_id["lead_req"]["simulations"]["by_engine"] == {"gsim": 2, "spike": 2}
    assert by_id["a_req"]["simulations"]["by_engine"] == {"spike": 2}
    assert by_id["a_req"]["simulations"]["carried"] == 2
    assert by_id["b_req"]["simulations"]["source"] == "response_tier_statuses"
    assert by_id["b_req"]["simulations"]["by_engine"] == {"gsim": 1, "spike": 1}
    # Promotion of X followed the passing lead verdict; W failed, so its promotion stays unattributed.
    assert lead["promotions"] == ["promo0_abc_X"] and report["unattributed_promotions"] == ["promo1_abc_W"]
    assert lead["simjobs"] == ["7_1"]
    assert lead["simulations"]["promotion_by_engine"] == {"gsim": 1}
    assert lead["simulations"]["simjob_by_engine"] == {"gsim": 1}
    assert report["totals"]["simulations_by_engine"] == {"gsim": 5, "spike": 5}
    assert a["edits"] == {"items": 1, "paths": 1, "first_paths": ["submission/lower.py"]}
    assert report["concurrency"]["max_active"] == 4


def test_a_forwarded_caller_thread_id_wins_over_the_command_window(tmp_path):
    run, workspace, ids = _synthetic_run(tmp_path)
    _request(workspace / ".qa_channel", "lead_req", 100.2, "X,Y", start=101, graded=200, caller=ids["b"])
    report = AA.account(run, now=T0 + 1000)
    owner = {q["request_id"]: row["thread_id"] for row in report["agents"] for q in row["selfcheck_requests"]}
    basis = {q["request_id"]: q["basis"] for row in report["agents"] for q in row["selfcheck_requests"]}
    assert owner["lead_req"] == ids["b"] and basis["lead_req"] == "caller_thread_id"


def test_a_helper_without_its_rollout_makes_the_record_incomplete(tmp_path):
    run, _workspace, ids = _synthetic_run(tmp_path, drop_helper=True)
    report = AA.account(run, now=T0 + 1000)
    assert report["complete"] is False and report["missing_subagents"] == [ids["c"]]
    assert AA.cost_fields(report)["subagent_tool_calls_tracked"] is False


def test_no_rollout_is_unavailable_not_zero(tmp_path):
    run = tmp_path / "run"
    (run / "rounds").mkdir(parents=True)
    report = AA.account(run)
    assert report["available"] is False and "rollout" in report["reason"]
    assert AA.heartbeat_line(report).startswith("[agents] available=false")
    assert AA.cost_fields(report)["subagent_tool_calls_tracked"] is False


def test_live_record_is_operator_side_and_heartbeat_is_parseable(tmp_path):
    run, workspace, _ids = _synthetic_run(tmp_path)
    for refused in (run / "agents.json", workspace / "agents.json"):
        with pytest.raises(ValueError, match="operator-side"):
            AA.write_live(run, refused, now=T0 + 1000)
    operator = tmp_path / "operator"
    line = AA.write_live(run, operator / "agents.json", now=T0 + 1000)
    document = json.loads((operator / "agents.json").read_text())
    assert document["schema"] == AA.SCHEMA and document["observed_at"]
    assert oct((operator / "agents.json").stat().st_mode & 0o777) == "0o600"
    fields = CR._fields(line[len("[agents] ") :])
    assert fields["active"] == "2" and fields["total"] == "4" and fields["subagents"] == "3"
    assert json.loads(fields["sims"]) == {"gsim": 5, "spike": 5} and fields["complete"] == "true"
    assert AA.main(["--run-dir", str(run), "--out", str(operator / "again.json")]) == 0


def test_the_cost_record_gains_subagent_fields_without_redefining_tool_calls(tmp_path):
    run, _workspace, _ids = _synthetic_run(tmp_path)
    assert AA.annotate_cost_record(run) is None  # no cost record yet: nothing is created
    (run / "cost_time_toolcalls.yaml").write_text(
        yaml.safe_dump({"tool_calls": 3, "subagent_tool_calls_tracked": False})
    )
    fields = AA.annotate_cost_record(run)
    document = yaml.safe_load((run / "cost_time_toolcalls.yaml").read_text())
    assert document["tool_calls"] == 3  # the lead stream's own count is unchanged
    assert document["subagent_tool_calls_tracked"] is True and fields["subagent_tool_calls_tracked"] is True
    assert document["subagent_tool_calls"] == 3 and document["tool_calls_all_agents"] == 6
    assert document["author_agents"]["subagents"] == 3
    assert len(document["author_agents"]["per_agent"]) == 4


def test_the_convergence_report_has_an_agents_section(tmp_path):
    run, _workspace, _ids = _synthetic_run(tmp_path)
    report = CR.build_report(run, now=T0 + 1000)
    assert report["agents"]["available"] and report["agents"]["totals"]["subagents"] == 3
    text = "\n".join(report["summary"])
    assert "agents: 4 threads (3 sub-agents)" in text
    assert "/root/a/c" in text
