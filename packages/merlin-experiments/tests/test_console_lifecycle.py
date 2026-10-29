"""Command console: the author lifecycle, per-core activity, orphaned agent turns and their findings."""

import json
import os
from datetime import UTC, datetime
from pathlib import Path

from merlin_experiments.tracking.console import diagnose
from merlin_experiments.tracking.console.io import Reader
from merlin_experiments.tracking.console.model import Context
from merlin_experiments.tracking.console.sources import activity, agents, events, lifecycle

T0 = 1792810000.0
#: The startup records' monotonic clock, offset from epoch seconds.
MONO = T0 - 400000.0


def _iso(t: float) -> str:
    return datetime.fromtimestamp(t, UTC).isoformat()


def _stage(stage, stage_id, turn, event, t, parent=None, wall=None, rnd=0):
    row = {
        "schema": "merlin.author_startup_stage.v1",
        "round": rnd,
        "turn": turn,
        "at": _iso(t),
        "monotonic_s": t - MONO,
        "stage": stage,
        "stage_id": stage_id,
        "parent_stage_id": parent,
        "event": event,
    }
    if wall is not None:
        row["wall_s"] = wall
    return row


def _write(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _run(tmp_path: Path, *, second_turn_open: bool = False) -> Path:
    """Turn 0 starts in 100 s and works 700 s; turn 1 starts for 750 s and is interrupted after 1 s."""
    run = tmp_path / "run"
    stages = [
        _stage("sandbox_assembly", "a0", 0, "start", T0),
        _stage("library_overlay", "a1", 0, "start", T0 + 10, parent="a0"),
        _stage("library_overlay", "a1", 0, "finish", T0 + 30, parent="a0", wall=20.0),
        _stage("sandbox_assembly", "a0", 0, "finish", T0 + 100, wall=100.0),
        _stage("sandbox_assembly", "b0", 1, "start", T0 + 810),
        _stage("library_overlay", "b1", 1, "start", T0 + 820, parent="b0"),
    ]
    if not second_turn_open:
        stages += [
            _stage("library_overlay", "b1", 1, "finish", T0 + 1260, parent="b0", wall=440.0),
            _stage("sandbox_assembly", "b0", 1, "finish", T0 + 1560, wall=750.0),
        ]
    _write(run / "rounds" / "author_startup.jsonl", stages)
    transcript = [
        {"type": "system", "subtype": "init", "started_at": _iso(T0 - 5)},
        {
            "type": "codex_token_admission",
            "turn": "0:0",
            "phase": "exploration",
            "admitted": True,
            "at_monotonic_s": T0 - MONO,
            "next_turn_allowance": 1000000,
        },
        {"type": "codex_operator_interrupt", "reason": "development boundary", "arrived_at": _iso(T0 + 800)},
        {
            "type": "codex_token_admission",
            "turn": "0:1",
            "phase": "integration",
            "admitted": True,
            "at_monotonic_s": T0 + 810 - MONO,
            "next_turn_allowance": 1000000,
        },
    ]
    if not second_turn_open:
        transcript += [
            {"type": "codex_operator_interrupt", "reason": "integration wall slice", "arrived_at": _iso(T0 + 1561)},
            {
                "type": "codex_session_turn",
                "continuing": False,
                "stopped_because": "host: integration wall slice",
                "arrived_at": _iso(T0 + 1565),
            },
            {
                "type": "codex_summary",
                "operator_interrupted": True,
                "interruption_reason": "integration wall slice",
                "exit_code": 130,
            },
        ]
    _write(run / "rounds" / "round_00.transcript.jsonl", transcript)
    return run


def _ctx(run: Path, shared=None, *, now=T0 + 1700, live=True) -> Context:
    experiment = {"id": str(run), "run_id": run.name, "run_dir": str(run), "last_record_at": now}
    return Context(
        experiment=experiment, reader=Reader(), state={}, operator_private=True, now=now, shared=shared or {}, live=live
    )


def test_turns_split_into_startup_and_agent_time(tmp_path):
    section = lifecycle.read(_ctx(_run(tmp_path)))
    first, second = section["turns"]
    assert (first["startup_s"], first["agent_s"], first["end_reason"]) == (100.0, 700.0, "development boundary")
    assert first["phase"] == "exploration" and abs(first["admitted_at"] - T0) < 1e-3
    assert (second["startup_s"], second["agent_s"]) == (750.0, 1.0)
    assert second["end_reason"] == "integration wall slice"
    assert [s["stage"] for s in second["stages"]] == ["sandbox_assembly", "library_overlay"]
    assert second["stages"][1]["depth"] == 1
    assert section["phase"] == "ended" and section["author_running"] is False
    assert section["phase_detail"] == "integration wall slice"


def test_open_startup_is_the_current_phase(tmp_path):
    section = lifecycle.read(_ctx(_run(tmp_path, second_turn_open=True), now=T0 + 1000))
    turn = section["turns"][-1]
    assert section["phase"] == "startup" and section["author_running"] is False
    assert turn["open_stage"]["stage"] == "library_overlay" and turn["startup_end"] is None
    assert turn["startup_s"] == 190.0 and turn["agent_start"] is None
    assert section["phase_detail"] == "library_overlay for 180s"


def test_a_slow_startup_that_starves_the_agent_is_the_worst_finding(tmp_path):
    life = lifecycle.read(_ctx(_run(tmp_path)))
    snapshot = {
        "generated_at": T0 + 1700,
        "experiment": {"state": "LIVE"},
        "sections": {"lifecycle": life},
    }
    findings = diagnose.diagnose(snapshot)
    first = findings[0]
    assert first["code"] == "AUTHOR_STARTUP_SLOW" and first["severity"] == "error"
    assert first["title"] == "Round 0 turn 1 spent 12m30s starting and the agent then worked 1s"
    assert "library_overlay 7m20s (22x its fastest)" in first["detail"]
    assert first["action"] and first["impact_s"] == 650.0
    stopped = next(f for f in findings if f["code"] == "AUTHOR_STOPPED")
    assert stopped["detail"] == "reason: integration wall slice"


def test_turns_end_when_the_author_stops(tmp_path):
    """A stream that stopped mid-turn (the process was interrupted) is not an agent still working."""
    index = agents.RolloutIndex()
    thread = agents._thread("t1")
    thread.update(file="x", role="lead", open_turn=T0 + 100, last_activity_at=T0 + 790, work_at=T0 + 790, turns=1)
    thread["state"] = "working"
    index.threads["t1"] = thread
    life = {"author_running": False, "turns": [{"startup_start": T0}], "phase": "ended", "phase_detail": "x"}
    session = agents.close_orphans(index, life, live=True)
    row = agents.view(thread, T0 + 5000, private=False, live=True)
    assert row["state"] == "stopped" and row["active_s"] == 690.0
    assert row["spans"][-1] == [T0 + 100, T0 + 790, "turn"]
    assert session == {"running": False, "phase": "ended", "since": None, "detail": "x"}
    # While a later turn runs, a thread silent since before it began belongs to the earlier turn.
    life = {"author_running": True, "turns": [{"startup_start": T0 + 900}]}
    agents.close_orphans(index, life, live=True)
    assert thread["closed_at"] == T0 + 790
    life = {"author_running": True, "turns": [{"startup_start": T0 + 500}]}
    agents.close_orphans(index, life, live=True)
    assert thread["closed_at"] is None


def test_an_unclosed_turn_ends_at_its_last_work_when_the_next_starts():
    index = agents.RolloutIndex()
    index.file_thread["f"] = "t1"
    index.threads["t1"] = agents._thread("t1")
    index.threads["t1"]["file"] = "f"

    def row(t, kind, payload):
        index._row("f", {"timestamp": _iso(t), "type": kind, "payload": payload})

    row(T0, "event_msg", {"type": "task_started", "started_at": T0})
    row(T0 + 600, "response_item", {"type": "message"})
    # Twelve minutes later the session resumes: bookkeeping rows, then a new turn.
    row(T0 + 1300, "event_msg", {"type": "thread_settings_applied"})
    row(T0 + 1301, "event_msg", {"type": "task_started", "started_at": T0 + 1301})
    thread = index.threads["t1"]
    assert thread["closed_turn_s"] == 600.0 and thread["turns"] == 2 and thread["open_turn"] == T0 + 1301


def test_lifecycle_events_read_as_milestones(tmp_path):
    run = _run(tmp_path)
    life = lifecycle.read(_ctx(run))
    ctx = _ctx(run, {"lifecycle": life})
    out: list = []
    events._lifecycle(ctx, out)
    texts = [e["text"] for e in out]
    assert "round 0 turn 1 startup finished in 12m30s (slowest: library_overlay 7m20s)" in texts
    assert "round 0 turn 1 ended: integration wall slice (agent worked 1s)" in texts
    assert "round 0 turn 0 admitted (exploration, 1.0M token allowance)" in texts
    assert all(e["kind"] == "author" for e in out)


def test_activity_buckets_keep_the_dominant_activity_and_halve_when_full():
    history = activity.History(T0)
    for i in range(10):
        history.add(T0 + i + 0.5, 1.0, 9, {"build": 0.9, "idle": 0.1}, 90.0)
    history.add(T0 + 10.5, 1.0, 9, {"L3": 1.0}, 100.0)
    encoded = history.encode()
    assert encoded["dt"] == activity.BUCKET_S and encoded["cores"][0]["codes"].startswith("bbbbb3")
    assert encoded["cores"][0]["util"].startswith("99999")
    assert round(history.totals["build"], 1) == 9.0 and "idle" not in history.totals
    # Far in the future: the buckets halve their resolution instead of growing.
    history.add(T0 + activity.BUCKET_S * activity.MAX_BUCKETS * 3, 1.0, 9, {"test": 1.0}, 50.0)
    encoded = history.encode()
    assert encoded["dt"] == activity.BUCKET_S * 4 and encoded["n"] <= activity.MAX_BUCKETS
    assert encoded["cores"][0]["codes"][0] == "b" and encoded["cores"][0]["codes"].rstrip()[-1] == "t"


def test_process_identity_names_the_activity():
    env = {"MERLIN_GSIM_EMU_X": "/opt/gsim/emulator", "MERLIN_SPIKE": "/opt/spike/bin/spike"}
    assert activity._own(["/opt/gsim/emulator", "+elf"], env) == ("L3", "gSIM")
    assert activity._own(["/opt/spike/bin/spike", "-p1"], env) == ("L2", "Spike")
    assert activity._own(["/usr/bin/codex", "exec"], env)[0] == "author"
    assert (
        activity._own(["python3", "-m", "merlin_experiments.phase1.brokers.selfcheck", "--ws", "w"], env)[0]
        == "validate"
    )
    assert activity._own(["python3", "-c", "x", "--model-grade", "spec.json"], env) == ("test", "model grade")
    assert (
        activity._own(["python3", "-m", "merlin_experiments.phase1.brokers.compiler_call_channel"], env)[0] == "build"
    )
    assert activity._own(["python3", "agent_selfcheck.py", "--tiers", "L3", "--sim", "gsim"], env) == ("L3", "gsim L3")
    assert activity._own(["python3", "-m", "merlin_experiments.phase1", "--descriptor", "d"], env) == (
        "runner",
        "runner",
    )
    assert activity._own(["/bin/cat", "x"], env) == (None, None)


def test_a_round_refused_before_its_turn_ran_no_agent(tmp_path):
    """A session end without its own time ends when its transcript was written; no turn means no agent."""
    run = _run(tmp_path)
    _write(
        run / "rounds" / "author_startup.jsonl",
        [
            *[json.loads(line) for line in (run / "rounds" / "author_startup.jsonl").read_text().splitlines()],
            _stage("sandbox_assembly", "c0", 0, "start", T0 + 2000, rnd=1),
            _stage("sandbox_assembly", "c0", 0, "finish", T0 + 3270, wall=1270.0, rnd=1),
        ],
    )
    transcript = run / "rounds" / "round_01.transcript.jsonl"
    _write(
        transcript,
        [
            {"type": "codex_summary", "turns_started": 0, "errors": ["continuation already admitted"]},
            {"type": "result", "subtype": "error", "result": "continuation already admitted"},
        ],
    )
    os.utime(transcript, (T0 + 3272, T0 + 3272))
    section = lifecycle.read(_ctx(run, now=T0 + 3600))
    turn = section["turns"][-1]
    assert (turn["round"], turn["startup_s"], turn["agent_start"], turn["agent_s"]) == (1, 1270.0, None, None)
    assert turn["agent_end"] == T0 + 3272 and turn["end_reason"] == "refused: continuation already admitted"
    assert section["phase"] == "ended" and section["phase_detail"] == "refused: continuation already admitted"


def test_agent_streams_know_their_round():
    assert agents.round_of("/x/codex/run-v1_r01/sessions/2026/10/24/rollout-a.jsonl", "run-v1") == 1
    assert agents.round_of("/x/run-v1/rounds/round_03.codex_rollout_snapshot/a.jsonl", "run-v1") == 3
    assert agents.round_of("/x/home/sessions/rollout-a.jsonl", "run-v1") is None
