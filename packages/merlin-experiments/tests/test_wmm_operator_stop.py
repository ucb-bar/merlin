"""An operator stops a measured run at its next SESSION BOUNDARY by writing a request into the run
directory -- never by signalling it: a round frozen or killed mid-flight is recorded as a failure, not a
stop.  The request is read before anything of a session starts, the watchdog never undoes it, and both
command lines write it."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2.whole_model_measured import cli as MCLI
from merlin_experiments.phase2.whole_model_measured import runs as RUNS
from merlin_experiments.phase2.whole_model_measured import sessions as SES
from merlin_experiments.phase2.whole_model_measured import watchdog as WD


class _Objective:
    def __init__(self, root: Path):
        root.mkdir(parents=True, exist_ok=True)
        self.screen = SimpleNamespace(root=root, history=lambda: [])
        self.config = {"plateau_hours": 6, "plateau_min_sessions": 2, "stop_at_bar": False}
        self.marked = 0

    def poll(self):
        return None

    def summary(self):
        return {"best": {"package_sha256": "b", "screen_whole_window_cycles": 1000}}

    def mark_session(self):
        self.marked += 1


def _run_dir(tmp_path: Path, name: str = "run") -> Path:
    run_dir = tmp_path / name
    (run_dir / "stage").mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps({"schema": RUNS.RUN_SCHEMA, "target": "toy", "method": "m"}))
    return run_dir


def test_no_request_means_no_stop_and_a_request_states_why(tmp_path):
    run_dir = _run_dir(tmp_path)
    assert SES.operator_stop(run_dir / SES.OPERATOR_STOP_FILE) is None
    with pytest.raises(ValueError, match="states why"):
        SES.request_stop(run_dir, why="  ")
    request = SES.request_stop(run_dir, why="reseed from the composite best", operator="op")
    evidence = SES.operator_stop(run_dir / SES.OPERATOR_STOP_FILE)
    assert evidence["kind"] == SES.OPERATOR_STOP and evidence["request"]["why"] == request["why"]
    assert "reseed from the composite best" in evidence["reason"]


def test_the_first_request_stands_and_an_unreadable_one_still_stops(tmp_path):
    run_dir = _run_dir(tmp_path)
    SES.request_stop(run_dir, why="first reason")
    again = SES.request_stop(run_dir, why="second reason")
    assert again["already_requested"] is True and again["why"] == "first reason"
    (run_dir / SES.OPERATOR_STOP_FILE).write_text("{not json")
    assert SES.operator_stop(run_dir / SES.OPERATOR_STOP_FILE)["request"]["why"] == "unreadable request file"


def test_the_request_is_read_before_anything_of_the_next_session_starts(tmp_path):
    """Session 1 runs to its end; the request made during it stops the loop before session 2 marks the
    stagnation clock, writes a plateau row or starts a round."""
    run_dir = _run_dir(tmp_path)
    objective = _Objective(tmp_path / "store")
    started = []

    def run_round(*, session, stage_root):
        started.append(session)
        SES.request_stop(run_dir, why="operator pause")
        return {"status": "authored"}

    document = SES.run_sessions(
        objective,
        run_round=run_round,
        stage_root=run_dir / "stage",
        run="r",
        max_sessions=5,
        total_seconds=1e9,
        stop_request=run_dir / SES.OPERATOR_STOP_FILE,
    )
    assert started == [1] and objective.marked == 1
    assert document["stopped"]["kind"] == SES.OPERATOR_STOP and document["stopped_on_evidence"] is False
    plateau = json.loads((tmp_path / "store" / SES.PLATEAU_FILE).read_text())
    assert [row["session"] for row in plateau["trace"]] == [1]
    assert json.loads((run_dir / "stage" / "sessions.json").read_text())["stopped"]["kind"] == SES.OPERATOR_STOP


def test_the_watchdog_never_relaunches_a_run_an_operator_stopped(tmp_path):
    """Even when the launcher died before reaching its boundary (no sessions.json), and even when the
    host's memory guard stopped it -- a relaunch would undo the operator's request."""
    run_dir = _run_dir(tmp_path)
    (run_dir / "stage" / "host_resource_telemetry.json").write_text(json.dumps({"status": "resource_limit"}))
    SES.request_stop(run_dir, why="hold the board for a control run")
    clock = iter(float(t) for t in range(0, 100_000, 1000))
    document = WD.watch(
        run_dir,
        1,
        launch=lambda prepared: pytest.fail("relaunched a run an operator stopped"),
        why="x",
        is_alive=lambda pid: False,
        memory=lambda: 100.0,
        sleep=lambda s: None,
        clock=lambda: next(clock),
        log=lambda line: None,
    )
    assert document["stopped"]["kind"] == SES.OPERATOR_STOP and document["relaunches"] == []


def test_the_command_lines_write_the_request_for_a_run_or_its_orchestration(tmp_path, monkeypatch, capsys):
    from merlin_experiments import cli as TOP

    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    run_dir = _run_dir(tmp_path)
    assert MCLI.main(["stop", str(run_dir), "--why", "reseed"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["request"]["why"] == "reseed" and printed["launcher_alive"] is None
    assert (run_dir / SES.OPERATOR_STOP_FILE).is_file()

    orchestration = tmp_path / "orchestration"
    (orchestration / "phase2").mkdir(parents=True)
    other = _run_dir(tmp_path, "other")
    (orchestration / "phase2" / "phase_run.json").write_text(json.dumps({"run_dir": str(other)}))
    assert TOP.main(["stop", str(orchestration), "--why", "end of the budget window"]) == 0
    assert json.loads((other / SES.OPERATOR_STOP_FILE).read_text())["why"] == "end of the budget window"
    assert TOP.main(["stop", str(tmp_path / "nothing"), "--why", "x"]) == 2


def test_stop_names_the_run_a_watchdog_already_resumed_into(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    root = tmp_path / "out" / "runs" / "toy" / "phase2"
    first = _run_dir(root, "20261001T000000Z_m_abcdef0")
    second = _run_dir(root, "20261002T000000Z_m_abcdef0")
    (second / "resumed_seed.json").write_text(json.dumps({"resumed_from_run": str(first)}))
    document = MCLI.stop(first, why="stop the old one")
    assert document["resumed_into"] == [str(second)]
