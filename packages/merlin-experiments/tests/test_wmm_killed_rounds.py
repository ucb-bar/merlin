"""A round whose agent or driver is KILLED: the agent's death is the round's recorded failure (its
bytes are not carried forward and its requests are never anyone's best), a driver killed mid-round
leaves an open marker the next ``start`` closes, and a relaunched ``start`` on the same run continues
its round numbering instead of colliding with a used round workspace."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from merlin_experiments.phase2.whole_model_measured import jobs as J
from merlin_experiments.phase2.whole_model_measured import rounds as R
from merlin_experiments.phase2.whole_model_measured import service as S
from merlin_experiments.phase2.whole_model_measured import sessions as SES


def _exec_done(child) -> None:
    """Wait until the child has exec'd: until then /proc shows an empty command line."""
    for _ in range(500):
        if Path(f"/proc/{child.pid}/cmdline").read_bytes():
            return
        time.sleep(0.01)
    raise AssertionError("the child never exec'd")


def test_the_next_session_is_one_past_every_round_the_stage_started(tmp_path):
    stage = tmp_path / "stage"
    assert R.next_session(stage) == 1
    (stage / "rounds").mkdir(parents=True)
    (stage / "rounds" / "round_00.round.json").write_text("{}")
    (stage / "rounds" / "round_00.transcript.jsonl").write_text("")
    assert R.next_session(stage) == 2
    (stage / "agent_workspaces" / "round_02").mkdir(parents=True)  # started, never recorded
    (stage / "rounds" / "notes.txt").write_text("")
    assert R.next_session(stage) == 4


def test_a_round_a_live_driver_owns_is_left_open(tmp_path):
    rounds = tmp_path / "stage" / "rounds"
    rounds.mkdir(parents=True)
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)", "live-run-name"])
    _exec_done(child)
    try:
        (rounds / f"round_03{R.OPEN_SUFFIX}").write_text(json.dumps({"pid": child.pid, "requested": ["d"]}))
        calls = []
        assert (
            R.recover_killed_rounds(tmp_path / "stage", attribute=lambda *a: calls.append(a), run_name="live-run-name")
            == []
        )
        assert calls == [] and (rounds / f"round_03{R.OPEN_SUFFIX}").exists()
    finally:
        child.kill()
        child.wait()
    # A round that ended (its record exists) only lost its marker's cleanup: nothing is re-attributed.
    (rounds / f"round_03{R.OPEN_SUFFIX}").write_text(json.dumps({"pid": os.getpid(), "requested": ["d"]}))
    (rounds / "round_03.round.json").write_text("{}")
    assert (
        R.recover_killed_rounds(tmp_path / "stage", attribute=lambda *a: pytest.fail("attributed"), run_name="x") == []
    )
    assert not (rounds / f"round_03{R.OPEN_SUFFIX}").exists()


def test_a_request_without_a_job_is_recorded_not_raised(tmp_path):
    rounds = tmp_path / "stage" / "rounds"
    rounds.mkdir(parents=True)
    (rounds / f"round_00{R.OPEN_SUFFIX}").write_text(json.dumps({"pid": 2**22 + 1, "requested": ["gone"]}))

    def attribute(digest, change):
        raise J.ServiceError(f"no job {digest} to attribute")

    (record,) = R.recover_killed_rounds(tmp_path / "stage", attribute=attribute, run_name="x")
    assert record["attribution"]["gone"].startswith("not attributed")


def test_start_closes_killed_rounds_and_continues_the_numbering(tmp_path, monkeypatch):
    """`start` on a run whose previous launcher was killed mid-round: the round is closed and the session
    loop begins after it, with the operator's stop file wired in."""
    from types import SimpleNamespace

    from merlin_experiments.phase2.whole_model_measured import cli, identity
    from merlin_experiments.phase2.whole_model_measured import profiles as P

    run_dir = tmp_path / "run"
    (run_dir / "stage" / "rounds").mkdir(parents=True)
    (run_dir / "stage" / "rounds" / f"round_04{R.OPEN_SUFFIX}").write_text(
        json.dumps({"pid": 2**22 + 1, "requested": ["abc"]})
    )
    attributed = []
    objective = SimpleNamespace(
        measure=lambda *a, **k: None,
        attribute=lambda digest, change: attributed.append((digest, change["state"])) or {"state": change["state"]},
        screen=SimpleNamespace(attribute=lambda *a: pytest.fail("not the previous run's")),
    )
    monkeypatch.setattr(cli, "_objective_of", lambda path: ({}, objective))
    monkeypatch.setattr(P, "check", lambda profile, price_table: {"resolved_model": profile["model"]})
    monkeypatch.setattr(identity, "load_builder", lambda spec, **kw: lambda **driver_kw: None)
    seen = {}
    monkeypatch.setattr(SES, "run_sessions", lambda objective, **kw: seen.update(kw) or {"stopped": None})
    monkeypatch.setattr(S, "spawn", lambda *a, **k: pytest.fail("spawned"))
    cli.start(run_dir, profile_name=P.names()[0], round_driver=cli.DEFAULT_ROUND_DRIVER, price_table=tmp_path / "p")
    assert attributed == [("abc", J.ATTRIBUTION_UNAUTHORED)]
    assert seen["first_session"] == 6 and seen["stop_request"] == run_dir / SES.OPERATOR_STOP_FILE
