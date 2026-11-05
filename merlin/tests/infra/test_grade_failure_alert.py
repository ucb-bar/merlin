"""Repeated identical in-turn grade failures must raise an operator alert, not 75 quiet "skipped" lines.

Measured on one 5 h authoring run: every one of 75 consecutive in-turn grades died with the same
``StageGateError`` ("operator grade child exited 1 without a verified completion"). Each was printed as an
ordinary skipped grade, the operator health record listed no infrastructure issue, and the agent worked
the whole run without a single in-turn grade landing.
"""

from __future__ import annotations

import json
import threading
import time
import types

import pytest
from merlin_experiments.phase1.feedback import lifecycle as FL


def _shrink_waits(monkeypatch, cap: float = 0.01) -> None:
    real = threading.Event.wait

    def fast(self, timeout=None):
        return real(self, cap if timeout is None else min(timeout, cap))

    monkeypatch.setattr(threading.Event, "wait", fast, raising=True)


def _spin(pred, seconds: float = 5.0) -> None:
    deadline = time.time() + seconds
    while not pred() and time.time() < deadline:
        time.sleep(0.01)


def _workspace(tmp_path):
    ws = tmp_path / "ws"
    (ws / "submission").mkdir(parents=True)
    (ws / "qa").mkdir()
    (ws / "qa" / "verdict.json").write_text("{}")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    return ws, run_dir


def _cadence():
    return types.SimpleNamespace(grade_interval=900, qa_timeout=60, no_oracle=False)


class StageGateError(RuntimeError):
    pass


def test_identical_failures_raise_the_alert_and_a_completed_grade_clears_it(tmp_path, monkeypatch, capsys):
    ws, run_dir = _workspace(tmp_path)
    monkeypatch.setenv(FL.GRADE_FAILURE_ALERT_ENV, "3")
    calls = []

    def grade(_ws, _rd, tick, _no_oracle, _timeout, **_kw):
        calls.append(tick)
        if len(calls) <= 4:
            # The tick differs on every call; the failure is still the same failure.
            raise StageGateError(f"operator grade child {tick} exited 1 without a verified completion")
        return {"all_pass": False, "n_passed": 1, "n_capsules": 2}

    _shrink_waits(monkeypatch)
    handle = FL.start_background(
        ws, run_dir, _cadence(), interval_grades=True, grade_callback=grade, fast_grade_callback=None
    )
    _spin(lambda: len(calls) >= 3)
    _spin(lambda: json.loads((run_dir / "grade_health.json").read_text()).get("alert") is True, 2)
    during = json.loads((run_dir / "grade_health.json").read_text())
    _spin(lambda: len(calls) >= 6)
    FL.stop_background(handle)

    assert during["alert"] is True and during["consecutive_identical_failures"] >= 3
    assert during["error"].startswith("StageGateError: operator grade child")
    out = capsys.readouterr().out
    assert out.count("[grade-alert] 3 consecutive in-turn grades failed with the same error") == 1
    after = json.loads((run_dir / "grade_health.json").read_text())
    assert after["alert"] is False and after["consecutive_identical_failures"] == 0
    assert after["alerts_raised"] == 1 and after["last_completed_at"]
    assert not (ws / "grade_health.json").exists() and not (ws / "qa" / "grade_health.json").exists()


def test_different_failures_do_not_accumulate(tmp_path):
    # An absent run directory makes the record unwritable: a diagnostic miss, never a raised error.
    monitor = FL.GradeFailureAlert(tmp_path / "absent", threshold=2)
    monitor.failed(ValueError("submission half-written"))
    monitor.failed(StageGateError("grade child exited 1"))
    assert monitor.count == 1 and not monitor.alert
    monitor.failed(StageGateError("grade child exited 2"))
    assert monitor.count == 2 and monitor.alert
    monitor.succeeded()
    assert monitor.count == 0 and not monitor.alert


def test_zero_records_without_alerting_and_bad_values_refuse(tmp_path, monkeypatch, capsys):
    monitor = FL.GradeFailureAlert(tmp_path, threshold=0)
    for _ in range(5):
        monitor.failed(StageGateError("same"))
    record = json.loads((tmp_path / "grade_health.json").read_text())
    assert record["consecutive_identical_failures"] == 5 and record["alert"] is False
    assert "[grade-alert]" not in capsys.readouterr().out
    for value in ("-1", "three", "1001"):
        monkeypatch.setenv(FL.GRADE_FAILURE_ALERT_ENV, value)
        with pytest.raises(ValueError):
            FL.grade_failure_alert_threshold()
    monkeypatch.delenv(FL.GRADE_FAILURE_ALERT_ENV)
    assert FL.grade_failure_alert_threshold() == FL.GRADE_FAILURE_ALERT_DEFAULT
