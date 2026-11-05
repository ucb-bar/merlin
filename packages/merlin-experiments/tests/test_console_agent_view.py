"""The agent-view source reads only what the agent can see, and its findings name what never reached it."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from merlin_experiments.tracking.console import diagnose
from merlin_experiments.tracking.console.io import Reader
from merlin_experiments.tracking.console.model import Context
from merlin_experiments.tracking.console.sources import agent_view, machine

NOW = time.time()


def _write(path: Path, document: dict, at: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document))
    os.utime(path, (at, at))


def _run(tmp_path: Path, *, delivered: bool):
    """A run whose newest grade (252 rows) finished 13 minutes ago, shaped like the live records."""
    run, ws = tmp_path / "run", tmp_path / "ws" / "workspace"
    token = int((NOW - 780) * 1e9)
    grade = {"per_capsule": [{"capsule": f"c{i}", "status": "pass"} for i in range(252)],
             "graded_source_sha256": "f" * 64, "graded_at": NOW - 780}
    _write(run / "qa_history" / f"verdict_fast_{token}.json", grade, NOW - 780)
    _write(run / "qa_history" / f"verdict_fast_{token - 10**12}.json", {"per_capsule": []}, NOW - 2000)
    _write(ws / "qa" / "verdict.json", {"model_gate": {"programs": {}}, "all_pass": None}, NOW - 90)
    _write(ws / "qa" / "progress.json", {"per_capsule": [{}] * 105, "updated_at": NOW - 2370,
                                          "graded_source_sha256": "a" * 64}, NOW - 2370)
    if delivered:
        latest = dict(grade, latest_completed_grade={"snapshot_current": False, "graded_source_sha256": "f" * 64,
                                                     "graded_at": NOW - 780},
                      feedback_brief={"performance_priority": {"development_headroom": {
                          "counts": {"ranked": 62, "passing_l2": 120}, "median_array_utilization_pct": 5.25}}})
        _write(ws / "qa" / "latest_completed_grade.json", latest, NOW - 770)
        _write(ws / "OPTIMIZATION_SUGGESTIONS.json", {"suggestions": [{}] * 8, "pending_workload_check": [{}] * 20,
                                                      "discovery": [{}] * 12}, NOW - 7000)
    channel = ws / ".qa_channel"
    for i, age in enumerate((3000, 600, 120)):
        rid = f"2_{int((NOW - age) * 1e9)}_{i:032x}"
        _write(channel / f"req_{rid}.json", {"request_id": rid}, NOW - age)
        if age != 3000:
            _write(channel / f"resp_{rid}.json", {"all_pass": False}, NOW - age + 30)
    experiment = {"id": str(run), "run_id": "r", "run_dir": str(run), "workspace": str(ws), "state": "LIVE"}
    return Context(experiment=experiment, reader=Reader(), state={}, operator_private=True, now=NOW, shared={}, live=True)


def _codes(section: dict) -> dict:
    snapshot = {"generated_at": NOW, "sections": {"agent_view": section}, "experiment": {}}
    return {f["code"]: f for f in diagnose.diagnose(snapshot)}


def test_an_undelivered_grade_and_an_empty_verdict_are_findings(tmp_path):
    section = agent_view.read(_run(tmp_path, delivered=False))
    assert section["verdict"]["model_gate_only"] and section["verdict"]["rows"] == 0
    assert section["newest_grade"]["rows"] == 252 and section["newest_grade"]["delivered"] == []
    assert section["selfchecks"]["pending"] == 1 and section["selfchecks"]["answered"] == 2
    codes = _codes(section)
    for code in ("AGENT_GRADE_NOT_DELIVERED", "AGENT_VERDICT_EMPTY", "AGENT_FEEDBACK_STALE",
                 "AGENT_SUGGESTIONS_MISSING", "AGENT_SELFCHECK_UNANSWERED"):
        assert code in codes, code
    assert codes["AGENT_GRADE_NOT_DELIVERED"]["evidence"].endswith(".json")


def test_a_grade_in_latest_completed_counts_as_delivered(tmp_path):
    section = agent_view.read(_run(tmp_path, delivered=True))
    assert section["newest_grade"]["delivered"] == ["latest_completed_grade.json"]
    assert section["latest_completed"]["snapshot_current"] is False
    assert section["latest_completed"]["headroom"] == {"ranked": 62, "passing_l2": 120, "median_utilization_pct": 5.25}
    assert section["suggestions"] == {"present": True, "suggestions": 8, "pending_workload_check": 20, "discovery": 12}
    codes = _codes(section)
    assert "AGENT_GRADE_NOT_DELIVERED" not in codes and "AGENT_VERDICT_EMPTY" not in codes
    assert "AGENT_SUGGESTIONS_MISSING" not in codes


def test_a_run_without_a_workspace_reports_nothing(tmp_path):
    ctx = Context(experiment={"id": "x"}, reader=Reader(), state={}, operator_private=True, now=NOW, shared={})
    section = agent_view.read(ctx)
    assert section["workspace"] is None and _codes(section) == {}


def test_volume_levels_and_the_disk_finding(tmp_path, monkeypatch):
    class Stat:
        f_frsize = 4096
        f_blocks = 1000 << 18  # ~1 TB

        def __init__(self, free_blocks):
            self.f_bavail = free_blocks

    for free_blocks, level in ((260 << 18, "ok"), (40 << 18, "warn"), (2 << 18, "critical")):
        monkeypatch.setattr(machine.os, "statvfs", lambda _p, n=free_blocks: Stat(n))
        assert machine.volume(tmp_path)["level"] == level
    monkeypatch.setattr(machine.os, "statvfs", lambda _p: Stat(2 << 18))
    snapshot = {"sections": {"machine": {"volume": machine.volume(tmp_path)}}, "experiment": {}}
    finding = next(f for f in diagnose.diagnose(snapshot) if f["code"] == "DISK_LOW")
    assert finding["severity"] == "error"
    assert machine.volume(None) is None


def _deferred_rows(count: int) -> list:
    return [{"capsule": f"c{i}", "status": "gated", "tiers": {}, "failure_plane": "gate",
             "failure_category": "ADMISSION_DEFERRED", "failure_detail": "background source superseded"}
            for i in range(count)]


def _replace_newest(tmp_path: Path, rows: list) -> None:
    newest = max((tmp_path / "run" / "qa_history").glob("verdict_fast_*.json"),
                 key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    _write(newest, {"per_capsule": rows, "graded_source_sha256": "f" * 64, "graded_at": NOW - 780}, NOW - 780)


def test_an_all_deferred_grade_executed_nothing_and_is_not_undelivered(tmp_path):
    ctx = _run(tmp_path, delivered=False)
    _replace_newest(tmp_path, _deferred_rows(252))
    section = agent_view.read(ctx)
    assert section["newest_grade"]["rows"] == 252 and section["newest_grade"]["executed"] == 0
    assert section["verdict"]["rows"] == 0 and not section["latest_completed"].get("rows")
    codes = _codes(section)
    assert "AGENT_GRADE_NOT_DELIVERED" not in codes and "AGENT_VERDICT_EMPTY" not in codes


def test_a_grade_with_executed_rows_that_reached_no_agent_file_is_still_flagged(tmp_path):
    ctx = _run(tmp_path, delivered=False)
    _replace_newest(tmp_path, _deferred_rows(251) + [{"capsule": "ran", "status": "pass", "tiers": {"L2": "pass"}}])
    section = agent_view.read(ctx)
    assert section["newest_grade"]["executed"] == 1 and section["newest_grade"]["delivered"] == []
    codes = _codes(section)
    assert "252 rows, 1 executed" in codes["AGENT_GRADE_NOT_DELIVERED"]["title"]
    assert "AGENT_VERDICT_EMPTY" in codes  # it ran a case, yet the agent's verdict shows no row


def test_policy_refused_selfchecks_are_not_selfcheck_errors(tmp_path):
    ctx = _run(tmp_path, delivered=True)
    channel = tmp_path / "ws" / "workspace" / ".qa_channel"
    for i in range(4):
        rid = f"3_{int((NOW - 60 + i) * 1e9)}_{i:032x}"
        _write(channel / f"req_{rid}.json", {}, NOW - 60 + i)
        _write(channel / f"resp_{rid}.json", {
            "all_pass": False, "joint_policy_refused": True,
            "error": "joint gSIM correctness concurrency or worker budget exhausted; obligation remains pending"},
            NOW - 50 + i)
    section = agent_view.read(ctx)
    assert section["selfchecks"]["recent_errors"] == 0 and section["selfchecks"]["recent_refusals"] == 4
    assert "AGENT_SELFCHECK_ERRORS" not in _codes(section)
