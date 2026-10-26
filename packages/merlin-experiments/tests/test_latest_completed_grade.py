"""Finished grades reach the author even when the source moved on; newer grades always win."""

from __future__ import annotations

import json
from types import SimpleNamespace

from merlin_experiments.phase1.feedback import completed_grade as CG
from merlin_experiments.phase1.feedback import grade_progress, snapshot_publication
from merlin_experiments.phase1.feedback.visible_model_gate import publish


def _ws(tmp_path):
    ws = tmp_path / "ws"
    (ws / "qa").mkdir(parents=True)
    (ws / "qa" / "verdict.json").write_text(json.dumps({"model_gate": {"source_sha256": None}, "all_pass": None}))
    return ws


def _grade(source, *, current, rows=1):
    return {"per_capsule": [{"capsule": f"t{i}", "status": "pass", "tiers": {"L2": "pass"}} for i in range(rows)],
            "graded_source_sha256": source, "graded_at": f"at-{source}", "snapshot_current": current,
            "feedback_brief": {"performance_priority": {"advisory_only": True}}}


def _live_writer(calls):
    def write(path, verdict):
        calls.append(verdict["graded_source_sha256"])
        return publish(path, verdict)
    return write


def test_a_superseded_grade_is_shown_as_latest_completed_evidence(tmp_path):
    ws, calls = _ws(tmp_path), []
    CG.publish(ws, _grade("old", current=False), 100, _live_writer(calls))
    latest = json.loads((ws / "qa" / CG.FILENAME).read_text())
    assert latest["per_capsule"] and latest["snapshot_current"] is False
    assert latest[CG.POINTER]["graded_source_sha256"] == "old" and latest[CG.POINTER]["token"] == "100"
    assert "development_headroom" in latest["feedback_brief"]["performance_priority"]
    live = json.loads((ws / "qa" / "verdict.json").read_text())
    assert calls == [] and live == {"model_gate": {"source_sha256": None}, "all_pass": None}  # live view untouched


def test_an_older_grade_finishing_late_only_fills_cases_the_newer_one_lacks(tmp_path):
    ws = _ws(tmp_path)
    CG.publish(ws, _grade("newer", current=False, rows=3), 200, _live_writer([]))
    CG.publish(ws, _grade("older", current=False, rows=5), 150, _live_writer([]))
    latest = json.loads((ws / "qa" / CG.FILENAME).read_text())
    assert latest["graded_source_sha256"] == "newer" and latest[CG.POINTER]["token"] == "200"
    sources = {row["capsule"]: row[CG.ORIGIN]["graded_source_sha256"] for row in latest["per_capsule"]}
    assert sources == {"t0": "newer", "t1": "newer", "t2": "newer", "t3": "older", "t4": "older"}
    assert [row["capsule"] for row in latest["per_capsule"]][:3] == ["t0", "t1", "t2"]


def test_a_current_grade_stays_authoritative_and_is_also_the_latest(tmp_path):
    ws, calls = _ws(tmp_path), []
    CG.publish(ws, _grade("now", current=True), 300, _live_writer(calls))
    live = json.loads((ws / "qa" / "verdict.json").read_text())
    assert calls == ["now"] and live["per_capsule"] and live["graded_source_sha256"] == "now"
    assert json.loads((ws / "qa" / CG.FILENAME).read_text())[CG.POINTER]["snapshot_current"] is True


def test_an_empty_grade_never_replaces_completed_evidence(tmp_path):
    ws = _ws(tmp_path)
    CG.publish(ws, _grade("full", current=False), 400, _live_writer([]))
    CG.publish(ws, {**_grade("empty", current=False), "per_capsule": []}, 500, _live_writer([]))
    assert json.loads((ws / "qa" / CG.FILENAME).read_text())["graded_source_sha256"] == "full"


def test_the_newest_owned_snapshot_refreshes_progress_even_when_not_current(tmp_path, monkeypatch):
    ws = _ws(tmp_path)
    (ws / "qa" / "progress.json").write_text(json.dumps({"graded_source_sha256": "seed", "n_completed": 110}))
    runs = tmp_path / "run" / "_qa_work" / "fruns_9"
    runs.mkdir(parents=True)
    monkeypatch.setattr(grade_progress.qa, "_per_capsule_from_results",
                        lambda root, only_capsule=None: {only_capsule: {"status": "pass"}})
    guard = SimpleNamespace(current=lambda: False, reason=lambda: None)
    progress = grade_progress.SnapshotProgress(ws, runs, "newer", guard)
    snapshot_publication.claim(ws, "9")
    with snapshot_publication.scope(ws, "9"):
        progress.completed({"capsule": "t0"})
    shown = json.loads((ws / "qa" / "progress.json").read_text())
    assert shown["graded_source_sha256"] == "newer" and shown["n_completed"] == 1
    # A grade that no longer owns publication, or runs outside any publication scope, cannot replace it.
    stale = grade_progress.SnapshotProgress(ws, runs, "stale", guard)
    with snapshot_publication.scope(ws, "8"):
        stale.completed({"capsule": "t1"})
    stale.completed({"capsule": "t2"})
    assert json.loads((ws / "qa" / "progress.json").read_text())["graded_source_sha256"] == "newer"


def test_an_all_gated_grade_never_replaces_completed_evidence(tmp_path):
    ws = _ws(tmp_path)
    CG.publish(ws, _grade("full", current=False), 400, _live_writer([]))
    gated = [{"capsule": f"t{i}", "status": "gated", "tiers": {}, "failure_plane": "gate",
              "failure_category": "ADMISSION_DEFERRED"} for i in range(252)]
    CG.publish(ws, {**_grade("gated", current=False), "per_capsule": gated}, 500, _live_writer([]))
    assert json.loads((ws / "qa" / CG.FILENAME).read_text())["graded_source_sha256"] == "full"
    # One executed row is evidence: a mostly gated newer grade still wins.
    mixed = [*gated[:-1], {"capsule": "t9", "status": "fail", "tiers": {"L2": "fail"}}]
    CG.publish(ws, {**_grade("mixed", current=False), "per_capsule": mixed}, 600, _live_writer([]))
    assert json.loads((ws / "qa" / CG.FILENAME).read_text())["graded_source_sha256"] == "mixed"


def _row(name, status, tiers=None, **extra):
    return {"capsule": name, "status": status, "tiers": tiers or {}, **extra}


def test_a_mostly_gated_newer_grade_keeps_the_older_grades_results_per_case(tmp_path):
    """The live failure: 52 passes graded earlier were hidden by a newer grade that ran a few cases."""
    ws = _ws(tmp_path)
    older = [_row(f"t{i}", "pass", {"L2": "pass"}) for i in range(52)]
    CG.publish(ws, {**_grade("older", current=False), "per_capsule": older}, 100, _live_writer([]))
    gated = [_row(f"t{i}", "gated", failure_category="ADMISSION_DEFERRED") for i in range(1, 52)]
    newer = [_row("t0", "fail", {"L2": "fail"}), *gated, _row("t60", "pass", {"L2": "pass"})]
    CG.publish(ws, {**_grade("newer", current=True), "per_capsule": newer, "current_source_sha256": "newer"},
               200, _live_writer([]))
    latest = json.loads((ws / "qa" / CG.FILENAME).read_text())
    rows = {row["capsule"]: row for row in latest["per_capsule"]}
    assert rows["t0"]["status"] == "fail" and rows["t0"][CG.ORIGIN]["token"] == "200"  # newer result wins
    assert all(rows[f"t{i}"]["status"] == "pass" and rows[f"t{i}"][CG.ORIGIN]["graded_source_sha256"] == "older"
               for i in range(1, 52))  # a gated row never hides a result
    assert rows["t60"][CG.ORIGIN]["graded_source_sha256"] == "newer"
    summary = latest[CG.SUMMARY]
    assert summary["rows"] == 53 and summary["passed"] == 52 and summary["from_newest_grade"] == 2
    assert summary["from_current_source"] == 2 and summary["recorded_grades"] == ["100", "200"]
    assert [entry["graded_source_sha256"] for entry in summary["by_grade"]] == ["newer", "older"]
    # Rows from older bytes are present, so the merged file never claims to describe current source.
    assert latest["snapshot_current"] is False and latest[CG.POINTER]["snapshot_current"] is False
    assert latest["graded_source_sha256"] == "newer"


def test_a_grade_is_recorded_once_and_a_late_grade_with_nothing_new_changes_nothing(tmp_path):
    ws = _ws(tmp_path)
    CG.publish(ws, _grade("newer", current=False, rows=3), 200, _live_writer([]))
    before = (ws / "qa" / CG.FILENAME).read_text()
    assert CG.record(ws, _grade("newer", current=False, rows=3), 200) is False
    assert CG.record(ws, _grade("older", current=False, rows=2), 150) is False  # newer rows already cover it
    assert (ws / "qa" / CG.FILENAME).read_text() == before


def test_a_file_written_before_row_labels_keeps_its_rows_attributed(tmp_path):
    ws = _ws(tmp_path)
    legacy = {**_grade("legacy", current=False, rows=2), CG.POINTER: {"token": "100"}}
    (ws / "qa" / CG.FILENAME).write_text(json.dumps(legacy))
    CG.publish(ws, {**_grade("new", current=False), "per_capsule": [_row("t9", "pass", {"L2": "pass"})]},
               200, _live_writer([]))
    rows = {row["capsule"]: row[CG.ORIGIN] for row in json.loads((ws / "qa" / CG.FILENAME).read_text())["per_capsule"]}
    assert rows == {"t9": {"token": "200", "graded_source_sha256": "new", "graded_at": "at-new"},
                    "t0": {"token": "100", "graded_source_sha256": "legacy", "graded_at": "at-legacy"},
                    "t1": {"token": "100", "graded_source_sha256": "legacy", "graded_at": "at-legacy"}}
