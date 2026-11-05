"""Compact public feedback preserves complete diagnostics and bounds reader IO."""

import hashlib
import json
from pathlib import Path

from merlin_experiments.phase1.feedback import completed_grade as CG
from merlin_experiments.phase1.feedback import result_details as R
from merlin_experiments.phase1.feedback import stage_timing
from merlin_experiments.tracking.console.io import Reader


def _row(name="case", **extra):
    return {
        "capsule": name, "status": "pass", "tiers": {"L2": "pass"},
        "performance": {"status": "PROVISIONAL", "estimated_whole_soc_cycles": 123,
                        "complete_event_accounting": True,
                        "feedback_brief": {"description": "performance " * 2000},
                        "scoped_resource_feedback": {"resources": [{"name": "device", "activity": 4}]}},
        "compiler_diagnostics": {"feedback_brief": {"description": "different compiler " * 2000}},
        **extra,
    }


def _full(row):
    return {key: value for key, value in row.items() if key != "detail"}


def test_large_distinct_diagnostics_roundtrip_and_legacy_rows(tmp_path):
    row = _row(graded={"token": "1", "graded_source_sha256": "a" * 64})
    compact = R.compact_document({"per_capsule": [row], "all_pass": False}, workspace=tmp_path)
    assert len(json.dumps(compact["per_capsule"][0]).encode()) < 1024
    assert compact["all_pass"] is False
    R._CACHE.clear()
    hydrated = R.hydrate(compact, document_path=tmp_path / "qa" / CG.FILENAME)
    assert _full(hydrated["per_capsule"][0]) == row
    assert R.load_row(row, workspace=tmp_path) is row
    assert R.hydrate(None, workspace=tmp_path) is None


def test_unchanged_or_carried_result_keeps_one_detail_and_only_new_results_write(tmp_path):
    first = R.compact_row(_row(graded={"token": "1"}), workspace=tmp_path)
    path = tmp_path / first["detail"]["path"]
    identity = path.stat().st_mtime_ns
    carried = R.compact_row({**R.load_row(first, workspace=tmp_path), "graded": {"token": "2"}}, workspace=tmp_path)
    assert carried["detail"] == first["detail"] and path.stat().st_mtime_ns == identity
    assert carried["graded"] == {"token": "2"}
    changed = R.compact_row(_row(status="fail"), workspace=tmp_path)
    assert changed["detail"] != first["detail"]
    assert len(list((tmp_path / "qa" / "cases" / "case").glob("*.json"))) == 2
    revised = R.load_row(first, workspace=tmp_path)
    revised["performance"]["estimated_whole_soc_cycles"] = 999
    revised = R.compact_row(revised, workspace=tmp_path)
    assert revised["detail"] != first["detail"]
    assert R.load_row(first, workspace=tmp_path)["performance"]["estimated_whole_soc_cycles"] == 123


def test_invalid_missing_changed_and_external_details_keep_only_the_summary(tmp_path):
    row = R.compact_row(_row(), workspace=tmp_path)
    path = tmp_path / row["detail"]["path"]
    path.write_text('{"capsule":"case","status":"fail"}')
    assert R.load_row(row, workspace=tmp_path) is row  # invalidates a previously cached detail
    path.unlink()
    assert R.load_row(row, workspace=tmp_path) is row
    outside = tmp_path / "outside.json"
    payload = json.dumps(_row()).encode()
    outside.write_bytes(payload)
    escaped = {**row, "detail": {"path": "outside.json", "sha256": hashlib.sha256(payload).hexdigest()}}
    assert R.load_row(escaped, workspace=tmp_path) is escaped
    directory = tmp_path / "qa" / "cases" / "linked"
    directory.symlink_to(tmp_path, target_is_directory=True)
    escaped["detail"]["path"] = "qa/cases/linked/outside.json"
    assert R.load_row(escaped, workspace=tmp_path) is escaped


def test_console_hydrates_with_its_ordinary_budget_and_legacy_values_match(tmp_path):
    full = {"per_capsule": [_row()]}
    compact = R.compact_document(full, workspace=tmp_path)
    path = tmp_path / "qa" / CG.FILENAME
    path.write_text(json.dumps(compact))
    reader = Reader(budget=None)
    result = reader.document(path)
    assert _full(result["per_capsule"][0]) == full["per_capsule"][0]
    tiny = Reader(budget=path.stat().st_size + 1)
    limited = tiny.document(path)
    assert limited["per_capsule"][0] == compact["per_capsule"][0]
    assert tiny.notices
    tiny.remaining = 1 << 20
    assert _full(tiny.document(path)["per_capsule"][0]) == full["per_capsule"][0]


def test_publication_appends_only_changed_rows_and_keeps_detail_pointer(tmp_path):
    grade = {"per_capsule": [_row()], "graded_source_sha256": "a" * 64}
    assert CG.record(tmp_path, grade, 1)
    events = tmp_path / "qa" / "completed_rows.jsonl"
    assert len(events.read_text().splitlines()) == 1
    assert not CG.record(tmp_path, grade, 1)
    assert len(events.read_text().splitlines()) == 1
    assert CG.record_rows(tmp_path, [_row("new")], 2, graded_source_sha256="b" * 64)
    assert len(events.read_text().splitlines()) == 2
    assert not CG.record_rows(tmp_path, [_row("new")], 2, graded_source_sha256="b" * 64)
    assert len(events.read_text().splitlines()) == 2
    latest = json.loads((tmp_path / "qa" / CG.FILENAME).read_text())
    assert all(row.get("detail") for row in latest["per_capsule"])


def test_profiling_keeps_original_error_and_records_wall_time(tmp_path):
    document = {}
    try:
        with stage_timing.measure("original_check", document, events=tmp_path / "times.jsonl"):
            raise ValueError("original error")
    except ValueError as error:
        assert str(error) == "original error"
    assert document["host_stage_timings_s"]["original_check"] >= 0
    assert json.loads((tmp_path / "times.jsonl").read_text())["stage"] == "original_check"
