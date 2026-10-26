"""A grade that aborts before producing results never erases the agent's last real results."""

from __future__ import annotations

import json
import os
from pathlib import Path

from merlin_experiments.phase1.feedback import visible_model_gate as G


def _grade(rows, **extra):
    return {"per_capsule": rows, "n_capsules": len(rows), "n_passed": sum(r["status"] == "pass" for r in rows),
            "all_pass": False, **extra}


def test_an_aborted_grade_keeps_previous_results_and_reports_its_failure(tmp_path):
    path = tmp_path / "qa" / "verdict.json"
    rows = [{"capsule": "case_a", "status": "pass", "tiers": {"L2": "pass"}}]
    G.publish(path, _grade(rows))
    G.publish(path, {"model_gate": {"source_sha256": "s"}})
    failed = _grade([], grade_error={"error": "RuntimeError: snapshot changed", "source_sha256": "new"},
                    graded_at="t1")
    shown = G.publish(path, failed)
    on_disk = json.loads(path.read_text())
    assert shown == on_disk
    assert on_disk["per_capsule"] == rows and on_disk["model_gate"]["source_sha256"] == "s"
    assert on_disk["snapshot_current"] is False
    assert on_disk["grade_error_notice"]["error"].startswith("RuntimeError") and on_disk["grade_error_notice"]["graded_at"] == "t1"
    later = [{"capsule": "case_a", "status": "fail", "tiers": {"L2": "fail"}}]
    G.publish(path, _grade(later))
    final = json.loads(path.read_text())
    assert final["per_capsule"] == later and "grade_error_notice" not in final


def test_an_aborted_first_grade_is_still_published(tmp_path):
    path = tmp_path / "qa" / "verdict.json"
    G.publish(path, _grade([], grade_error={"error": "RuntimeError: x", "source_sha256": "s"}))
    assert json.loads(path.read_text())["grade_error"]["error"] == "RuntimeError: x"


def test_tests_never_share_the_repository_content_store():
    from merlin.common import content_store
    from merlin.common.paths import artifacts_dir

    selected = Path(os.environ["MERLIN_BUNDLE_CAS"]).resolve()
    assert content_store.store_root() == Path(os.environ["MERLIN_BUNDLE_CAS"])
    assert not selected.is_relative_to((artifacts_dir() / "cache").resolve())
