"""The feedback-files skill names only files the harness writes, and its status script reads them."""

from __future__ import annotations

import subprocess
import sys

from merlin_experiments.phase1 import skill_staging as SK
from merlin_experiments.phase1 import task_disclosure
from merlin_experiments.phase1.feedback import completed_grade as CG
from merlin_experiments.phase1.feedback import failure_clusters as FC
from merlin_experiments.phase1.feedback import idle_tuning

SKILL = SK.SOURCE / SK.FEEDBACK_SKILL
SCRIPT = SKILL / "scripts" / "qa_status.py"


def _quoted_paths(text: str, prefix: str) -> set[str]:
    """Backticked tokens of a Markdown text that start with ``prefix``."""
    pieces = text.split("`")
    return {piece for piece in pieces[1::2] if piece.startswith(prefix)}


def _writer_sources() -> str:
    phase1 = SK.SOURCE.parent
    return "\n".join(path.read_text(encoding="utf-8") for path in phase1.rglob("*.py")
                     if SK.SOURCE not in path.parents)


def test_every_named_feedback_file_is_written_by_the_harness():
    text = (SKILL / SK.SKILL_FILE).read_text()
    named = _quoted_paths(text, "qa/")
    assert {f"qa/{CG.FILENAME}", f"qa/{CG.SUMMARY_FILENAME}", "qa/verdict.json", "qa/progress.json",
            "qa/tier_state.json"} <= named
    sources = _writer_sources()
    for path in named:
        name = path.split("/", 1)[1]
        assert f'"{name}"' in sources, f"{path} is not a file any phase1 writer names"
    assert FC.KEY in text and "selfcheck_out/NAME/" in text and '"selfcheck_out"' in sources
    assert "--verdict qa/latest_completed_grade.json" in text


def test_the_guide_replaces_the_generated_one_and_points_performance_elsewhere():
    text = (SKILL / SK.SKILL_FILE).read_text()
    generated = task_disclosure.feedback_skill([])
    # What the generated guide offered is still reachable from here (directly or via the named skills).
    for item in (f"qa/{CG.SUMMARY_FILENAME}", f"qa/{CG.FILENAME}", FC.KEY, "feedback_brief", "performance_brief",
                 "TUNING_KNOBS.json"):
        assert item in generated and item in text
    assert str(idle_tuning.LEDGER) in generated and "idle-time timing" in text
    for name in ("performance-feedback-levels", "roofline-and-utilization", "cycle-model-and-gsim"):
        assert f"`{name}`" in text


def test_status_script_reports_age_size_and_freshness_fields(tmp_path):
    ws = tmp_path / "ws"
    (ws / "qa").mkdir(parents=True)
    grade = {"per_capsule": [{"capsule": "case_a", "status": "fail", "tiers": {"L2": "fail"},
                              "failure_plane": "numeric", "failure_category": "MISMATCH", "failure_detail": "x"}],
             "graded_source_sha256": "a" * 64, "current_source_sha256": "b" * 64, "graded_at": "at-1",
             "snapshot_current": False, "n_passed": 0, "n_capsules": 1}
    CG.publish(ws, grade, 100, lambda path, verdict: verdict)
    done = subprocess.run([sys.executable, "-B", "-I", str(SCRIPT), "--workspace", str(ws)], capture_output=True,
                          text=True, check=False, timeout=60)
    assert done.returncode == 0, done.stderr
    out = done.stdout
    assert f"qa/{CG.FILENAME}  age=" in out and f"qa/{CG.SUMMARY_FILENAME}  age=" in out
    assert "graded_source_sha256=aaaaaaaaaaaaaaaa current_source_sha256=bbbbbbbbbbbbbbbb" in out
    assert "completed_rows: rows=1 executed=1 passed=0" in out and "failure_clusters: failing_test_cases=1" in out
    empty = subprocess.run([sys.executable, "-B", "-I", str(SCRIPT), "--workspace", str(tmp_path)],
                           capture_output=True, text=True, check=False)
    assert empty.returncode == 1 and "no qa/ directory" in empty.stdout
