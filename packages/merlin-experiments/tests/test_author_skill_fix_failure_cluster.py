"""The fix-failure-cluster skill reads the clusters the grader actually writes and drafts a usable brief."""

from __future__ import annotations

import json
import subprocess
import sys

from merlin_experiments.phase1 import skill_staging as SK
from merlin_experiments.phase1.feedback import completed_grade as CG
from merlin_experiments.phase1.feedback import failure_clusters as FC

SKILL = SK.SOURCE / "fix-failure-cluster"
SCRIPT = SKILL / "scripts" / "clusters.py"


def _grade(token, rows, current=False):
    per_capsule = [{"capsule": name, "status": status, "tiers": {"L2": status},
                    "failure_plane": None if status == "pass" else plane,
                    "failure_category": None if status == "pass" else "MISMATCH",
                    "failure_detail": None if status == "pass" else f"{name}: tile{index} differs at 0x{index}f",
                    "mismatch_count": 0 if status == "pass" else index}
                   for index, (name, status, plane) in enumerate(rows, 1)]
    return {"per_capsule": per_capsule, "graded_source_sha256": f"src-{token}", "graded_at": f"at-{token}",
            "current_source_sha256": f"src-{token}", "snapshot_current": current, "n_passed": 1,
            "n_capsules": len(rows)}


def _workspace(tmp_path):
    ws = tmp_path / "ws"
    (ws / "qa").mkdir(parents=True)
    rows = [("case_a", "pass", None), ("case_b", "fail", "numeric"), ("case_c", "fail", "numeric"),
            ("case_d", "fail", "emission")]
    CG.publish(ws, _grade(100, rows), 100, lambda path, verdict: verdict)
    return ws


def _run(ws, *args):
    done = subprocess.run([sys.executable, "-B", "-I", str(SCRIPT), "--workspace", str(ws), *args],
                          capture_output=True, text=True, check=False, timeout=60)
    assert done.returncode == 0, done.stderr
    return done.stdout


def test_the_script_reads_every_key_the_cluster_writer_emits(tmp_path):
    ws = _workspace(tmp_path)
    section = json.loads((ws / "qa" / CG.FILENAME).read_text())[FC.KEY]
    entry = section["clusters"][0]
    script = SCRIPT.read_text()
    for key in ("count", "failure_plane", "failure_category", "examples", "test_cases", "example_failure_detail",
                "reproduce", "inspect", "clusters_total", "failing_test_cases", "clusters_not_listed"):
        assert f'"{key}"' in script or f"'{key}'" in script
    assert {"count", "failure_plane", "failure_category", "test_cases", "reproduce"} <= set(entry)
    assert f'"{FC.KEY}"' in script and f'"{CG.FILENAME}"' in script and '"verdict.json"' in script


def test_list_full_and_brief_views(tmp_path):
    ws = _workspace(tmp_path)
    listing = _run(ws)
    assert f"source: qa/{CG.FILENAME}" in listing and "3 failing test case(s) in 2 cluster(s)" in listing
    assert "1. 2 case(s) numeric/MISMATCH [2/2 graded on the current source]" in listing
    assert "reproduce: python agent_selfcheck.py --submission submission --capsules case_b" in listing
    full = _run(ws, "--cluster", "1")
    assert "members: case_b, case_c" in full
    assert "all members: python agent_selfcheck.py --submission submission --capsules case_b,case_c" in full
    assert "case_c fail L2=fail 3 at-100" in full
    brief = _run(ws, "--brief", "1")
    for field in ("GOAL:", "FAILURE", "CASES: case_b, case_c", "REPRODUCE:", "ALL CASES:", "OWNED FILES:",
                  "SKILLS TO OPEN: fix-failure-cluster", "REPORT:"):
        assert field in brief
    assert json.loads(_run(ws, "--json"))["clusters_total"] == 2


def test_a_current_live_verdict_is_preferred_and_absence_is_explained(tmp_path):
    ws = _workspace(tmp_path)
    live = FC.attach(ws, _grade(300, [("case_e", "fail", "link")], current=True))
    (ws / "qa" / "verdict.json").write_text(json.dumps(live))
    assert "source: qa/verdict.json" in _run(ws) and "link/MISMATCH" in _run(ws)
    assert f"source: qa/{CG.FILENAME}" in _run(ws, "--file", "completed")
    empty = tmp_path / "empty"
    empty.mkdir()
    done = subprocess.run([sys.executable, "-B", "-I", str(SCRIPT), "--workspace", str(empty)], capture_output=True,
                          text=True, check=False)
    assert done.returncode == 1 and "no readable verdict" in done.stdout


def test_the_skill_names_only_commands_the_workspace_tools_offer():
    text = (SKILL / SK.SKILL_FILE).read_text()
    tool = (SK.SOURCE.parent / "tools" / "selfcheck.py").read_text()
    for flag in ("--attach", "--compile-only", "--submission", "--capsules failing", "--wait-budget"):
        assert flag in tool and flag in text
    assert f"qa/{CG.SUMMARY_FILENAME}" in text and f"qa/{CG.FILENAME}" in text and FC.KEY in text
    assert "await_verdict.py --verdict" in text and "--verdict" in SK.SOURCE.parent.joinpath(
        "tools", "await_verdict.py").read_text()
    assert 'fork_turns "none"' in text and "delegate-to-helpers" in text
