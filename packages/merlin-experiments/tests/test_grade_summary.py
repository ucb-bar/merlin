"""Every completed grade is readable as one short line per test case beside the full file."""

from __future__ import annotations

import json

from merlin_experiments.phase1.feedback import completed_grade as CG
from merlin_experiments.phase1.feedback import grade_summary
from merlin_experiments.phase1.feedback.capability_brief import render as render_brief


def _grade(token, rows):
    per_capsule = [{"capsule": name, "status": status, "tiers": {"L2": status, "L3": "unavailable"},
                    "failure_plane": None if status == "pass" else "emission",
                    "failure_category": None if status == "pass" else "LINK", "failure_detail": "undefined symbol",
                    **({"performance": {"status": "estimated", "estimated_whole_soc_cycles": 1234,
                                        "feedback_brief": {"bulk": "x" * 5000}}}
                       if status == "pass" else {})} for name, status in rows]
    return {"per_capsule": per_capsule, "graded_source_sha256": f"src-{token}", "graded_at": f"at-{token}",
            "snapshot_current": False, "n_passed": 1, "n_capsules": len(rows)}


def test_every_completed_grade_rewrites_the_one_line_per_case_view(tmp_path):
    ws = tmp_path / "ws"
    (ws / "qa").mkdir(parents=True)
    rows = [("case_a", "pass"), ("case_b", "fail"), ("case_c", "fail")]
    CG.publish(ws, _grade(100, rows), 100, lambda path, verdict: verdict)
    CG.publish(ws, _grade(200, [("case_a", "pass")]), 200, lambda path, verdict: verdict)
    full = json.loads((ws / "qa" / CG.FILENAME).read_text())
    assert full[CG.POINTER]["summary"] == f"qa/{grade_summary.FILENAME}"
    text = (ws / "qa" / grade_summary.FILENAME).read_text()
    assert text == grade_summary.render(full)
    lines = text.splitlines()
    assert "case_a pass L2=pass,L3=unavailable - perf=estimated cycles=1234" in lines
    assert "case_b fail L2=fail,L3=unavailable emission/LINK graded=100" in lines  # kept from the older grade
    assert any(line.startswith("1. 2 case(s) emission/LINK") for line in lines)
    assert any(line.strip() == "members: case_b, case_c" for line in lines)
    assert "newest grade token=200" in text and "passed=1" in text
    assert len(text) < 2000 and len(json.dumps(full)) > 5000  # the per-case bulk stays in the full file


def test_a_failing_view_never_touches_the_full_file(tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    (ws / "qa").mkdir(parents=True)
    monkeypatch.setattr(grade_summary, "render", lambda document: (_ for _ in ()).throw(KeyError("bad")))
    CG.publish(ws, _grade(100, [("case_a", "pass")]), 100, lambda path, verdict: verdict)
    assert json.loads((ws / "qa" / CG.FILENAME).read_text())["per_capsule"]
    assert not (ws / "qa" / grade_summary.FILENAME).exists()


def test_the_joint_brief_points_every_arm_at_the_view_and_at_idle_tuning():
    inventory = {"members": {"case_a": {"cohort": "scalability_performance", "family": "f"}}}
    text = render_brief(inventory, {"per_capsule": [{"capsule": "case_a", "status": "fail"}]})
    assert f"qa/{grade_summary.FILENAME}" in text
    assert "submission/TUNING_KNOBS.json" in text and "qa/idle_tuning.json" in text and "variants/<name>/" in text


def test_joint_brief_refresh_preserves_instructions_and_discloses_mixed_sources(tmp_path):
    ws = tmp_path / "ws"
    (ws / "qa").mkdir(parents=True)
    inventory = {"schema": "merlin.public_capability_inventory.v1", "members": {
        "case_a": {"cohort": "original_correctness", "family": "f"},
        "case_b": {"cohort": "scalability_performance", "family": "f"}}}
    (ws / "PUBLIC_CAPABILITIES.json").write_text(json.dumps(inventory))
    brief = ws / "qa/round_brief.md"
    prefix = "## Operator instructions\nKeep selected numerical policy.\n\n"
    brief.write_text(prefix + "# Joint compiler progress\nstale\n## Compiler approach ledger\nnotes\n")
    CG.publish(ws, _grade(100, [("case_a", "pass"), ("case_b", "fail")]), 100, lambda p, v: v)
    CG.publish(ws, _grade(200, [("case_a", "pass")]), 200, lambda p, v: v)
    text = brief.read_text()
    assert text.startswith(prefix) and "stale" not in text
    assert "Merged rows can describe different sources" in text
    assert "not automatically current-source qualification" in text
    assert "Completed-grade token: `200`" in text
    assert "1 / 1 | 0" in text  # older failure remains visible; no invented combined pass
    assert "Compiler approach ledger\nnotes" in text
    from merlin_experiments.phase1.feedback.capability_brief import build_if_selected
    resumed = build_if_selected(tmp_path / "run", ws)
    assert "Compiler snapshot: `src-200`" in resumed
    assert "different sources" in resumed
    assert "Passing (locked)" not in text
    before = brief.read_bytes()
    assert not CG.record(ws, _grade(150, [("case_a", "pass")]), 150)
    assert brief.read_bytes() == before


def test_joint_brief_failure_cannot_discard_durable_grade(tmp_path, monkeypatch):
    from merlin_experiments.phase1.feedback import capability_brief
    ws = tmp_path / "ws"
    monkeypatch.setattr(capability_brief, "refresh_completed", lambda *_: (_ for _ in ()).throw(OSError("busy")))
    CG.publish(ws, _grade(100, [("case_a", "pass")]), 100, lambda p, v: v)
    assert json.loads((ws / "qa" / CG.FILENAME).read_text())[CG.POINTER]["token"] == "100"


def test_passing_behavior_is_preserved_without_locking_out_performance_work():
    from merlin_experiments.phase1.feedback.brief import _focus_section
    text = "\n".join(_focus_section(_grade(100, [("case_a", "pass"), ("case_b", "fail")])))
    assert "Do NOT re-work" not in text and "Passing (locked)" not in text
    assert "performance alongside correctness repair" in text


def test_a_passing_case_lacking_trace_evidence_is_shown_while_the_author_can_act():
    rows = [
        {"capsule": "looped", "status": "pass", "tiers": {"L2": "pass"}, "trace_status": "fail",
         "trace_requirement_status": "failed",
         "trace_violations": ["mode resident_reuse declared but <2 output commits (no reuse visible)"]},
        {"capsule": "unresolved", "status": "pass", "tiers": {"L2": "pass"}, "trace_status": "pass",
         "trace_requirement_status": "unavailable_fields", "trace_unavailable_requirements": ["i8"]},
        {"capsule": "clean", "status": "pass", "tiers": {"L2": "pass"}, "trace_status": "pass",
         "trace_requirement_status": "pass"},
        {"capsule": "model", "status": "pass", "tiers": {"L2": "pass"}},
    ]
    lines = grade_summary.render({"per_capsule": rows}).splitlines()
    assert ("looped pass L2=pass - trace=failed: mode resident_reuse declared but <2 output commits "
            "(no reuse visible)") in lines
    assert "unresolved pass L2=pass - trace=unavailable_fields: evidence unavailable for i8" in lines
    assert "clean pass L2=pass -" in lines and "model pass L2=pass -" in lines
    header = next(line for line in lines if line.startswith("## trace evidence"))
    assert header.startswith("## trace evidence: 2 passing case(s)")
    assert "- looped: failed: mode resident_reuse declared" in "\n".join(lines)
