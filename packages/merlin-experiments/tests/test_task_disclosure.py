"""TASK.md sections reach the author on demand as skills; TASK.md keeps every byte."""

from __future__ import annotations

import json

import pytest
from merlin_experiments.phase1 import run_inputs as RI
from merlin_experiments.phase1 import task_disclosure as D

TASK = (
    "# Phase 1: a compiler task\n\nBuild the compiler in `submission/`.\n\n"
    "## Deliverable\n\nWrite a manifest.\n\n"
    "## Available tools\n\n- `python agent_selfcheck.py --capsules A,B`: focused checks with <angle> text.\n"
    + "  More detail about the tool and its options goes here.\n" * 30
    + "\n```\n## not a heading inside a fence\n```\n\n"
    "## Scheduling and pass exploration\n\nTry `pipeline_policy.evaluate_variants` on IR clones,\n"
    "and keep the winners. Second sentence.\n" + "Body text.\n" * 120
)


def _frontmatter(text: str) -> dict:
    head = text.split("---\n", 2)[1]
    fields = dict(line.split(": ", 1) for line in head.splitlines())
    return {"name": fields["name"], "description": json.loads(fields["description"])}


def test_sections_are_a_lossless_cut_at_headings_outside_fences():
    pieces = D.sections(TASK)
    assert "".join(piece["text"] for piece in pieces) == TASK
    assert [piece["heading"] for piece in pieces] == [
        "Phase 1: a compiler task", "Deliverable", "Available tools", "Scheduling and pass exploration"]
    assert pieces[2]["first_line"] == 9 and pieces[3]["first_line"] == pieces[2]["last_line"] + 1


def test_only_longer_sections_become_skills_with_verbatim_bodies_and_valid_frontmatter():
    skills = D.plan(TASK)
    assert set(skills) == {"task-available-tools", "task-scheduling-and-pass-exploration", D.FEEDBACK_SKILL}
    pieces = {piece["heading"]: piece["text"] for piece in D.sections(TASK)}
    for name, text in skills.items():
        meta = _frontmatter(text)
        assert meta["name"] == name and len(name) <= 64 and name == name.lower() and "--" not in name
        assert 0 < len(meta["description"]) <= 1024
        assert "<" not in meta["description"] and ">" not in meta["description"]
    assert skills["task-available-tools"].endswith(pieces["Available tools"])
    assert "lines 9-" in _frontmatter(skills["task-available-tools"])["description"]
    description = _frontmatter(skills["task-scheduling-and-pass-exploration"])["description"]
    assert "Try `pipeline_policy.evaluate_variants` on IR clones, and keep the winners." in description
    guide = skills[D.FEEDBACK_SKILL]
    for text in ("qa/latest_grade_summary.txt", "qa/latest_completed_grade.json", "failure_clusters",
                 "TUNING_KNOBS.json", "merlin.tuning_knobs.v1", "qa/idle_tuning.json", "variants/",
                 "suggested_mechanisms", "`task-scheduling-and-pass-exploration`"):
        assert text in guide


def test_staging_writes_records_and_seals_the_skills(tmp_path):
    ws, run, bundle = tmp_path / "ws", tmp_path / "run", tmp_path / "bundle"
    for path in (ws, run, bundle):
        path.mkdir()
    for path in (ws / "TASK.md", run / "TASK.md"):
        path.write_text(TASK)
    for path in (bundle / "input_bundle_manifest.yaml", run / "input_bundle_manifest.yaml"):
        path.write_text("allowed: []\n")
    before = RI.treatment_snapshot_record(ws, run, bundle, ["tool"])
    assert not any(row["name"].startswith("served/.agents") for row in before["files"])
    record = D.stage(ws, run)
    assert (ws / "TASK.md").read_text() == TASK  # TASK.md keeps every byte
    assert [row["name"] for row in record["skills"]] == list(D.plan(TASK))
    sealed = RI.treatment_snapshot_record(ws, run, bundle, ["tool"])
    names = {row["name"] for row in sealed["files"]}
    assert f"archived/{D.RECORD}" in names and "served/.agents/skills/task-available-tools/SKILL.md" in names
    RI.verify_treatment_snapshot(sealed, ws, run, bundle, ["tool"])
    (ws / ".agents/skills/task-available-tools/SKILL.md").write_text("edited")
    with pytest.raises(RuntimeError, match="drifted"):
        RI.verify_treatment_snapshot(sealed, ws, run, bundle, ["tool"])
    with pytest.raises(RuntimeError, match="already exist"):
        D.stage(ws, run)
