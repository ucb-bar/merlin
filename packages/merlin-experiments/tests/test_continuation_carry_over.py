"""Opt-in certificate carry-over across relaunches, and the previous-session handoff note."""
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from merlin.targetgen import tier_cache as TC
from merlin_experiments.phase1 import continuation_handoff as H, joint_control as J, joint_seed as S
from merlin_experiments.phase1.options import parse_options
from merlin_experiments.phase1.run_inputs import graded_source_hash


def _tree(root: Path) -> list[Path]:
    files = []
    for name, text in (("src/merlin/a.py", "a = 1\n"), ("src/merlin/b.py", "b = 2\n")):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        files.append(path)
    return files


def test_portable_instrument_names_files_relative_to_the_checkout(tmp_path, monkeypatch):
    first, second = _tree(tmp_path / "export-1"), _tree(tmp_path / "export-2")
    roots = iter([tmp_path / "export-1", tmp_path / "export-2", tmp_path / "export-1"])
    monkeypatch.setattr("merlin.common.paths.checkout_root", lambda: next(roots))
    one, two = TC._portable_source_digest(first), TC._portable_source_digest(second)
    assert one == two  # same bytes, different export directories: the same judge
    (tmp_path / "export-1/src/merlin/a.py").write_text("a = 3\n")
    assert TC._portable_source_digest(first) != two  # changed bytes: a different judge


def test_portable_mode_is_off_unless_set(monkeypatch):
    monkeypatch.delenv(TC.PORTABLE_ENV, raising=False)
    assert TC.portable() is False
    monkeypatch.setenv(TC.PORTABLE_ENV, "1")
    assert TC.portable() is True


@pytest.fixture
def seeded(tmp_path, monkeypatch):
    monkeypatch.delenv(J.ENV, raising=False)
    monkeypatch.delenv("MERLIN_JOINT_PERFORMANCE_SELECTION", raising=False)
    monkeypatch.delenv(TC.PORTABLE_ENV, raising=False)
    source = tmp_path / "checkpoint" / "submission"
    (source / "docs").mkdir(parents=True)
    (source / "manifest.yaml").write_text("version: 1\n")
    (source / "docs/iteration_notes.md").write_text("tried A\nnext: try resident RHS on dense large cases\n")
    prev_run, prev_ws = tmp_path / "prev-run", tmp_path / "prev-ws"
    (prev_run / "rounds").mkdir(parents=True)
    (prev_run / "rounds/round_00.turn00.final.txt").write_text(f"Report at {tmp_path}/secret/path: conv done\n")
    (prev_ws / "qa").mkdir(parents=True)
    (prev_ws / "qa/latest_completed_grade.json").write_text(json.dumps({
        "graded_source_sha256": "ab" * 32,
        "per_capsule": [{"capsule": "dense_big", "status": "fail", "tiers": {"L2": "fail"}, "failure_category": "NUMERIC"},
                        {"capsule": "conv_small", "status": "pass", "tiers": {"L2": "pass"}}],
        "feedback_brief": {"performance_priority": {"development_headroom": {"largest_headroom": [
            {"test_case": "dense_large", "cycles": 1000, "gap_ratio": 7.0}]}}}}))

    def build(**extra):
        review = {"schema": "operator.joint_seed_review.v1", "approved": True, "review_id": "r",
                  "source": str(source), "sha256": graded_source_hash(source),
                  "inherit_clock": False, "inherit_results": False, "variants": [], **extra}
        path = tmp_path / "review.json"
        path.write_text(json.dumps(review))
        origin = {k: v for k, v in review.items() if k not in ("schema", "approved")}
        origin.update(schema=S.SCHEMA, review={"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        policy = tmp_path / "policy.json"
        policy.write_text(json.dumps({"version": 1, "duration_s": 21600, "final_model_s": 3600, "drain_s": 3600,
                                      "compiler_origin": origin}))
        return replace(parse_options(["--run-id", "new"]), joint_campaign=str(policy), seed_submission=str(source),
                       fresh_backend=False, sandbox="bwrap", schedule="continuous"), origin

    return build, {"run_dir": str(prev_run), "workspace": str(prev_ws)}, tmp_path


def _stage(options, tmp_path):
    workspace, run = tmp_path / "ws", tmp_path / "run"
    workspace.mkdir(exist_ok=True)
    run.mkdir(exist_ok=True)
    return S.stage(options, workspace, run), workspace


def test_carry_over_is_off_by_default_and_recorded(seeded):
    build, _, tmp_path = seeded
    options, _ = build()
    result, _ = _stage(options, tmp_path)
    assert result["certificate_carry_over"]["enabled"] is False
    assert result["current_credit_granted"] is False


def test_carry_over_needs_both_review_and_launch_switch(seeded, monkeypatch):
    build, _, tmp_path = seeded
    options, _ = build(**{S.CARRY: True})
    with pytest.raises(ValueError, match="needs both"):
        _stage(options, tmp_path)  # approved, but the launch did not set the portable instrument
    monkeypatch.setenv(TC.PORTABLE_ENV, "1")
    result, _ = _stage(options, tmp_path)
    assert result["certificate_carry_over"]["enabled"] is True


def test_launch_switch_without_review_is_refused(seeded, monkeypatch):
    build, _, tmp_path = seeded
    options, _ = build()
    monkeypatch.setenv(TC.PORTABLE_ENV, "1")
    with pytest.raises(ValueError, match="needs both"):
        _stage(options, tmp_path)


def test_review_must_approve_the_exact_additions(seeded):
    build, previous, _ = seeded
    options, origin = build(**{S.CARRY: True})
    policy = Path(options.joint_campaign)
    value = json.loads(policy.read_text())
    value["compiler_origin"][S.PREVIOUS] = previous  # added after review
    policy.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="does not approve"):
        J.validate(options)


def test_previous_session_note_is_bounded_scrubbed_and_read_only(seeded):
    build, previous, tmp_path = seeded
    options, _ = build(**{S.PREVIOUS: previous})
    result, workspace = _stage(options, tmp_path)
    note = workspace / H.FILENAME
    text = note.read_text()
    assert result["previous_session"]["inputs_found"] == {"turn_report": True, "iteration_notes": True, "grade": True}
    assert "conv done" in text and "resident RHS" in text and "`dense_big`: NUMERIC" in text
    assert "`dense_large`: 1000 cycles" in text and str(tmp_path) not in text
    assert not os.access(note, os.W_OK) or os.geteuid() == 0
    assert len(text) < 4 * H.SECTION_CHARS


def test_previous_session_missing_inputs_say_not_recorded(tmp_path):
    text = H.render(tmp_path / "none-run", tmp_path / "none-ws", tmp_path / "none-sub")
    assert text.count("not recorded") >= 3
