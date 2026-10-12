from pathlib import Path

import pytest

from merlin_experiments.phase1.brokers.selfcheck import _freeze_passed_source, _submission_digest


def test_passed_snapshot_survives_author_edit_after_done(tmp_path, monkeypatch):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    ws = tmp_path / "workspace"
    source = ws / "submission" / "compiler.py"
    source.parent.mkdir(parents=True)
    source.write_text("passed compiler source\n")
    graded = _submission_digest(source.parent)
    frozen = _freeze_passed_source(ws, graded)
    source.write_text("next compiler source\n")
    assert _submission_digest(frozen / "submission") == graded
    assert (frozen / "submission" / "compiler.py").read_text() == "passed compiler source\n"
    assert _submission_digest(source.parent) != graded


def test_source_change_before_freeze_refuses_promotion(tmp_path, monkeypatch):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    ws = tmp_path / "workspace"
    source = ws / "submission" / "compiler.py"
    source.parent.mkdir(parents=True)
    source.write_text("passed source\n")
    graded = _submission_digest(source.parent)
    source.write_text("ungraded source\n")
    with pytest.raises(RuntimeError, match="source moved while snapshotting"):
        _freeze_passed_source(ws, graded)
