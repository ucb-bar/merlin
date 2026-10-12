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


def test_direct_job_and_promotion_keep_same_source_after_author_edit(tmp_path, monkeypatch):
    from merlin_experiments.phase1.brokers.simjob import _freeze_direct_job_source

    monkeypatch.setenv("TMPDIR", str(tmp_path))
    ws = tmp_path / "workspace"
    source = ws / "submission" / "compiler.py"
    source.parent.mkdir(parents=True)
    source.write_text("graded source\n")
    submission, frozen, error = _freeze_direct_job_source(ws)
    assert error is None
    assert submission == frozen / "submission"
    source.write_text("later author edit\n")
    assert (submission / "compiler.py").read_text() == "graded source\n"


def test_direct_job_refuses_source_movement_while_freezing(tmp_path, monkeypatch):
    from merlin_experiments.phase1.brokers import simjob
    from merlin_experiments.phase1.feedback import snapshots

    monkeypatch.setenv("TMPDIR", str(tmp_path))
    ws = tmp_path / "workspace"
    source = ws / "submission" / "compiler.py"
    source.parent.mkdir(parents=True)
    source.write_text("ungraded source\n")
    original = snapshots.create_promotion_snapshot

    def mutate(*args, **kwargs):
        source.write_text("changed during freeze\n")
        return original(*args, **kwargs)

    monkeypatch.setattr(snapshots, "create_promotion_snapshot", mutate)
    submission, frozen, error = simjob._freeze_direct_job_source(ws)
    assert submission is None and frozen is None
    assert "source moved while snapshotting" in error
