"""The grading interpreter must be resolved, not inherited from whatever launched the run.

``sys.executable`` is whatever started the broker, and the broker is started by whatever started the run.
A batch driven through an orchestrator venv handed the grading child an interpreter that could not import
the submission's own dependencies, so every capsule died in its parse entrypoint with
``ModuleNotFoundError`` and was recorded as ``plane: parse, category: tool_crash`` -- i.e. charged to the
agent. Measured on radiance: 4561 promoted certs across three repeats, one cause, zero certifications,
while the same submission bytes graded clean under the repo's own venv.

Nothing here names a target or a real venv path: the resolver is exercised against temp files.
"""
from __future__ import annotations

import subprocess
import sys

from merlin.common.paths import merlin_dir

sys.path.insert(0, str(merlin_dir() / "experiments/capsule_bench/harness"))

import grading_env as GE  # noqa: E402


def test_an_explicit_override_wins(tmp_path, monkeypatch):
    fake = tmp_path / "python"
    fake.write_text("#!/bin/sh\n")
    monkeypatch.setenv("MERLIN_GRADE_PYTHON", str(fake))
    assert GE.grading_python() == str(fake)


def test_an_override_pointing_at_nothing_is_ignored_rather_than_returned(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_GRADE_PYTHON", str(tmp_path / "does_not_exist"))
    got = GE.grading_python()
    assert got != str(tmp_path / "does_not_exist")
    assert got, "a bad override must fall through to a real interpreter, never to an empty string"


def test_it_resolves_to_something_that_exists(monkeypatch):
    monkeypatch.delenv("MERLIN_GRADE_PYTHON", raising=False)
    from pathlib import Path
    assert Path(GE.grading_python()).is_file()


def test_the_repo_venv_is_preferred_over_the_launching_interpreter(monkeypatch):
    """The whole point: a run launched from another venv still grades under the repo's own."""
    monkeypatch.delenv("MERLIN_GRADE_PYTHON", raising=False)
    from merlin.common.paths import repo_root
    from pathlib import Path
    venv = Path(repo_root()) / ".venv" / "bin" / "python"
    if not venv.is_file():
        import pytest
        pytest.skip("no repo venv in this checkout; the fallback path is covered above")
    assert GE.grading_python() == str(venv)


def test_announce_returns_the_same_answer_as_the_resolver(capsys):
    assert GE.announce("test") == GE.grading_python()


def test_the_resolution_holds_when_called_from_a_foreign_interpreter(monkeypatch):
    """Called from an interpreter that is NOT the repo venv, the resolver must still pick the venv.

    Simulated in-process by asserting the resolver does not simply echo ``sys.executable`` whenever a
    repo venv exists — that echo was the defect.
    """
    monkeypatch.delenv("MERLIN_GRADE_PYTHON", raising=False)
    from merlin.common.paths import repo_root
    from pathlib import Path
    venv = Path(repo_root()) / ".venv" / "bin" / "python"
    if not venv.is_file():
        import pytest
        pytest.skip("no repo venv in this checkout")
    if Path(sys.executable).resolve() == venv.resolve():
        assert GE.grading_python() == str(venv)
    else:
        assert GE.grading_python() != sys.executable


def test_both_brokers_spawn_the_grader_with_the_resolved_interpreter():
    """A broker reaching for sys.executable directly is the bug coming back."""
    h = merlin_dir() / "experiments/capsule_bench/harness"
    sim = (h / "simjob_broker.py").read_text()
    sel = (h / "selfcheck_broker.py").read_text()
    assert "PY = _announce_py(" in sim, "simjob_broker must resolve the grading interpreter"
    assert "PY = sys.executable" not in sim, "the inherited-interpreter defect must not return"
    assert "[_grading_python(), str(SELFCHECK)" in sel, \
        "selfcheck_broker must spawn the grader with the resolved interpreter"
    assert "[sys.executable, str(SELFCHECK)" not in sel


def test_both_brokers_still_import():
    """They are spawned as scripts, so an import-time error is only found by importing them."""
    h = merlin_dir() / "experiments/capsule_bench/harness"
    for mod in ("simjob_broker", "selfcheck_broker"):
        r = subprocess.run([sys.executable, "-c", f"import sys; sys.path.insert(0, r'{h}'); import {mod}"],
                           capture_output=True, text=True, timeout=180)
        assert r.returncode == 0, f"{mod} failed to import: {r.stderr[-600:]}"
