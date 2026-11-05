"""The full-model check child runs under the model2MLIR capture interpreter, which lacks ``aet``.

Both fresh10 final grades reported every full-model obligation as ``No module named 'aet'``: the
child classified its first kernel gate failure through an eager ``aet`` import, and the report named
the enclosing worktree's HEAD instead of the run's frozen export.
"""

import json
import subprocess
import sys

import pytest

from merlin.targetgen import capsule_common as CC
from merlin.targetgen.package_runtime import CertFailure


def _without_aet(monkeypatch):
    # A None entry makes the import raise ModuleNotFoundError, as an interpreter without aet does.
    for name in ("aet", "aet.core", "aet.core.failures"):
        monkeypatch.setitem(sys.modules, name, None)


def test_gate_failure_classification_needs_no_aet(monkeypatch):
    _without_aet(monkeypatch)
    failure = CertFailure("build", CC._cat("ELABORATION_ERROR"), "tool missing: x")
    assert failure.category == "elaboration_error" and failure.category.value == "elaboration_error"
    assert failure.detail == "tool missing: x"


def test_gate_failure_classification_keeps_the_aet_enum_when_present():
    failures = pytest.importorskip("aet.core.failures")
    assert CC._cat("TOOL_CRASH") is failures.FailureCategory.TOOL_CRASH
    assert CC._cat("NOT_A_MEMBER") is failures.FailureCategory.RUNNER_CRASH


def test_full_child_probe_exercises_failure_classification(monkeypatch):
    from merlin_experiments.phase1.brokers import model_check as MC

    calls = []
    monkeypatch.setattr(CC, "_cat", lambda name: calls.append(name) or name.lower())
    MC.import_probe()
    assert calls == ["TOOL_CRASH"]


def _git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def enclosing(tmp_path):
    """A live checkout with a frozen export directory inside it, as run-frameworks/<id> is."""
    repo = tmp_path / "worktree"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "f").write_text("x")
    _git(repo, "add", "f")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@e", "-c", "commit.gpgsign=false", "commit", "-qm", "c")
    export = repo / "out/build/run-frameworks/abc"
    export.mkdir(parents=True)
    return repo, export


def test_report_names_the_export_revision_not_the_enclosing_head(enclosing, monkeypatch):
    from merlin.model_check import bundle as B
    from merlin.model_check import report as R

    repo, export = enclosing
    revision = "a" * 40
    (export / ".merlin-source-revision.json").write_text(
        json.dumps({"schema": "merlin.source_revision.v1", "revision": revision}))
    monkeypatch.setenv("MERLIN_REPO_ROOT", str(export))
    assert R._tool_commit() == revision and B._git_commit(export) == revision
    (export / ".merlin-source-revision.json").unlink()
    # Without its own record an export is unknown, never the worktree's moving HEAD.
    assert R._tool_commit() is None and B._git_commit(export) is None
    monkeypatch.setenv("MERLIN_REPO_ROOT", str(repo))
    assert R._tool_commit() == _git(repo, "rev-parse", "HEAD")
