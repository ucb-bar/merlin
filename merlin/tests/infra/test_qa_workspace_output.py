"""Generated phase-1 workspace placement, old-run discovery, and sandbox negative controls."""

from __future__ import annotations

import importlib
import inspect
import subprocess

import pytest
import yaml

from merlin.common import storage_lifecycle as lifecycle
from merlin.common.paths import repo_root
from merlin.targetgen.sandbox import bwrap as BW
from merlin.targetgen.sandbox.preflight import probe_sandbox


@pytest.fixture
def workspaces(monkeypatch, tmp_path):
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "output"))
    return importlib.import_module("merlin_experiments.phase1.workspaces")


def test_new_workspace_respects_output_root_and_never_reuses_legacy(workspaces, tmp_path):
    experiment = tmp_path / "source" / "target"
    legacy = experiment / "_qa_ws" / "run-1" / "workspace"
    legacy.mkdir(parents=True)
    marker = legacy / "candidate.txt"
    marker.write_text("historical candidate")
    root = workspaces.select_workspace_root(
        target="test-target", arm="raw_baseline", run_dir=tmp_path / "run-1", experiment=experiment, resume=False
    )
    assert root == tmp_path / "output/build/agent-workspaces/phase1/test-target/raw_baseline/run-1"
    assert not root.exists()
    assert marker.read_text() == "historical candidate"


def test_resume_prefers_recorded_path_then_generated_then_legacy(workspaces, tmp_path):
    run = tmp_path / "run-1"
    run.mkdir()
    experiment = tmp_path / "source" / "target"
    legacy = experiment / "_qa_ws/run-1/workspace"
    legacy.mkdir(parents=True)
    generated = workspaces.workspace_parent("test-target", "arm") / "run-1/workspace"
    generated.mkdir(parents=True)
    archived = tmp_path / "old-output" / "run-1" / "workspace"
    archived.mkdir(parents=True)
    (run / "environment.yaml").write_text(yaml.safe_dump({"workspace_path": str(archived)}))
    kwargs = dict(target="test-target", arm="arm", run_dir=run, experiment=experiment, resume=True)
    assert workspaces.select_workspace_root(**kwargs) == archived.parent
    archived.rmdir()
    assert workspaces.select_workspace_root(**kwargs) == generated.parent
    generated.rmdir()
    assert workspaces.select_workspace_root(**kwargs) == legacy.parent
    assert str(archived) in (run / "environment.yaml").read_text()


@pytest.mark.parametrize("part", ["", "..", "/outside", "target/child", "target*", "target\\child"])
def test_workspace_components_cannot_escape_or_glob(workspaces, part):
    with pytest.raises(ValueError, match="workspace path component"):
        workspaces.workspace_parent(part, "arm")


def test_workspace_session_keeps_active_and_uncertain_workers_protected(workspaces, tmp_path, capsys):
    workspace = tmp_path / "workspace"

    @workspaces.workspace_session
    def orderly(*, _workspace_leases):
        _workspace_leases.append(lifecycle.acquire(workspace, owner="fixture"))
        assert lifecycle.blockers(workspace)
        return 0

    assert orderly() == 0
    assert not lifecycle.blockers(workspace, require_terminal=True)

    @workspaces.workspace_session
    def interrupted(*, _workspace_leases):
        _workspace_leases.append(lifecycle.acquire(workspace, owner="fixture"))
        raise RuntimeError("child shutdown is uncertain")

    with pytest.raises(RuntimeError, match="shutdown is uncertain"):
        interrupted()
    assert any("running" in reason for reason in lifecycle.blockers(workspace))
    assert "acknowledge_abandoned" in capsys.readouterr().err
    assert "def interrupted" in inspect.getsource(interrupted)


def test_evidence_snapshot_uses_recorded_workspace_without_codex_cache(workspaces, tmp_path):
    timing = importlib.import_module("merlin_experiments.phase1.telemetry.evidence")
    run = tmp_path / "relocated-runs" / "run-1"
    run.mkdir(parents=True)
    workspace = workspaces.workspace_parent("test-target", "arm") / "run-1/workspace"
    evidence = workspace / ".qa_channel" / "reply.json"
    evidence.parent.mkdir(parents=True)
    evidence.write_text('{"status":"pass"}')
    (run / "environment.yaml").write_text(yaml.safe_dump({"workspace_path": str(workspace)}))
    copied = timing.snapshot_agent_evidence(run)
    assert copied == [run / "agent_evidence_snapshot/.qa_channel/reply.json"]
    assert copied[0].read_bytes() == evidence.read_bytes()


@pytest.fixture
def masked_output(tmp_path):
    repo = tmp_path / "repo"
    workspace = repo / "out/build/agent-workspaces/phase1/test-target/arm/run-1/workspace"
    workspace.mkdir(parents=True)
    (workspace / "candidate.txt").write_text("candidate")
    private = repo / "out/runs/other/answer.txt"
    private.parent.mkdir(parents=True)
    private.write_text("private answer")
    sibling = workspace.parent.parent / "run-other" / "workspace" / "candidate.txt"
    sibling.parent.mkdir(parents=True)
    sibling.write_text("other candidate")
    bundle = {"denied": [{"path": "out"}]}
    return repo, workspace, private, sibling, bundle


def test_output_mask_then_workspace_rebind_does_not_expose_siblings(masked_output):
    repo, workspace, private, sibling, bundle = masked_output
    argv = BW.base_argv(workspace, bundle, repo=repo)
    for policy in (argv, BW.reapply_bundle_snapshot(argv, workspace, bundle, repo=repo)):
        assert policy[-3:] == ["--bind", str(workspace), str(workspace)]
        assert BW.is_exposed(policy, workspace / "candidate.txt")
        assert not BW.is_exposed(policy, private)
        assert not BW.is_exposed(policy, sibling)
    # Negative control: without the explicit bind the output mask hides the authoring workspace.
    assert not BW.is_exposed(argv[:-3], workspace / "candidate.txt")


def test_generated_workspace_documentation_is_ignored():
    root = repo_root()
    for name in ("README.md", "AGENT.md", "submission.py"):
        path = f"out/build/agent-workspaces/phase1/target/arm/run/workspace/{name}"
        result = subprocess.run(["git", "check-ignore", "--no-index", path], cwd=root, capture_output=True)
        assert result.returncode == 0, path
    control = subprocess.run(["git", "check-ignore", "--no-index", "out/README.md"], cwd=root, capture_output=True)
    assert control.returncode == 1, "output skeleton documentation must remain trackable"


def test_live_output_workspace_can_chdir_without_reading_masked_siblings(masked_output):
    probe = probe_sandbox()
    if not probe.usable:
        pytest.skip(probe.describe())
    repo, workspace, private, sibling, bundle = masked_output
    policy = BW.reapply_bundle_snapshot(BW.base_argv(workspace, bundle, repo=repo), workspace, bundle, repo=repo)
    script = "from pathlib import Path; import sys; assert Path('candidate.txt').read_text() == 'candidate'; "
    script += "assert all(not Path(p).exists() for p in sys.argv[1:]); Path('authored.txt').write_text('ok')"
    result = subprocess.run(
        [*policy, "/usr/bin/python3", "-c", script, str(private), str(sibling)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert (workspace / "authored.txt").read_text() == "ok"
    blocked = subprocess.run(
        [*policy, "--tmpfs", str(repo / "out"), "/usr/bin/true"], capture_output=True, text=True, timeout=30
    )
    assert blocked.returncode != 0, "negative control unexpectedly retained a masked workspace"
