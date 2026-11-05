import json
from pathlib import Path

import pytest
from merlin_experiments.phase1 import capsule_timing_archive as archive
from merlin_experiments.phase1.feedback import raw_selfcheck_evidence as E


def test_retained_selfcheck_is_discovered_by_existing_timing_audit(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    run = tmp_path / "run"
    root = run / "_qa_work" / "selfchecks"
    env = E.broker_environment(root, workspace, environ={"OTHER": "kept"})
    first = E.create_runs_root(environ=env)
    second = E.create_runs_root(environ=env)
    assert env["OTHER"] == "kept"
    assert first != second and first.parent == second.parent == root
    assert first.stat().st_mode & 0o777 == 0o700
    capsule = first / "runs" / "fixture-suite" / "fixture-capsule"
    capsule.mkdir(parents=True)
    (capsule / "capsule_result.json").write_text(json.dumps({
        "capsule": "fixture-capsule", "status": "fail", "tiers": {},
    }))
    report = archive.audit(run)
    assert report["capsule_records"] == 1
    assert report["records"][0]["result"].startswith("_qa_work/selfchecks/selfcheck_")
    assert report["timing_ready"] is False
    assert not list(workspace.rglob("capsule_result.json"))


@pytest.mark.parametrize("selection", ["inside", "same", "ancestor", "relative", "symlink", "parent_symlink"])
def test_refuses_visible_and_noncanonical_raw_evidence_roots(tmp_path, selection):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    host = tmp_path / "host"
    host.mkdir()
    if selection == "inside":
        root = workspace / "raw"
    elif selection == "same":
        root = workspace
    elif selection == "ancestor":
        root = tmp_path
    elif selection == "relative":
        root = Path("relative-root")
    else:
        link = tmp_path / "linked"
        link.symlink_to(host, target_is_directory=True)
        root = link if selection == "symlink" else link / "raw"
    with pytest.raises(ValueError, match="canonical|disjoint"):
        E.broker_environment(root, workspace, environ={})


@pytest.mark.parametrize("key", [E.ROOT_ENV, E.WORKSPACE_ENV])
def test_incomplete_selection_refuses_instead_of_falling_back(tmp_path, key):
    with pytest.raises(ValueError):
        E.create_runs_root(environ={key: str(tmp_path)})


def test_selection_is_rechecked_in_worker_before_creating_raw_files(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    root = tmp_path / "host" / "raw"
    env = E.broker_environment(root, workspace, environ={})
    root.parent.mkdir()
    root.symlink_to(workspace, target_is_directory=True)
    with pytest.raises(ValueError, match="canonical"):
        E.create_runs_root(environ=env)
    assert not list(workspace.iterdir())


def test_unselected_broker_clears_ambient_selection_and_keeps_legacy_tmpdir(tmp_path, monkeypatch):
    monkeypatch.setattr(E.tempfile, "tempdir", str(tmp_path))
    env = E.broker_environment(None, tmp_path, environ={
        E.ROOT_ENV: "/old/run", E.WORKSPACE_ENV: "/old/workspace", "OTHER": "kept",
    })
    assert env == {"OTHER": "kept"}
    assert E.create_runs_root(environ=env).parent == tmp_path
