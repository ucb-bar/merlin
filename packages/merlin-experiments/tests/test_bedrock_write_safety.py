"""Host write-tool checks only; no provider or kernel sandbox is launched."""

import pytest
from merlin_experiments.phase1.providers.bedrock_agent import _write_file


@pytest.mark.parametrize("relative", ["note.txt", "submission/compiler.py"])
@pytest.mark.parametrize("mode", [None, "", "0"])
def test_ordinary_write(tmp_path, monkeypatch, relative, mode):
    if mode is None:
        monkeypatch.delenv("MERLIN_PINNED_SUBMISSION_READ_ONLY", raising=False)
    else:
        monkeypatch.setenv("MERLIN_PINNED_SUBMISSION_READ_ONLY", mode)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    assert _write_file(workspace, relative, "content").startswith("wrote")
    assert (workspace / relative).read_text() == "content"


@pytest.mark.parametrize("route", ["sibling", "absolute", "parent", "symlink"])
def test_resolved_escape_does_not_mutate(tmp_path, monkeypatch, route):
    monkeypatch.delenv("MERLIN_PINNED_SUBMISSION_READ_ONLY", raising=False)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "workspace-sibling"
    outside.mkdir()
    victim = outside / "existing.txt"
    victim.write_text("original")
    if route == "symlink":
        (workspace / "alias").symlink_to(outside, target_is_directory=True)
        relative = "alias/existing.txt"
    elif route == "absolute":
        relative = str(victim)
    elif route == "parent":
        relative = "../new-parent/new.txt"
    else:
        relative = "../workspace-sibling/existing.txt"
    assert _write_file(workspace, relative, "mutated").startswith("[refused]")
    assert victim.read_text() == "original"
    assert not (tmp_path / "new-parent").exists()


@pytest.mark.parametrize("relative", ["new/note.txt", "submission/compiler.py"])
def test_retired_readonly_request_refuses_all_writes(tmp_path, monkeypatch, relative):
    monkeypatch.setenv("MERLIN_PINNED_SUBMISSION_READ_ONLY", " 1 ")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    assert "unsupported" in _write_file(workspace, relative, "content")
    assert list(workspace.iterdir()) == []
    existing = workspace / relative
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_text("original")
    assert "unsupported" in _write_file(workspace, relative, "mutated")
    assert existing.read_text() == "original"
