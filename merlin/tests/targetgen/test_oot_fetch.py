"""Hermetic tests for the native OOT-repo fetch (``merlin.targetgen.oot_fetch``).

Offline: a ``file://`` git repo is built in a tmp dir as the stand-in ``<target>-mlir`` repo, so no
network is touched. The target name is synthetic (``fixturenpu``) — the mechanism is target-agnostic.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from merlin.targetgen import oot_fetch


def _git(args, cwd):
    subprocess.run(["git", *args], cwd=str(cwd), check=True, capture_output=True, text=True)


def _make_target_repo(root: Path, *, name: str = "fixturenpu", with_contract: bool = True) -> Path:
    """A minimal bare-ish git repo that looks like a published <target>-mlir package."""
    root.mkdir(parents=True, exist_ok=True)
    _git(["init", "-q", "-b", "main"], root)
    _git(["config", "user.email", "t@t"], root)
    _git(["config", "user.name", "t"], root)
    if with_contract:
        (root / "contracts").mkdir()
        (root / "contracts" / "target_contract.yaml").write_text(
            f"name: {name}\nfamily: tensor_resident\nplugin:\n  backend: backend\n", encoding="utf-8")
    else:
        (root / "README.md").write_text("no contract here\n", encoding="utf-8")
    _git(["add", "-A"], root)
    _git(["commit", "-q", "-m", "init"], root)
    return root


def test_repo_url_template_and_override(monkeypatch):
    monkeypatch.delenv("MERLIN_TARGET_REPO_TEMPLATE", raising=False)
    monkeypatch.delenv("MERLIN_TARGET_REPO_FOO", raising=False)
    # default template
    assert oot_fetch.repo_url("foo") == "https://github.com/ucb-bar/foo-mlir.git"
    # template override
    monkeypatch.setenv("MERLIN_TARGET_REPO_TEMPLATE", "git@host:org/{target}.git")
    assert oot_fetch.repo_url("foo") == "git@host:org/foo.git"
    # per-target exact override wins over the template
    monkeypatch.setenv("MERLIN_TARGET_REPO_FOO", "file:///somewhere/foo.git")
    assert oot_fetch.repo_url("foo") == "file:///somewhere/foo.git"


def test_repo_url_template_missing_placeholder_raises(monkeypatch):
    monkeypatch.setenv("MERLIN_TARGET_REPO_TEMPLATE", "https://host/fixed.git")
    with pytest.raises(oot_fetch.FetchError):
        oot_fetch.repo_url("foo")


def test_fetch_from_file_url(tmp_path):
    src = _make_target_repo(tmp_path / "src")
    dest = tmp_path / "home" / "fixturenpu"
    root = oot_fetch.fetch("fixturenpu", url=f"file://{src}", dest=dest)
    assert root == dest
    assert (root / "contracts" / "target_contract.yaml").is_file()
    # re-fetch (update path) is idempotent
    root2 = oot_fetch.fetch("fixturenpu", url=f"file://{src}", dest=dest)
    assert root2 == dest


def test_fetch_rejects_non_target_repo(tmp_path):
    src = _make_target_repo(tmp_path / "src", with_contract=False)
    dest = tmp_path / "home" / "x"
    with pytest.raises(oot_fetch.FetchError):
        oot_fetch.fetch("x", url=f"file://{src}", dest=dest)


def test_fetch_into_generated_home(tmp_path, monkeypatch):
    """With MERLIN_OUT_ROOT set, fetch drops the package into the generated-target home so the
    registry resolves it with zero env."""
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    src = _make_target_repo(tmp_path / "src", name="fixturenpu")
    root = oot_fetch.fetch("fixturenpu", url=f"file://{src}")
    assert root.name == "fixturenpu"
    assert str(tmp_path / "out") in str(root)
