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
    return subprocess.run(
        ["git", "-c", "commit.gpgSign=false", "-c", "tag.gpgSign=false", *args],
        cwd=str(cwd),
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()


def _make_target_repo(root: Path, *, name: str = "fixturenpu", with_contract: bool = True) -> Path:
    """A minimal bare-ish git repo that looks like a published <target>-mlir package."""
    root.mkdir(parents=True, exist_ok=True)
    _git(["init", "-q", "-b", "main"], root)
    _git(["config", "user.email", "t@t"], root)
    _git(["config", "user.name", "t"], root)
    if with_contract:
        (root / "contracts").mkdir()
        (root / "contracts" / "target_contract.yaml").write_text(
            f"name: {name}\nfamily: tensor_resident\nplugin:\n  backend: backend\n", encoding="utf-8"
        )
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


def _commit(root, name, contents):
    (root / name).write_text(contents)
    _git(["add", name], root)
    _git(["commit", "-q", "-m", "fixture update"], root)
    return _git(["rev-parse", "HEAD"], root)


def test_repeated_published_tag_fetch(tmp_path):
    src = _make_target_repo(tmp_path / "src")
    _git(["tag", "v1-fixture"], src)
    dest = tmp_path / "dest"
    expected = _git(["rev-parse", "HEAD"], src)
    for _ in range(3):
        oot_fetch.fetch("fixturenpu", url=src.as_uri(), dest=dest, champion="v1-fixture")
        assert _git(["rev-parse", "HEAD"], dest) == expected


def test_switch_single_branch_clone_and_follow_advanced_branch(tmp_path):
    src = _make_target_repo(tmp_path / "src")
    base = _git(["rev-parse", "HEAD"], src)
    _git(["checkout", "-qb", "stable/first"], src)
    first = _commit(src, "compiler.txt", "first")
    _git(["checkout", "-qb", "stable/second", base], src)
    second = _commit(src, "compiler.txt", "second")
    dest = tmp_path / "dest"
    for branch, expected in (("stable/first", first), ("stable/second", second), ("stable/first", first)):
        oot_fetch.fetch("fixturenpu", url=src.as_uri(), dest=dest, champion=branch)
        assert _git(["rev-parse", "HEAD"], dest) == expected
    _git(["checkout", "-q", "stable/first"], src)
    advanced = _commit(src, "compiler.txt", "advanced")
    oot_fetch.fetch("fixturenpu", url=src.as_uri(), dest=dest, champion="stable/first")
    assert _git(["rev-parse", "HEAD"], dest) == advanced
    # The old local branch is retained, not reset to the newly fetched commit.
    assert _git(["rev-parse", "refs/heads/stable/first"], dest) == first


@pytest.mark.parametrize("change", ["tracked", "staged", "untracked", "commit", "tagged_commit"])
def test_local_work_is_preserved(tmp_path, change):
    src = _make_target_repo(tmp_path / "src")
    dest = oot_fetch.fetch("fixturenpu", url=src.as_uri(), dest=tmp_path / "dest")
    _git(["config", "user.email", "fixture@local"], dest)
    _git(["config", "user.name", "Fixture"], dest)
    path = dest / ("local.txt" if change == "untracked" else "contracts/target_contract.yaml")
    path.write_text(path.read_text() + "# local changes\n" if path.exists() else "local work\n")
    if change in {"staged", "commit", "tagged_commit"}:
        _git(["add", str(path.relative_to(dest))], dest)
    if change in {"commit", "tagged_commit"}:
        _git(["commit", "-qm", "local work"], dest)
    if change == "tagged_commit":
        _git(["tag", "local-tag"], dest)
    before = (path.read_bytes(), _git(["rev-parse", "HEAD"], dest), _git(["status", "--porcelain"], dest))
    with pytest.raises(oot_fetch.FetchError, match="local"):
        oot_fetch.fetch("fixturenpu", url=src.as_uri(), dest=dest)
    assert (path.read_bytes(), _git(["rev-parse", "HEAD"], dest), _git(["status", "--porcelain"], dest)) == before


def test_missing_ref_preserves_checkout_and_allows_retry(tmp_path):
    src = _make_target_repo(tmp_path / "src")
    _git(["tag", "v1-fixture"], src)
    dest = oot_fetch.fetch("fixturenpu", url=src.as_uri(), dest=tmp_path / "dest", champion="v1-fixture")
    before = _git(["rev-parse", "HEAD"], dest)
    with pytest.raises(oot_fetch.FetchError):
        oot_fetch.fetch("fixturenpu", url=src.as_uri(), dest=dest, champion="missing")
    assert _git(["rev-parse", "HEAD"], dest) == before
    oot_fetch.fetch("fixturenpu", url=src.as_uri(), dest=dest, champion="v1-fixture")
    assert _git(["rev-parse", "HEAD"], dest) == before


def test_ignored_collision_is_preserved_but_unrelated_build_output_is_allowed(tmp_path):
    src = _make_target_repo(tmp_path / "src")
    _commit(src, ".gitignore", "build/\ncompiler.bin\n")
    dest = oot_fetch.fetch("fixturenpu", url=src.as_uri(), dest=tmp_path / "dest")
    (dest / "build").mkdir()
    (dest / "build/local.o").write_text("keep build output")
    _commit(src, "revision.txt", "next")
    oot_fetch.fetch("fixturenpu", url=src.as_uri(), dest=dest)
    assert (dest / "build/local.o").read_text() == "keep build output"
    (dest / "compiler.bin").write_text("local ignored executable")
    (src / "compiler.bin").write_text("remote executable")
    _git(["add", "-f", "compiler.bin"], src)
    _git(["commit", "-qm", "publish compiler"], src)
    before = _git(["rev-parse", "HEAD"], dest)
    with pytest.raises(oot_fetch.FetchError):
        oot_fetch.fetch("fixturenpu", url=src.as_uri(), dest=dest)
    assert (dest / "compiler.bin").read_text() == "local ignored executable"
    assert _git(["rev-parse", "HEAD"], dest) == before


@pytest.mark.parametrize("kind", ["missing", "symlink"])
def test_invalid_remote_contract_does_not_replace_checkout(tmp_path, kind):
    src = _make_target_repo(tmp_path / "src")
    dest = oot_fetch.fetch("fixturenpu", url=src.as_uri(), dest=tmp_path / "dest")
    before = _git(["rev-parse", "HEAD"], dest)
    contract = src / "contracts/target_contract.yaml"
    contract.unlink()
    if kind == "symlink":
        contract.symlink_to("outside.yaml")
    _git(["add", "-A"], src)
    _git(["commit", "-qm", "invalid contract"], src)
    with pytest.raises(oot_fetch.FetchError, match="ordinary"):
        oot_fetch.fetch("fixturenpu", url=src.as_uri(), dest=dest)
    assert _git(["rev-parse", "HEAD"], dest) == before
    assert (dest / "contracts/target_contract.yaml").is_file()


def test_update_refuses_different_origin(tmp_path):
    src = _make_target_repo(tmp_path / "src")
    other = _make_target_repo(tmp_path / "other")
    dest = oot_fetch.fetch("fixturenpu", url=src.as_uri(), dest=tmp_path / "dest")
    before = _git(["rev-parse", "HEAD"], dest)
    with pytest.raises(oot_fetch.FetchError, match="origin differs"):
        oot_fetch.fetch("fixturenpu", url=other.as_uri(), dest=dest)
    assert _git(["rev-parse", "HEAD"], dest) == before


@pytest.mark.parametrize("ref", ["--upload-pack=unexpected", "--force", "", "bad..ref"])
def test_unsafe_ref_refuses_before_clone(tmp_path, ref):
    dest = tmp_path / "dest"
    with pytest.raises(oot_fetch.FetchError):
        oot_fetch.fetch("fixturenpu", url=(tmp_path / "absent").as_uri(), dest=dest, champion=ref)
    assert not dest.exists()
