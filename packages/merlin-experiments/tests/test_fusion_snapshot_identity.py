"""Fusion uses the sealed source origin after Git metadata has been removed."""

import shutil
import subprocess

import pytest
from merlin_experiments import frozen_python
from merlin_experiments import source_snapshot as SNAP
from merlin_experiments.phase0 import fusion_validation as V


@pytest.fixture
def checkout(tmp_path):
    source = tmp_path / "checkout"
    (source / "python/toy").mkdir(parents=True)
    (source / "python/toy/__init__.py").write_text("VALUE = 1\n")
    subprocess.run(["git", "init", "--quiet", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "add", "python"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.hooksPath=/dev/null",
            "commit",
            "--quiet",
            "-m",
            "fixture",
        ],
        check=True,
    )
    return source, V._revision(source)


def snapshot(source, destination):
    SNAP.create(
        source,
        destination,
        output_root=destination.parent / "output",
        source_roots=("python",),
        python_roots=("python",),
        legacy_roots=(),
        runtime_links=False,
    )
    return destination


def select_frozen(monkeypatch, root):
    seal, _ = SNAP.load_seal(root, "snapshot")
    identity = dict(path=str(seal), sha256=SNAP.sha_file(seal))
    monkeypatch.setattr(V, "repo_root", lambda: root)
    monkeypatch.setattr(frozen_python, "active_source_identity", lambda: identity)

    def no_git(*args):
        pytest.fail("frozen identity must not consult Git")

    monkeypatch.setattr(V, "_revision", no_git)


def test_live_checkout_identity_uses_git(checkout, monkeypatch):
    source, commit = checkout
    monkeypatch.setattr(V, "repo_root", lambda: source)
    monkeypatch.setattr(frozen_python, "active_source_identity", lambda: None)
    assert V._repository_commit() == commit


def test_sealed_origin_survives_removal_of_original_checkout(checkout, tmp_path, monkeypatch):
    source, commit = checkout
    frozen = snapshot(source, tmp_path / "frozen")
    assert not (frozen / ".git").exists()
    assert SNAP.verify(frozen)["repository_commit"] == commit
    shutil.rmtree(source)
    select_frozen(monkeypatch, frozen)
    assert V._repository_commit() == commit


@pytest.mark.parametrize("change", ["receipt", "source", "missing_commit", "invalid_commit", "guard"])
def test_frozen_identity_refuses_changed_or_missing_provenance(checkout, tmp_path, monkeypatch, change):
    source, _ = checkout
    frozen = snapshot(source, tmp_path / "frozen")
    seal, receipt = SNAP.load_seal(frozen, "snapshot")
    if change in {"missing_commit", "invalid_commit"}:
        receipt.pop("repository_commit")
        if change == "invalid_commit":
            receipt["repository_commit"] = "not-a-commit"
        frozen.chmod(0o700)
        seal.unlink()
        SNAP.seal(frozen, "snapshot", receipt)
        frozen.chmod(0o500)
    select_frozen(monkeypatch, frozen)
    if change in {"receipt", "source"}:
        path = seal if change == "receipt" else frozen / "python/toy/__init__.py"
        path.chmod(0o600)
        path.write_bytes(path.read_bytes() + b"\n")
        path.chmod(0o400)
    elif change == "guard":
        monkeypatch.setattr(frozen_python, "active_source_identity", lambda: {"path": str(seal), "sha256": "0" * 64})
    with pytest.raises((SNAP.SnapshotError, ValueError), match="seal|source changed|repository commit|guarded"):
        V._repository_commit()


def test_source_without_git_never_inherits_parent_commit(checkout, tmp_path, monkeypatch):
    parent, _ = checkout
    source = parent / "installed"
    (source / "python").mkdir(parents=True)
    (source / "python/toy.py").write_text("VALUE = 1\n")
    frozen = snapshot(source, tmp_path / "frozen")
    assert "repository_commit" not in SNAP.verify(frozen)
    select_frozen(monkeypatch, frozen)
    with pytest.raises(ValueError, match="lacks originating repository commit"):
        V._repository_commit()


def test_commit_change_during_copy_is_not_sealed(checkout, tmp_path, monkeypatch):
    source, commit = checkout
    commits = iter([commit, "0" * 40])
    monkeypatch.setattr(SNAP, "_checkout_commit", lambda root: next(commits))
    frozen = tmp_path / "frozen"
    with pytest.raises(SNAP.SnapshotError, match="commit changed during snapshot creation"):
        snapshot(source, frozen)
    assert not list(frozen.glob("snapshot.*.json"))
