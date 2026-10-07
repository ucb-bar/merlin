"""A target's selected elaboration cannot silently switch checkout or gitlink."""

from __future__ import annotations

import subprocess

import pytest

from merlin.targetgen.rtl import declared_revision, introspect

GITLINK_REVISION = "1" * 40


def _git(root, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True)
    return result.stdout.strip()


def _source_repo(tmp_path):
    root = tmp_path / "selected"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "RTL Test")
    _git(root, "config", "user.email", "rtl-test@example.invalid")
    (root / "source.txt").write_text("selected source\n")
    _git(root, "add", "source.txt")
    _git(root, "commit", "-qm", "Add source")
    _git(root, "update-index", "--add", "--cacheinfo", f"160000,{GITLINK_REVISION},vendor/device")
    _git(root, "commit", "-qm", "Select device")
    return root, _git(root, "rev-parse", "HEAD")


def test_declared_revision_checks_parent_and_uninitialized_gitlink(tmp_path):
    root, revision = _source_repo(tmp_path)
    declared_revision.verify_declared_revision(root, revision, (("vendor/device", GITLINK_REVISION),))
    with pytest.raises(introspect.RtlSourceInvalid, match="checkout revision differs"):
        declared_revision.verify_declared_revision(root, "0" * 40, ())
    with pytest.raises(introspect.RtlSourceInvalid, match="gitlink vendor/device differs"):
        declared_revision.verify_declared_revision(root, revision, (("vendor/device", "2" * 40),))
    with pytest.raises(introspect.RtlSourceInvalid, match="gitlink declaration is invalid"):
        declared_revision.verify_declared_revision(root, revision, (("../device", GITLINK_REVISION),))
    with pytest.raises(introspect.RtlSourceInvalid, match="require a parent"):
        declared_revision.verify_declared_revision(root, None, (("vendor/device", GITLINK_REVISION),))


def test_selected_source_reader_enforces_declared_revision(tmp_path, monkeypatch):
    root, revision = _source_repo(tmp_path)
    declaration = tmp_path / "descriptor.yaml"
    declaration.write_text(
        "target: test_grid\nrtl:\n  elaboration:\n    ext_root: test_grid\n"
        "    config: TestConfig\n    generator: grid\n"
        f"    source_revision: '{revision}'\n"
        f"    gitlinks:\n      vendor/device: '{GITLINK_REVISION}'\n"
    )
    monkeypatch.setattr(introspect, "_declaration_files", lambda target: [declaration])
    monkeypatch.setattr(introspect, "ext_path", lambda name: root)
    source = introspect.declared_rtl_source("test_grid")
    assert source.config == "TestConfig" and source.source_revision == revision
    assert source.gitlinks == (("vendor/device", GITLINK_REVISION),)

    (root / "source.txt").write_text("changed and uncommitted\n")
    with pytest.raises(introspect.RtlSourceInvalid, match="modified tracked source"):
        introspect.declared_rtl_source("test_grid")
    _git(root, "add", "source.txt")
    _git(root, "commit", "-qm", "Change source")
    with pytest.raises(introspect.RtlSourceInvalid, match="checkout revision differs"):
        introspect.declared_rtl_source("test_grid")
