"""Pinned dependency restoration preserves links without following or executing them."""

import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tarfile

import pytest

from merlin.common.paths import repo_root

SCRIPT = repo_root() / "build_tools/scripts/restore_paper_dependencies.py"
sys.path.insert(0, str(SCRIPT.parent))
try:
    spec = importlib.util.spec_from_file_location("paper_dependency_restore", SCRIPT)
    R = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(R)
finally:
    sys.path.pop(0)


def capture(tmp_path, trees):
    archives, metadata = tmp_path / "archives", tmp_path / "metadata"
    archives.mkdir()
    metadata.mkdir()
    rows = []
    for identity, source_root, entries in trees:
        members, raw = [], io.BytesIO()
        with tarfile.open(fileobj=raw, mode="w") as stream:
            for name, kind, payload in entries:
                item = tarfile.TarInfo("./" + name)
                item.mode = 0o755 if kind == "directory" else 0o777 if kind == "symlink" else 0o644
                row = dict(path=name, kind="file" if kind == "hardlink" else kind, mode=oct(item.mode))
                if kind == "directory":
                    item.type = tarfile.DIRTYPE
                    stream.addfile(item)
                elif kind == "symlink":
                    item.type = tarfile.SYMTYPE
                    item.linkname = payload
                    row["target"] = payload
                    stream.addfile(item)
                else:
                    value = payload[1] if kind == "hardlink" else payload
                    row.update(bytes=len(value), sha256=hashlib.sha256(value).hexdigest())
                    if kind == "hardlink":
                        item.type = tarfile.LNKTYPE
                        item.linkname = "./" + payload[0]
                        stream.addfile(item)
                    else:
                        item.size = len(value)
                        stream.addfile(item, io.BytesIO(value))
                members.append(row)
        archive = archives / (identity + ".tar.zst")
        result = subprocess.run(["zstd", "-q", "-c"], input=raw.getvalue(), capture_output=True, check=True)
        archive.write_bytes(result.stdout)
        tree = hashlib.sha256(json.dumps(members, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        inventory = metadata / (identity + ".inventory.json")
        inventory.write_text(
            json.dumps(dict(schema="merlin.paper_dependency_inventory.v1", members=members, tree_sha256=tree))
        )
        rows.append(
            dict(
                id=identity,
                source=source_root,
                archive=archive.name,
                sha256=R.digest(archive),
                bytes=archive.stat().st_size,
                inventory=inventory.name,
                inventory_sha256=R.digest(inventory),
                members=len(members),
                tree_sha256=tree,
            )
        )
    path = tmp_path / "capture.json"
    path.write_text(json.dumps(dict(schema="merlin.paper_dependency_capture.v1", archives=rows)))
    return path, R.digest(path), archives, metadata


def test_forward_hardlink_and_cross_declared_absolute_symlink(tmp_path, monkeypatch):
    payload = b"#!/bin/sh\necho SHOULD_NOT_EXECUTE\n"
    inputs = capture(
        tmp_path,
        [
            (
                "one",
                "/original/one",
                [
                    ("bin", "directory", None),
                    ("copy", "hardlink", ("bin/tool", payload)),
                    ("bin/tool", "file", payload),
                    ("python", "symlink", "/original/two/bin/python"),
                ],
            ),
            ("two", "/original/two", [("bin", "directory", None), ("bin/python", "file", b"interpreter bytes")]),
        ],
    )
    original = R.extract

    def checked(*args):
        # Across archives, deferred links cannot participate in extraction paths.
        assert not any(p.is_symlink() for p in args[-1].parent.rglob("*"))
        return original(*args)

    monkeypatch.setattr(R, "extract", checked)
    destination = tmp_path / "restored"
    results = R.restore(*inputs, destination)
    assert len(results) == 2
    first = destination / "dependencies/one"
    assert (first / "copy").read_bytes() == payload
    assert (first / "copy").stat().st_ino == (first / "bin/tool").stat().st_ino
    assert (first / "bin/tool").stat().st_mode & 0o777 == 0o644
    assert (first / "bin").stat().st_mode & 0o777 == 0o755
    assert os.readlink(first / "python") == "/original/two/bin/python"
    assert json.loads((destination / "dependency-restoration.json").read_bytes())["programs_executed"] is False
    with pytest.raises(ValueError, match="must not exist"):
        R.restore(*inputs, destination)
    assert (first / "copy").read_bytes() == payload


@pytest.mark.parametrize("target", ["/outside/python", "../../outside", "/original/one-more/tool"])
def test_external_symlink_refuses_before_output_creation(tmp_path, target):
    inputs = capture(tmp_path, [("one", "/original/one", [("link", "symlink", target)])])
    with pytest.raises(ValueError, match="leaves all declared"):
        R.restore(*inputs, tmp_path / "restored")
    assert not (tmp_path / "restored").exists()


@pytest.mark.parametrize("damage", ["capture", "inventory", "archive"])
def test_changed_pins_refuse(tmp_path, damage):
    inputs = capture(tmp_path, [("one", "/original/one", [("file", "file", b"original")])])
    path = {"capture": inputs[0], "inventory": inputs[3] / "one.inventory.json", "archive": inputs[2] / "one.tar.zst"}[
        damage
    ]
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="pinned file differs"):
        R.restore(*inputs, tmp_path / "restored")
    assert not (tmp_path / "restored").exists()


def test_symlink_cannot_be_an_extraction_parent(tmp_path):
    inputs = capture(
        tmp_path,
        [("one", "/original/one", [("link", "symlink", "/original/one/directory"), ("link/file", "file", b"no")])],
    )
    with pytest.raises(ValueError, match="parent"):
        R.restore(*inputs, tmp_path / "restored")
    assert not (tmp_path / "restored").exists()


def test_hardlink_cannot_reference_another_archive(tmp_path):
    inputs = capture(
        tmp_path,
        [
            ("one", "/original/one", [("copy", "hardlink", ("../two/file", b"no"))]),
            ("two", "/original/two", [("file", "file", b"no")]),
        ],
    )
    with pytest.raises(ValueError, match="unsafe archive path"):
        R.restore(*inputs, tmp_path / "restored")
    assert not (tmp_path / "restored").exists()
    assert not list(tmp_path.glob(".dependency-restore-*"))


def test_existing_or_indirect_destination_is_never_overwritten(tmp_path):
    inputs = capture(tmp_path, [("one", "/original/one", [("file", "file", b"original")])])
    destination = tmp_path / "existing"
    destination.mkdir()
    (destination / "keep").write_bytes(b"keep")
    with pytest.raises(ValueError, match="must not exist"):
        R.restore(*inputs, destination)
    assert (destination / "keep").read_bytes() == b"keep"
    alias = tmp_path / "alias"
    alias.symlink_to(destination, target_is_directory=True)
    with pytest.raises(ValueError, match="indirect parent"):
        R.restore(*inputs, alias / "new")


def test_late_existing_destination_is_not_replaced_or_removed(tmp_path, monkeypatch):
    inputs = capture(tmp_path, [("one", "/original/one", [("file", "file", b"original")])])
    destination = tmp_path / "late"
    original = R.extract

    def concurrent(*args):
        result = original(*args)
        destination.mkdir()
        (destination / "foreign").write_bytes(b"other owner")
        return result

    monkeypatch.setattr(R, "extract", concurrent)
    with pytest.raises(FileExistsError):
        R.restore(*inputs, destination)
    assert (destination / "foreign").read_bytes() == b"other owner"
    assert not (destination / "dependencies").exists()
    assert not list(tmp_path.glob(".dependency-restore-*"))


def test_changed_inventory_during_extraction_refuses_publication(tmp_path, monkeypatch):
    inputs = capture(tmp_path, [("one", "/original/one", [("file", "file", b"original")])])
    original = R.extract

    def changed(*args):
        result = original(*args)
        (inputs[3] / "one.inventory.json").write_bytes(b"changed")
        return result

    monkeypatch.setattr(R, "extract", changed)
    with pytest.raises(ValueError, match="pinned file differs"):
        R.restore(*inputs, tmp_path / "restored")
    assert not (tmp_path / "restored").exists()
    assert not list(tmp_path.glob(".dependency-restore-*"))


def test_cyclic_hardlinks_refuse_without_publication(tmp_path):
    inputs = capture(
        tmp_path, [("one", "/original/one", [("a", "hardlink", ("b", b"same")), ("b", "hardlink", ("a", b"same"))])]
    )
    with pytest.raises(ValueError, match="cyclic archive hardlink"):
        R.restore(*inputs, tmp_path / "restored")
    assert not (tmp_path / "restored").exists()
    assert not list(tmp_path.glob(".dependency-restore-*"))
