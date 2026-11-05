"""Portable archive transport never executes its contents or overwrites a tree."""
import hashlib
import importlib.util
import io
from pathlib import Path
import tarfile

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "build_tools/scripts/restore_paper_artifact.py"
spec = importlib.util.spec_from_file_location("paper_artifact_restore", SCRIPT)
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)


def bundle(tmp_path, entries):
    path = tmp_path / "records.tar.gz"
    size = 0
    with tarfile.open(path, "w:gz") as stream:
        for name, kind, payload in entries:
            member = tarfile.TarInfo(name)
            member.mode = 0o644
            if kind == "file":
                member.size = len(payload)
                size += member.size
                stream.addfile(member, io.BytesIO(payload))
            else:
                member.type = tarfile.LNKTYPE if kind == "hardlink" else tarfile.SYMTYPE
                member.linkname = payload
                stream.addfile(member)
    row = dict(id="records", filename=path.name, format="tar.gz", bytes=path.stat().st_size,
               sha256=hashlib.sha256(path.read_bytes()).hexdigest(), members=len(entries),
               expanded_bytes=size, top_level=["records"], destination=".")
    return path, row


def test_restore_files_and_forward_hardlinks(tmp_path):
    path, row = bundle(tmp_path, [("records/copy", "hardlink", "records/original"),
                                 ("records/original", "file", b"untouched source")])
    destination = tmp_path / "data"
    result = R.restore(tmp_path, [row], destination)
    assert result[0]["restored"] is True
    assert (destination / "records/copy").read_bytes() == b"untouched source"
    assert (destination / "records/original").stat().st_mode & 0o777 == 0o644
    assert (destination / "artifact-restoration.json").is_file()
    with pytest.raises(ValueError, match="must not exist"):
        R.restore(tmp_path, [row], destination)


@pytest.mark.parametrize("entries", [
    [("../escape", "file", b"no")],
    [("records/link", "symlink", "../../escape")],
    [("records/link", "hardlink", "../escape")],
    [("records/link", "hardlink", "records/missing")],
    [("records/a", "hardlink", "records/b"), ("records/b", "hardlink", "records/a")],
    [("records/a", "file", b"one"), ("records/a", "file", b"two")],
])
def test_unsafe_members_refuse_without_publishing(tmp_path, entries):
    _, row = bundle(tmp_path, entries)
    destination = tmp_path / "data"
    with pytest.raises(ValueError):
        R.restore(tmp_path, [row], destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".artifact-restore-*"))


def test_changed_archive_refuses(tmp_path):
    path, row = bundle(tmp_path, [("records/source", "file", b"original")])
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checksum or size"):
        R.inspect_archive(path, row)


def test_indirect_destination_refuses(tmp_path):
    _, row = bundle(tmp_path, [("records/source", "file", b"original")])
    (tmp_path / "alias").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="indirect parent"):
        R.restore(tmp_path, [row], tmp_path / "alias/data")
