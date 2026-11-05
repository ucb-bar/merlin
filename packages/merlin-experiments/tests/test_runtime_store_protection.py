"""Protected shared runtime inodes can still supply exact, independent snapshots."""

import errno
import stat

import pytest
from merlin_experiments.capture_execution import runtime_store
from merlin_experiments.capture_execution.sealed_m2m import _snapshot_tree


def test_protected_runtime_copy_preserves_selected_bytes_and_modes(tmp_path, monkeypatch):
    entry, destination = tmp_path / "entry", tmp_path / "runtime"
    source = entry / "lib/package/data.bin"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"exact runtime bytes\0\xff")
    source.chmod(0o440)
    source.parent.chmod(0o550)
    destination.mkdir()
    identity = _snapshot_tree(entry)

    def protected(*_args):
        raise PermissionError(errno.EPERM, "immutable source")

    monkeypatch.setattr(runtime_store.os, "link", protected)
    runtime_store._link_entry(entry, destination)
    copied = destination / source.relative_to(entry)
    assert copied.stat().st_ino != source.stat().st_ino
    assert stat.S_IMODE(copied.stat().st_mode) == 0o440
    assert _snapshot_tree(destination) == identity
    assert _snapshot_tree(entry) == identity
    # Copy fallback must never overwrite an existing snapshot member.
    copied.chmod(0o640)
    copied.write_bytes(b"existing member")
    with pytest.raises(FileExistsError):
        runtime_store._link_entry(entry, destination)
    assert copied.read_bytes() == b"existing member"


@pytest.mark.parametrize("error", [errno.EACCES, errno.EXDEV, errno.EIO])
def test_unrelated_link_failures_still_refuse_snapshot(tmp_path, monkeypatch, error):
    entry, destination = tmp_path / "entry", tmp_path / "runtime"
    entry.mkdir()
    (entry / "data").write_bytes(b"selected")
    destination.mkdir()

    def failed(*_args):
        raise OSError(error, "link failure")

    monkeypatch.setattr(runtime_store.os, "link", failed)
    with pytest.raises(OSError) as raised:
        runtime_store._link_entry(entry, destination)
    assert raised.value.errno == error
    assert not (destination / "data").exists()
