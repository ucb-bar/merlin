"""Real byte handoffs and refusal at mutable file boundaries."""

import hashlib
import os
from pathlib import Path

import pytest

from merlin.common.pinned_files import FileHandoffError, PinnedFile, snapshot_files, verify_files


def selected(path, data):
    path.write_bytes(data)
    return PinnedFile(path, hashlib.sha256(data).hexdigest())


def test_snapshot_retains_complete_bytes_and_explicit_input_order(tmp_path):
    # Beyond one streaming chunk; equal basenames remain distinct sources.
    (tmp_path / "one").mkdir()
    (tmp_path / "two").mkdir()
    payload = bytes(range(256)) * 8193
    inputs = (selected(tmp_path / "one" / "same.bin", payload), selected(tmp_path / "two" / "same.bin", b"second"))
    destination = tmp_path / "private"
    snapshots = snapshot_files(inputs, destination)
    assert [item.path.read_bytes() for item in snapshots] == [payload, b"second"]
    assert [item.sha256 for item in snapshots] == [item.sha256 for item in inputs]
    assert destination.stat().st_mode & 0o777 == 0o700
    assert destination.stat().st_uid == os.geteuid()
    assert all(item.path.stat().st_mode & 0o777 == 0o444 for item in snapshots)
    inputs[0].path.write_bytes(b"later source mutation")
    verify_files(snapshots)
    with pytest.raises(FileHandoffError, match="changed"):
        verify_files(inputs)


def test_changed_admitted_source_refuses_before_snapshot_creation(tmp_path):
    item = selected(tmp_path / "source", b"original")
    item.path.write_bytes(b"different")
    with pytest.raises(FileHandoffError, match="changed"):
        snapshot_files((item,), tmp_path / "private")
    assert not (tmp_path / "private").exists()


def test_existing_destination_is_preserved_and_never_reused(tmp_path):
    item = selected(tmp_path / "source", b"original")
    root = tmp_path / "private"
    root.mkdir()
    marker = root / "existing"
    marker.write_bytes(b"preserve")
    with pytest.raises(FileHandoffError, match="could not be created"):
        snapshot_files((item,), root)
    assert marker.read_bytes() == b"preserve"


def test_source_mutation_during_handoff_returns_no_selection(tmp_path, monkeypatch):
    item = selected(tmp_path / "source", b"original")
    original_fsync = os.fsync

    def mutate_source(descriptor):
        original_fsync(descriptor)
        item.path.write_bytes(b"changed during the handoff")

    monkeypatch.setattr(os, "fsync", mutate_source)
    with pytest.raises(FileHandoffError, match="changed"):
        snapshot_files((item,), tmp_path / "private")
    # The exact copied bytes remain as a failed-attempt diagnostic, not a grant.
    assert (tmp_path / "private" / "input-0000-source").read_bytes() == b"original"


def test_changed_snapshot_refuses_even_if_it_has_readonly_mode(tmp_path):
    item = selected(tmp_path / "source", b"original")
    snapshot = snapshot_files((item,), tmp_path / "private")[0]
    # Its owner can change permissions. Read-only mode alone is never proof.
    snapshot.path.chmod(0o600)
    snapshot.path.write_bytes(b"replacement")
    snapshot.path.chmod(0o444)
    with pytest.raises(FileHandoffError, match="changed"):
        verify_files((snapshot,))


def test_symlink_substitution_and_fifo_refuse_without_reading(tmp_path):
    item = selected(tmp_path / "source", b"original")
    other = tmp_path / "other"
    other.write_bytes(b"original")
    item.path.unlink()
    item.path.symlink_to(other)
    with pytest.raises(FileHandoffError):
        verify_files((item,))
    item.path.unlink()
    os.mkfifo(item.path)
    with pytest.raises(FileHandoffError, match="regular"):
        verify_files((item,))


@pytest.mark.parametrize("digest", [True, None, "", "A" * 64, "0" * 63])
def test_ambiguous_or_invalid_digest_refuses(tmp_path, digest):
    with pytest.raises(FileHandoffError):
        PinnedFile(tmp_path / "source", digest)


def test_duplicate_or_untyped_selection_cannot_issue_snapshot(tmp_path):
    item = selected(tmp_path / "source", b"original")
    for files in ((), (item, item), [item], (object(),)):
        with pytest.raises(FileHandoffError):
            snapshot_files(files, tmp_path / "private")
    assert not (tmp_path / "private").exists()


def test_relative_or_parent_symlink_paths_refuse(tmp_path):
    with pytest.raises(FileHandoffError):
        PinnedFile(Path("relative"), "0" * 64)
    actual = tmp_path / "actual"
    actual.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(actual, target_is_directory=True)
    with pytest.raises(FileHandoffError):
        PinnedFile(alias / "file", "0" * 64)
