"""Protect exact historical evidence when restoring into a portable artifact tree."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile

import pytest

from merlin.common.paths import repo_root

SPEC = importlib.util.spec_from_file_location(
    'restore_reference_archive', repo_root() / 'build_tools/scripts/restore_reference_archive.py'
)
ARCHIVE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ARCHIVE)


def fixture(tmp_path, *, member='records/original.txt', special=False):
    payload = b'original prompt bytes\n'
    archive = tmp_path / 'records.tar.gz'
    with tarfile.open(archive, 'w:gz') as target:
        info = tarfile.TarInfo(member)
        info.mode = 0o644
        info.size = len(payload)
        if special:
            info.type = tarfile.SYMTYPE
            info.linkname = '/outside'
            target.addfile(info)
        else:
            target.addfile(info, io.BytesIO(payload))
    record = {'schema': 'merlin.reference_archive.v1',
              'archive': {'bytes': archive.stat().st_size,
                          'sha256': ARCHIVE.sha256_file(archive)},
              'files': [{'path': member, 'member': member, 'mode': '0o644',
                         'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}]}
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(record))
    return archive, manifest, payload


def test_restore_preserves_original_bytes_and_reuses_verified_files(tmp_path):
    archive, manifest, payload = fixture(tmp_path)
    _, rows = ARCHIVE.load_members(manifest, archive)
    ARCHIVE.verify(archive, rows)
    destination = tmp_path / 'evidence'
    assert ARCHIVE.restore(archive, destination, rows) == 1
    assert (destination / 'records/original.txt').read_bytes() == payload
    assert ARCHIVE.restore(archive, destination, rows) == 0


def test_different_existing_file_is_never_overwritten(tmp_path):
    archive, manifest, _ = fixture(tmp_path)
    _, rows = ARCHIVE.load_members(manifest, archive)
    destination = tmp_path / 'evidence'
    (destination / 'records').mkdir(parents=True)
    target = destination / 'records/original.txt'
    target.write_bytes(b'local edits')
    with pytest.raises(ValueError, match='refusing to overwrite'):
        ARCHIVE.restore(archive, destination, rows)
    assert target.read_bytes() == b'local edits'


def test_archive_checksum_mismatch_stops_before_extraction(tmp_path):
    archive, manifest, _ = fixture(tmp_path)
    archive.write_bytes(archive.read_bytes() + b'tamper')
    with pytest.raises(ValueError, match='archive bytes'):
        ARCHIVE.load_members(manifest, archive)


@pytest.mark.parametrize('name', ['../escape', '/absolute', 'records/../escape', './alias'])
def test_unsafe_member_paths_are_rejected(tmp_path, name):
    archive, manifest, _ = fixture(tmp_path, member=name)
    with pytest.raises(ValueError, match='member identity'):
        ARCHIVE.load_members(manifest, archive)


def test_symlink_archive_member_is_rejected(tmp_path):
    archive, manifest, _ = fixture(tmp_path, special=True)
    _, rows = ARCHIVE.load_members(manifest, archive)
    with pytest.raises(ValueError, match='non-file'):
        ARCHIVE.verify(archive, rows)


def test_symlink_destination_is_rejected_without_touching_target(tmp_path):
    archive, manifest, _ = fixture(tmp_path)
    _, rows = ARCHIVE.load_members(manifest, archive)
    destination = tmp_path / 'evidence'
    outside = tmp_path / 'outside'
    outside.mkdir()
    destination.mkdir()
    (destination / 'records').symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match='symlink destination'):
        ARCHIVE.restore(archive, destination, rows)
    assert not list(outside.iterdir())
