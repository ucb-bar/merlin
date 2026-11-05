"""Public CAS placement retains actual Linux custody across snapshot aliases."""

from __future__ import annotations

import errno
import hashlib
import os

import pytest

from merlin.common import bundle_snapshot as S
from merlin.common import content_store as C
from merlin.common import read_lease_digest as L


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch):
    L.close()
    monkeypatch.setenv("MERLIN_SNAPSHOT_LEASE_CACHE", "1")
    monkeypatch.setattr(L, "MIN_BYTES", 4096)
    yield
    L.close()


def selected(tmp_path, monkeypatch):
    source, store = tmp_path / "source", tmp_path / "store"
    source.write_bytes(b"original selected public bytes\n" * 512)
    monkeypatch.delenv("MERLIN_SNAPSHOT_LEASE_CACHE")
    first = tmp_path / "first"
    assert C.place_file(source, first, store)
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    obj = store / sha[:2] / sha
    monkeypatch.setenv("MERLIN_SNAPSHOT_LEASE_CACHE", "1")
    return source, store, obj, first


def require_selected_lease(obj):
    if L._CACHE is None or (obj.stat().st_dev, obj.stat().st_ino) not in L._CACHE.entries:
        pytest.skip("Linux read leases unavailable on the selected temporary filesystem")


def break_lease(path):
    path.chmod(0o600)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_NONBLOCK)
    except OSError as error:
        assert error.errno in (errno.EAGAIN, errno.EWOULDBLOCK)
    else:
        os.close(descriptor)


def test_reused_object_and_snapshot_aliases_hash_same_inode_once(tmp_path, monkeypatch):
    source, store, obj, _ = selected(tmp_path, monkeypatch)
    original = L.hashlib.file_digest
    reads = []

    def counted(stream, *args):
        reads.append((os.fstat(stream.fileno()).st_ino, os.fstat(stream.fileno()).st_size))
        return original(stream, *args)

    monkeypatch.setattr(L.hashlib, "file_digest", counted)
    snapshots = [tmp_path / "snapshot-one", tmp_path / "snapshot-two"]
    for root in snapshots:
        root.mkdir()
        assert C.place_file(source, root / "payload", store)
        require_selected_lease(obj)
        assert (root / "payload").samefile(obj)
        digest = S.snapshot_content(root)
        assert digest[1:] == (1, source.stat().st_size)
    assert reads == [(obj.stat().st_ino, source.stat().st_size)]


def test_default_reuse_retains_independent_object_reads(tmp_path, monkeypatch):
    source, store, obj, _ = selected(tmp_path, monkeypatch)
    monkeypatch.delenv("MERLIN_SNAPSHOT_LEASE_CACHE")
    original, reads = C.digest_file, []

    def counted(path):
        reads.append(path)
        return original(path)

    monkeypatch.setattr(C, "digest_file", counted)
    assert C.place_file(source, tmp_path / "one", store)
    assert C.place_file(source, tmp_path / "two", store)
    assert reads == [source, obj, source, obj]
    assert L._CACHE is None


def test_existing_writer_keeps_fresh_full_hash_and_detects_corruption(tmp_path, monkeypatch):
    source, store, obj, _ = selected(tmp_path, monkeypatch)
    obj.chmod(0o600)
    writer = os.open(obj, os.O_RDWR)
    try:
        assert C.place_file(source, tmp_path / "before", store)
        assert not L._CACHE.entries
        saved = obj.stat()
        os.pwrite(writer, b"changed!", 0)
        os.utime(obj, ns=(saved.st_atime_ns, saved.st_mtime_ns))
        assert C.place_file(source, tmp_path / "after", store)
        assert (tmp_path / "after").read_bytes() == source.read_bytes()
        assert not (tmp_path / "after").samefile(tmp_path / "before")
    finally:
        os.close(writer)


@pytest.mark.parametrize("change", ["write_attempt", "replacement", "custodian_loss"])
def test_changed_selected_custody_refuses_placement_result(tmp_path, monkeypatch, change):
    source, store, obj, _ = selected(tmp_path, monkeypatch)
    assert C.place_file(source, tmp_path / "warm", store)
    require_selected_lease(obj)
    original = os.link
    destination, accepted = tmp_path / "next", tmp_path / "accepted"

    def changed(src, dst, *args, **kwargs):
        if dst == destination:
            if change == "write_attempt":
                break_lease(obj)
            elif change == "replacement":
                replacement = tmp_path / "replacement"
                replacement.write_bytes(source.read_bytes())
                os.replace(replacement, obj)
            else:
                L._CACHE.process.kill()
                L._CACHE.process.wait(timeout=3)
        return original(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "link", changed)
    with pytest.raises(RuntimeError, match="lease custody changed"):
        C.place_file(source, destination, store)
        accepted.write_bytes(b"placement accepted")
    assert not accepted.exists()


def test_private_permission_preserving_copy_never_leases_public_store(tmp_path, monkeypatch):
    source = tmp_path / "private-source"
    source.write_bytes(b"owner-only input bytes\n" * 512)
    source.chmod(0o600)
    calls = []
    monkeypatch.setattr(L, "digest", lambda path: calls.append(path))
    destination = tmp_path / "private-copy"
    assert not C.place_file(source, destination, None, preserve_permissions=True)
    assert not calls
    assert destination.read_bytes() == source.read_bytes()
    assert destination.stat().st_mode & 0o7777 == 0o400
    assert not destination.samefile(source)
