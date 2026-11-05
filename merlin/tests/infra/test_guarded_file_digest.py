"""Exact request-scoped file reuse; real change watches, no timestamp trust."""
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor

import pytest

from merlin.common import guarded_file_digest as G


@pytest.fixture
def watched(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_SNAPSHOT_LEASE_CACHE", "1")
    path = tmp_path / "selected.fir"
    path.write_bytes(b"original source\n")
    owner = G.GuardedDigests()
    calls = []
    read = G._read
    def counted(path):
        calls.append(path)
        return read(path)
    monkeypatch.setattr(G, "_read", counted)
    yield path, owner, calls
    owner.close()


def test_concurrent_scopes_hash_once_and_unscoped_reads_remain_fresh(watched):
    path, owner, calls = watched
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    with owner.scope():
        assert G.sha256(path) == expected
    if not owner._entries:
        pytest.skip("kernel watches unavailable")
    def verify(_):
        with owner.scope():
            return G.sha256(path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(verify, range(12))) == [expected] * 12
    assert len(calls) == 1
    G.sha256(path)
    G.sha256(path)
    assert len(calls) == 3


@pytest.mark.parametrize("mutation", ["write", "hardlink", "replace", "symlink"])
def test_mutation_never_reuses_old_bytes_even_with_restored_mtime(watched, mutation):
    path, owner, _ = watched
    original = path.stat()
    alias = path.parent / "alias"
    if mutation == "hardlink":
        os.link(path, alias)
    with pytest.raises(ValueError):
        with owner.scope():
            G.sha256(path)
            if not owner._entries:
                pytest.skip("kernel watches unavailable")
            if mutation == "replace":
                alias.write_bytes(b"different bytes\n")
                alias.replace(path)
            elif mutation == "symlink":
                path.rename(alias)
                path.symlink_to(alias)
            else:
                (alias if mutation == "hardlink" else path).write_bytes(b"modified source\n")
                os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))
            G.sha256(path)


def test_final_scope_check_catches_mutation_after_last_consumer(watched):
    path, owner, _ = watched
    with pytest.raises(ValueError, match="bytes changed"):
        with owner.scope():
            G.sha256(path)
            if not owner._entries:
                pytest.skip("kernel watches unavailable")
            path.write_bytes(b"changed after observation")


def test_watch_unavailable_uses_full_reads(watched, monkeypatch):
    path, owner, calls = watched
    monkeypatch.delenv("MERLIN_SNAPSHOT_LEASE_CACHE")
    with owner.scope():
        G.sha256(path)
        G.sha256(path)
    assert len(calls) == 2 and not owner._entries


def test_closed_guard_falls_back_and_never_rearms(watched):
    path, owner, calls = watched
    with owner.scope():
        G.sha256(path)
        if not owner._entries:
            pytest.skip("kernel watches unavailable")
        owner._entries[path][1].close()  # also the overflow/error transition
        G.sha256(path)
        G.sha256(path)
    assert len(calls) == 4  # entry, two reads, publication recheck
    assert owner._entries[path][1].fd is None


def test_inherited_and_closed_owners_cannot_reuse(watched, monkeypatch):
    path, owner, calls = watched
    with owner.scope():
        G.sha256(path)
    monkeypatch.setattr(owner, "pid", -1)
    with pytest.raises(ValueError, match="original live process"):
        with owner.scope():
            pass
    before = len(calls)
    owner._digest(path)
    assert len(calls) == before + 1
    monkeypatch.setattr(owner, "pid", os.getpid())
    owner.close()
    with pytest.raises(ValueError, match="original live process"):
        with owner.scope():
            pass
