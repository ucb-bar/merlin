"""Frozen corpus digests reuse kernel custody, never a cached tree roster."""
import pytest

from merlin.common import read_lease_digest as leases
from merlin_experiments.phase1 import source_inputs as sources


@pytest.fixture
def lease_cache(monkeypatch):
    leases.close()
    monkeypatch.setenv("MERLIN_SNAPSHOT_LEASE_CACHE", "1")
    monkeypatch.setenv("MERLIN_SNAPSHOT_LEASE_MIN_BYTES", "4096")
    yield
    leases.close()


def test_frozen_fingerprint_reuses_bytes_but_rechecks_members(tmp_path, monkeypatch, lease_cache):
    root = tmp_path / "corpus"
    root.mkdir()
    payload = root / "large.bin"
    payload.write_bytes(b"inputs" * 8192)
    expected = sources.fingerprint(root)
    if leases.digest(payload) is None:
        pytest.skip("Linux read leases unavailable")
    assert sources.fingerprint(root, leased=True) == expected
    original = sources._sha
    reads = []

    def observe(path):
        reads.append(path)
        return original(path)

    monkeypatch.setattr(sources, "_sha", observe)
    assert sources.fingerprint(root, leased=True) == expected
    assert payload not in reads
    (root / "new-directory").mkdir()
    assert sources.fingerprint(root, leased=True) != expected
    # Ordinary mutable-source callers retain their original full-read behavior.
    sources.fingerprint(root)
    assert payload in reads


def test_frozen_fingerprint_refuses_replaced_leased_input(tmp_path, lease_cache):
    payload = tmp_path / "payload"
    payload.write_bytes(b"a" * 8192)
    if leases.digest(payload) is None:
        pytest.skip("Linux read leases unavailable")
    sources.fingerprint(tmp_path, leased=True)
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"b" * 8192)
    replacement.replace(payload)
    with pytest.raises(RuntimeError, match="custody changed"):
        sources.fingerprint(tmp_path, leased=True)


def test_unsupported_leases_preserve_symlink_digest(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_SNAPSHOT_LEASE_CACHE", "0")
    root = tmp_path / "corpus"
    root.mkdir()
    target = tmp_path / "input"
    target.write_bytes(b"data")
    (root / "link").symlink_to(target)
    assert sources.fingerprint(root, leased=True) == sources.fingerprint(root)
