"""Destructive custody fixtures cannot mutate another snapshot through CAS links."""

import os

import pytest
import test_compiler_profile_child as PROFILE

from merlin.common import bundle_snapshot as BW
from merlin.common import content_store
from merlin.common.digest import sha256_file
from merlin.targetgen.sandbox import bwrap


def test_source_mutating_profile_fixture_is_private_while_peer_snapshots_still_share(tmp_path, monkeypatch):
    # Never reproduce corruption in the operator's store. With the old fixture,
    # all three paths below shared only this explicitly selected test-local CAS.
    store = tmp_path / "test-cas"
    monkeypatch.setenv(content_store.LOCATION_ENV, str(store))
    case = tmp_path / "destructive-case"
    case.mkdir()
    original = PROFILE.original.__wrapped__(case, monkeypatch)
    frozen = original.frozen
    contents, metadata = frozen.read_bytes(), frozen.stat()

    # Ordinary sharing is a positive control. Only the destructive fixture
    # chooses private copies; production CAS behavior remains unchanged.
    monkeypatch.setenv(content_store.LOCATION_ENV, str(store))
    peers = []
    relative = frozen.relative_to(BW.bundle_snapshot_root(original.owner.workspace))
    for index in range(2):
        workspace = tmp_path / f"peer-{index}" / "workspace"
        workspace.mkdir(parents=True)
        bwrap.materialize_bundle_inputs(workspace, original.owner.bundle, repo=original.owner.repo)
        peers.append(BW.bundle_snapshot_root(workspace) / relative)
    assert os.path.samefile(peers[0], peers[1]), "positive control did not use the test-local CAS"
    assert not os.path.samefile(frozen, peers[0]), "destructive fixture still shares another snapshot's inode"
    assert frozen.stat().st_nlink == 1
    peer_digests = [sha256_file(peer) for peer in peers]
    digest = sha256_file(frozen)
    obj = store / digest[:2] / (digest + (".x" if metadata.st_mode & 0o111 else ""))
    assert os.path.samefile(obj, peers[0])

    frozen.chmod(0o600)
    frozen.write_bytes(bytes([contents[0] ^ 1]) + contents[1:])
    os.utime(frozen, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
    with pytest.raises(RuntimeError, match="content differs"):
        BW.verify_bundle_snapshot(original.owner.workspace, original.owner.bundle, repo=original.owner.repo)
    assert [sha256_file(peer) for peer in peers] == peer_digests
    assert sha256_file(obj) == digest
