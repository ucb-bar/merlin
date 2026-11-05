"""Destructive custody fixtures cannot mutate another snapshot through CAS links."""

import os

import pytest
import test_common_request_snapshot_scope as REQUEST
import test_compiler_profile_child as PROFILE
import test_compiler_tool_profile_v6 as RELEASE
import test_lean_emission_context as EMISSION
import test_optional_import_batch as OPTIONAL

from merlin.common import bundle_snapshot as BW
from merlin.common import content_store
from merlin.common.digest import sha256_file
from merlin.targetgen.sandbox import bwrap


def _destructive_case(kind, root, monkeypatch):
    if kind == "profile":
        case = PROFILE.original.__wrapped__(root, monkeypatch)
        return case.owner, case.frozen
    if kind == "request":
        case = REQUEST.original.__wrapped__(root, monkeypatch)
        return case.service, case.padding
    if kind == "optional":
        owner, sources = OPTIONAL.selected.__wrapped__(root, monkeypatch)
        source = sources[0]
    elif kind == "emission":
        owner = EMISSION._owner({"tmp": root}, monkeypatch)
        source = next(path for row in owner.bundle["allowed"] if (path := owner.repo / row["path"]).is_file()
                      and path.stat().st_size > 1)
    else:
        assert kind == "release"
        owner, release = RELEASE.original_owner(root)
        source = release / "merlin/__init__.py"
    return owner, BW.snapshot_path(BW.bundle_snapshot_root(owner.workspace), source, owner.repo)


@pytest.mark.parametrize("kind", ["profile", "request", "optional", "emission", "release"])
def test_source_mutating_fixture_is_private_while_peer_snapshots_still_share(tmp_path, monkeypatch, kind):
    # Never reproduce corruption in the operator's store. With the old fixture,
    # all three paths below shared only this explicitly selected test-local CAS.
    store = tmp_path / "test-cas"
    monkeypatch.setenv(content_store.LOCATION_ENV, str(store))
    case = tmp_path / "destructive-case"
    case.mkdir()
    owner, frozen = _destructive_case(kind, case, monkeypatch)
    contents, metadata = frozen.read_bytes(), frozen.stat()

    # Ordinary sharing is a positive control. Only the destructive fixture
    # chooses private copies; production CAS behavior remains unchanged.
    monkeypatch.setenv(content_store.LOCATION_ENV, str(store))
    peers = []
    relative = frozen.relative_to(BW.bundle_snapshot_root(owner.workspace))
    for index in range(2):
        workspace = tmp_path / f"peer-{index}" / "workspace"
        workspace.mkdir(parents=True)
        bwrap.materialize_bundle_inputs(workspace, owner.bundle, repo=owner.repo)
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
        BW.verify_bundle_snapshot(owner.workspace, owner.bundle, repo=owner.repo)
    assert [sha256_file(peer) for peer in peers] == peer_digests
    assert sha256_file(obj) == digest
