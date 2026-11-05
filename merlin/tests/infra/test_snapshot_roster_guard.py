"""Kernel membership shortcut keeps fresh payload boundaries and fallback scans."""

import os

import pytest

from merlin.common import bundle_snapshot as S
from merlin.targetgen.sandbox import bwrap as B


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_SNAPSHOT_LEASE_CACHE", "1")
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "source"
    source.write_bytes(b"original bytes")
    ws = tmp_path / "workspace"
    ws.mkdir()
    bundle = {"allowed": [{"path": str(source), "mode": "ro"}], "host_inputs": [], "denied": []}
    B.materialize_bundle_inputs(ws, bundle, repo=repo)
    root = S.bundle_snapshot_root(ws)
    payload = S.snapshot_path(root, source, repo)
    root.chmod(0o700)
    payload.parent.chmod(0o700)
    payload.chmod(0o600)
    return ws, bundle, repo, root, payload


def test_repeated_verified_reads_avoid_roster_walk_but_boundaries_stay_fresh(frozen, monkeypatch):
    ws, bundle, repo, root, _ = frozen
    scans = []
    scan = S._scanned_members

    def record(path):
        scans.append(path)
        return scan(path)

    monkeypatch.setattr(S, "_scanned_members", record)
    with S.verified_snapshot_scope(ws, bundle, repo=repo):
        guard = S._ACTIVE_VERIFICATION.get().roster_guard
        if guard.fd is None:
            pytest.skip("kernel watches unavailable; production retains ordinary scanning")
        count = len(scans)
        for _ in range(8):
            S.verify_bundle_snapshot(ws, bundle, repo=repo)
        assert len(scans) == count
    assert len(scans) == count and guard.fd is None  # final payload read reuses only the live roster
    assert scans[-1] == root


@pytest.mark.parametrize("change", ["add", "resize_alias", "symlink"])
def test_membership_mutations_including_external_hardlinks_refuse(frozen, tmp_path, change):
    ws, bundle, repo, root, payload = frozen
    with pytest.raises(RuntimeError, match="changed|differs|indirect"):
        with S.verified_snapshot_scope(ws, bundle, repo=repo):
            if change == "add":
                (root / "extra").write_bytes(b"unexpected")
            elif change == "symlink":
                payload.unlink()
                payload.symlink_to(tmp_path / "elsewhere")
            else:
                alias = tmp_path / "outside-alias"
                os.link(payload, alias)
                alias.chmod(0o600)
                alias.write_bytes(b"longer changed bytes through another name")
            S.verify_bundle_snapshot(ws, bundle, repo=repo)


def test_unavailable_watch_retains_complete_scan(frozen, monkeypatch):
    ws, bundle, repo, _, _ = frozen
    monkeypatch.setattr(S._RosterGuard, "unchanged", lambda self: False)
    with S.verified_snapshot_scope(ws, bundle, repo=repo):
        assert S.verify_bundle_snapshot(ws, bundle, repo=repo)["n_files"] == 1


def test_mount_change_invalidates_shortcut(frozen, monkeypatch):
    ws, bundle, repo, _, _ = frozen
    with S.verified_snapshot_scope(ws, bundle, repo=repo):
        guard = S._ACTIVE_VERIFICATION.get().roster_guard
        if guard.fd is None:
            pytest.skip("kernel watches unavailable")
        guard.mounts += b"changed mount namespace"
        assert guard.unchanged() is False and guard.fd is None
        assert S.verify_bundle_snapshot(ws, bundle, repo=repo)["n_files"] == 1
