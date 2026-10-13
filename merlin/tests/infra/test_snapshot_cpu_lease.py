"""Snapshot payload preparation owns a bounded shared CPU stage, never its caller."""

import fcntl
import os

import pytest

from merlin.common import bundle_snapshot as S
from merlin.common import cpu_pool as C
from merlin.targetgen.sandbox import bwrap as B


@pytest.fixture
def frozen(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "input"
    source.write_bytes(b"original payload")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    bundle = {"allowed": [{"path": str(source), "mode": "ro"}], "host_inputs": [], "denied": []}
    B.materialize_bundle_inputs(workspace, bundle, repo=repo)
    cpu = min(os.sched_getaffinity(0))
    slots = tmp_path / "slots"
    monkeypatch.setenv("MERLIN_COMPUTE_CPUS", str(cpu))
    monkeypatch.setenv("MERLIN_COMPUTE_CPU_SLOTS", str(slots))
    monkeypatch.setenv("MERLIN_SNAPSHOT_LEASE_CACHE", "0")
    return workspace, bundle, repo, cpu, slots


def test_snapshot_metadata_hash_owns_only_its_stage(frozen, monkeypatch):
    workspace, bundle, repo, cpu, slots = frozen
    original_affinity = os.sched_getaffinity(0)
    content, mapping = S.snapshot_content, S.snapshot_path
    observed = []

    def path(*args):
        assert C._ACTIVE_LEASE.get() is None
        return mapping(*args)

    def hash_stage(root):
        owner = C._ACTIVE_LEASE.get()
        assert owner is not None and owner.active
        assert owner.pid == os.getpid() and os.sched_getaffinity(0) == {cpu}
        with (slots / f"cpu-{cpu}.lock").open("r+") as competing:
            with pytest.raises(BlockingIOError):
                fcntl.flock(competing, fcntl.LOCK_EX | fcntl.LOCK_NB)
        observed.append(root)
        return content(root)

    monkeypatch.setattr(S, "snapshot_path", path)
    monkeypatch.setattr(S, "snapshot_content", hash_stage)
    assert S.verify_bundle_snapshot(workspace, bundle, repo=repo)["n_files"] == 1
    assert observed == [S.bundle_snapshot_root(workspace)]
    assert C._ACTIVE_LEASE.get() is None
    assert os.sched_getaffinity(0) == original_affinity


def test_snapshot_stage_reuses_exact_outer_step_lease(frozen, monkeypatch):
    workspace, bundle, repo, cpu, _ = frozen
    content = S.snapshot_content
    with C.shared_compute_pool(timeout_s=0):
        owner = C._ACTIVE_LEASE.get()

        def hash_stage(root):
            assert C._ACTIVE_LEASE.get() is owner
            assert os.sched_getaffinity(0) == {cpu}
            return content(root)

        monkeypatch.setattr(S, "snapshot_content", hash_stage)
        S.verify_bundle_snapshot(workspace, bundle, repo=repo)
        assert C._ACTIVE_LEASE.get() is owner and owner.active
    assert C._ACTIVE_LEASE.get() is None


def test_hash_and_content_refusals_release_slot(frozen, monkeypatch):
    workspace, bundle, repo, cpu, slots = frozen
    original_affinity = os.sched_getaffinity(0)
    content = S.snapshot_content

    def refuse(root):
        assert C._ACTIVE_LEASE.get() is not None
        raise RuntimeError("payload read refused")

    monkeypatch.setattr(S, "snapshot_content", refuse)
    with pytest.raises(RuntimeError, match="payload read refused"):
        S.verify_bundle_snapshot(workspace, bundle, repo=repo)
    assert C._ACTIVE_LEASE.get() is None
    assert os.sched_getaffinity(0) == original_affinity
    with C.shared_compute_pool(timeout_s=0):
        pass

    monkeypatch.setattr(S, "snapshot_content", content)
    payload = S.snapshot_path(S.bundle_snapshot_root(workspace), repo / "input", repo)
    payload.chmod(0o600)
    payload.write_bytes(b"ORIGINAL PAYLOAD")
    with pytest.raises(RuntimeError, match="content differs"):
        S.verify_bundle_snapshot(workspace, bundle, repo=repo)
    assert C._ACTIVE_LEASE.get() is None
    assert os.sched_getaffinity(0) == original_affinity
    with (slots / f"cpu-{cpu}.lock").open("r+") as competing:
        fcntl.flock(competing, fcntl.LOCK_EX | fcntl.LOCK_NB)
