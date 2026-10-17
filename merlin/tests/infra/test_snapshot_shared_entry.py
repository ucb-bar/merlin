"""One grade-wide snapshot entry serves its scopes; every scope keeps a fresh final read."""

import os
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from merlin.common import bundle_snapshot as S
from merlin.common import cpu_pool as C
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
    return ws, bundle, repo, payload


def _payload_reads(monkeypatch, payload):
    reads = []
    digest = S.hashlib.file_digest

    def measured(stream, *args, **kwargs):
        if str(stream.name) == str(payload):
            reads.append(threading.current_thread().name)
        return digest(stream, *args, **kwargs)

    monkeypatch.setattr(S.hashlib, "file_digest", measured)
    return reads


def _entry(ws, bundle, repo):
    entry = S.shared_entry_verification(ws, bundle, repo=repo)
    if entry is None:
        pytest.skip("kernel watches unavailable; production keeps every scope's full entry read")
    return entry


def test_scopes_on_other_threads_start_from_the_entry_and_read_only_their_final(frozen, monkeypatch):
    ws, bundle, repo, payload = frozen
    reads = _payload_reads(monkeypatch, payload)
    entry = _entry(ws, bundle, repo)
    assert len(reads) == 1
    reads.clear()

    def scope(index):
        with S.verified_snapshot_scope(ws, bundle, repo=repo, entry=entry) as document:
            # Nested reads inside the scope reuse the entry's digest, as within any scope.
            assert S.verify_bundle_snapshot(ws, bundle, repo=repo) == document
            return index

    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="capsule") as pool:
        assert sorted(pool.map(scope, range(3))) == [0, 1, 2]
    assert len(reads) == 3  # exactly one fresh final read per scope, none at entry
    assert entry.guard.fd is not None  # a scope's view never closes the grade's guard
    entry.close()
    assert entry.guard.fd is None


def test_entry_needs_the_guard_selection(frozen, monkeypatch):
    ws, bundle, repo, _ = frozen
    monkeypatch.setenv("MERLIN_SNAPSHOT_LEASE_CACHE", "0")
    assert S.shared_entry_verification(ws, bundle, repo=repo) is None


def test_a_change_after_the_entry_ends_reuse_and_is_refused(frozen, monkeypatch):
    ws, bundle, repo, payload = frozen
    reads = _payload_reads(monkeypatch, payload)
    entry = _entry(ws, bundle, repo)
    metadata = payload.stat()
    payload.write_bytes(b"ORIGINAL BYTES")  # same size, mtime restored: only content changed
    os.utime(payload, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
    reads.clear()
    with pytest.raises(RuntimeError, match="content differs"):
        with S.verified_snapshot_scope(ws, bundle, repo=repo, entry=entry):
            pass
    assert reads  # the scope read the payload afresh instead of reusing the entry
    assert entry.guard.fd is None


def test_concurrent_scope_entries_lease_one_cpu_at_a_time(frozen, monkeypatch, tmp_path):
    ws, bundle, repo, _ = frozen
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) < 2:
        pytest.skip("requires two available CPUs")
    cpus, slots = (cpus[0], cpus[-1]), tmp_path / "slots"
    monkeypatch.setenv("MERLIN_COMPUTE_CPUS", ",".join(map(str, cpus)))
    monkeypatch.setenv("MERLIN_COMPUTE_CPU_SLOTS", str(slots))
    scan, together, held = S._scanned_members, threading.Barrier(2, timeout=0.5), []

    def slow_scan(root):
        try:
            together.wait()  # passes only if two threads scan at the same time
        except threading.BrokenBarrierError:
            pass
        held.append(len(cpus) - C._free_slots(slots, cpus, fresh=True))
        return scan(root)

    def scope(_):
        with S.verified_snapshot_scope(ws, bundle, repo=repo):
            pass

    monkeypatch.setattr(S, "_scanned_members", slow_scan)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(scope, range(4)))
    # Every leased stage (payload digest, roster scan, guard setup) waits at the admission gate.
    assert held and max(held) == 1
