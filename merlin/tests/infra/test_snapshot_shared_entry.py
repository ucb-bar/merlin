"""One grade-wide snapshot entry serves its scopes; every scope keeps a fresh final read."""

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context

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


def test_implicit_nested_entries_have_distinct_tokens_and_fresh_finals(frozen, monkeypatch):
    ws, bundle, repo, payload = frozen
    entry = _entry(ws, bundle, repo)
    reads = _payload_reads(monkeypatch, payload)
    try:
        with S.shared_entry_scope(entry):
            with S.verified_snapshot_scope(ws, bundle, repo=repo):
                outer = S._ACTIVE_VERIFICATION.get()
                with S.verified_snapshot_scope(ws, bundle, repo=repo):
                    assert S._ACTIVE_VERIFICATION.get() is not outer
                    assert not reads
                assert S._ACTIVE_VERIFICATION.get() is outer
                assert len(reads) == 1
            assert len(reads) == 2
        assert S._SHARED_ENTRY.get() is None and S._FINAL_ROSTER.get() is None
        with S.verified_snapshot_scope(ws, bundle, repo=repo):
            pass
        assert len(reads) == 4  # Selection ended; independent entry and final.
    finally:
        entry.close()


def test_independent_consumer_clears_implicit_entry_and_guarded_roster(frozen, monkeypatch):
    ws, bundle, repo, payload = frozen
    entry = _entry(ws, bundle, repo)
    reads = _payload_reads(monkeypatch, payload)
    try:
        with S.shared_entry_scope(entry):
            with S.verified_snapshot_scope(ws, bundle, repo=repo):
                with pytest.raises(ValueError, match="consumer failed"):
                    with S.independent_snapshot_reads():
                        assert S._SHARED_ENTRY.get() is None and S._FINAL_ROSTER.get() is None
                        with S.verified_snapshot_scope(ws, bundle, repo=repo):
                            raise ValueError("consumer failed")
                assert len(reads) == 2
            assert len(reads) == 3
    finally:
        entry.close()


@pytest.mark.parametrize("consumer", ["thread", "fork"])
def test_implicit_entry_context_cannot_cross_thread_or_process(frozen, consumer):
    ws, bundle, repo, _ = frozen
    entry = _entry(ws, bundle, repo)

    def verify():
        assert S._selected_entry() is None
        with S.verified_snapshot_scope(ws, bundle, repo=repo):
            assert type(S._ACTIVE_VERIFICATION.get().roster_guard) is S._RosterGuard
        return True

    try:
        with S.shared_entry_scope(entry):
            if consumer == "thread":
                with ThreadPoolExecutor(max_workers=1) as pool:
                    assert pool.submit(copy_context().run, verify).result(timeout=10)
            else:
                pid = os.fork()
                if pid == 0:
                    try:
                        verify()
                    except BaseException:
                        os._exit(1)
                    os._exit(0)
                _, status = os.waitpid(pid, 0)
                assert os.waitstatus_to_exitcode(status) == 0
        assert entry.guard.unchanged()  # Child never closes the parent's inherited guard.
    finally:
        entry.close()


@pytest.mark.parametrize("changed", ["policy", "root", "closed_guard"])
def test_implicit_selection_falls_back_for_different_or_closed_entry(frozen, monkeypatch, tmp_path, changed):
    ws, bundle, repo, payload = frozen
    entry = _entry(ws, bundle, repo)
    if changed == "policy":
        bundle = {**bundle, "denied": [{"path": "different-private-selection"}]}
    elif changed == "root":
        ws = tmp_path / "another" / "workspace"
        ws.mkdir(parents=True)
        B.materialize_bundle_inputs(ws, bundle, repo=repo)
        payload = S.snapshot_path(S.bundle_snapshot_root(ws), repo / "source", repo)
    else:
        entry.close()
    reads = _payload_reads(monkeypatch, payload)
    try:
        with S.shared_entry_scope(entry), S.verified_snapshot_scope(ws, bundle, repo=repo):
            assert len(reads) == 1
        assert len(reads) == 2
    finally:
        entry.close()


def test_forged_or_expired_context_does_not_select_a_shared_entry(frozen):
    ws, bundle, repo, _ = frozen
    entry = _entry(ws, bundle, repo)
    forged = object.__new__(S.SharedEntry)
    with pytest.raises(ValueError, match="original live issuer"):
        with S.shared_entry_scope(forged):
            pytest.fail("unissued entry admitted")
    with S.shared_entry_scope(entry):
        copied = copy_context()

    def expired():
        assert S._selected_entry() is None
        with S.verified_snapshot_scope(ws, bundle, repo=repo):
            assert type(S._ACTIVE_VERIFICATION.get().roster_guard) is S._RosterGuard

    try:
        copied.run(expired)
    finally:
        entry.close()


def test_mutation_during_final_hash_restarts_complete_roster_and_refuses(frozen, monkeypatch):
    ws, bundle, repo, payload = frozen
    entry = _entry(ws, bundle, repo)
    original = S.hashlib.file_digest
    changed = False

    def mutate_after_hash(stream, *args, **kwargs):
        nonlocal changed
        result = original(stream, *args, **kwargs)
        if not changed and str(stream.name) == str(payload):
            changed = True
            payload.write_bytes(b"ORIGINAL BYTES")
        return result

    monkeypatch.setattr(S.hashlib, "file_digest", mutate_after_hash)
    try:
        with pytest.raises(RuntimeError, match="content differs"):
            with S.shared_entry_scope(entry), S.verified_snapshot_scope(ws, bundle, repo=repo):
                pass
        assert changed and entry.guard.fd is None
        assert S._ACTIVE_VERIFICATION.get() is None and S._FINAL_ROSTER.get() is None
    finally:
        entry.close()


@pytest.mark.parametrize("selection", ["policy", "marker"])
def test_selection_mutated_during_final_read_cannot_publish(frozen, monkeypatch, tmp_path, selection):
    ws, bundle, repo, payload = frozen
    entry = _entry(ws, bundle, repo)
    original, changed = S.hashlib.file_digest, False
    output = tmp_path / "accepted.json"

    def mutate(stream, *args, **kwargs):
        nonlocal changed
        result = original(stream, *args, **kwargs)
        if not changed and str(stream.name) == str(payload):
            changed = True
            if selection == "policy":
                bundle["denied"].append({"path": "different-private-selection"})
            else:
                marker = S.bundle_snapshot_root(ws) / S.MARKER
                marker.chmod(0o600)
                marker.write_bytes(marker.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(S.hashlib, "file_digest", mutate)
    try:
        with pytest.raises(RuntimeError, match="policy or marker changed"):
            with S.shared_entry_scope(entry), S.verified_snapshot_scope(ws, bundle, repo=repo):
                pass
            output.write_text("must not publish")
        assert changed and not output.exists()
        assert S._ACTIVE_VERIFICATION.get() is None and S._FINAL_ROSTER.get() is None
    finally:
        entry.close()


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


def test_identical_bytes_behind_a_replaced_parent_alias_are_still_refused(frozen, tmp_path):
    ws, bundle, repo, payload = frozen
    entry = _entry(ws, bundle, repo)
    original_parent = payload.parent
    moved = tmp_path / "moved-identical-inputs"
    original_parent.rename(moved)
    original_parent.symlink_to(moved, target_is_directory=True)
    try:
        with pytest.raises(RuntimeError, match="indirect|changed|differs"):
            with S.verified_snapshot_scope(ws, bundle, repo=repo, entry=entry):
                pass
    finally:
        entry.close()


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


def test_canonical_metadata_is_admitted_before_capsules_fan_out(frozen, monkeypatch, tmp_path):
    ws, bundle, repo, _ = frozen
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) < 2:
        pytest.skip("requires two available CPUs")
    monkeypatch.setenv("MERLIN_COMPUTE_CPUS", f"{cpus[0]},{cpus[-1]}")
    monkeypatch.setenv("MERLIN_COMPUTE_CPU_SLOTS", str(tmp_path / "metadata-slots"))
    direct = S._require_direct_snapshot_member
    lock, active, observed = threading.Lock(), 0, []

    def measured(*args):
        nonlocal active
        with lock:
            active += 1
            observed.append(active)
        try:
            time.sleep(0.005)  # Concurrent callers would overlap this filesystem wait.
            return direct(*args)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(S, "_require_direct_snapshot_member", measured)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: S.verify_bundle_snapshot(ws, bundle, repo=repo), range(4)))
    assert observed and max(observed) == 1


def test_exact_nested_inspection_bypasses_global_admission_but_final_rechecks(frozen, monkeypatch):
    ws, bundle, repo, payload = frozen
    entry = _entry(ws, bundle, repo)
    calls = []
    admission = S._snapshot_read_admission
    from contextlib import contextmanager
    @contextmanager
    def counted():
        calls.append(threading.current_thread().name)
        with admission():
            yield
    try:
        with S.verified_snapshot_scope(ws,bundle,repo=repo,entry=entry):
            monkeypatch.setattr(S,'_snapshot_read_admission',counted)
            for _ in range(3):
                assert S.verify_bundle_snapshot(ws,bundle,repo=repo)==entry.document
            assert calls==[]
        assert calls  # Actual command completion still performs fresh admission.
    finally:
        entry.close()


@pytest.mark.parametrize('change',['payload','marker','policy','replacement'])
def test_fast_nested_inspection_cannot_hide_scope_mutation(frozen, change):
    ws,bundle,repo,payload=frozen
    entry=_entry(ws,bundle,repo)
    try:
        with pytest.raises(RuntimeError):
            with S.verified_snapshot_scope(ws,bundle,repo=repo,entry=entry):
                S.verify_bundle_snapshot(ws,bundle,repo=repo)
                if change=='payload':payload.write_bytes(b'changed bytes!')
                elif change=='marker':
                    marker=S.bundle_snapshot_root(ws)/S.MARKER
                    marker.chmod(0o600);marker.write_bytes(marker.read_bytes()+b' ')
                elif change=='policy':bundle['allowed']=[]
                else:
                    payload.unlink();payload.symlink_to(repo/'source')
                S.verify_bundle_snapshot(ws,bundle,repo=repo)
    finally:
        entry.close()
