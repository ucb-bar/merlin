"""Snapshot payload preparation owns a bounded shared CPU stage, never its caller."""

import contextlib
import fcntl
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

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


def test_same_process_metadata_waiters_do_not_reserve_other_slots(frozen, monkeypatch):
    workspace, bundle, repo, _, slots = frozen
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) < 2:
        pytest.skip("requires two available CPUs")
    cpus = (cpus[0], cpus[-1])
    monkeypatch.setenv("MERLIN_COMPUTE_CPUS", ",".join(map(str, cpus)))
    content, admission = S.snapshot_content, S._snapshot_read_admission
    admitted, release, started = threading.Event(), threading.Event(), threading.Event()
    count, checked, lock = 0, [], threading.Lock()

    @contextlib.contextmanager
    def waiting():
        nonlocal count
        with lock:
            count += 1
            if count == 4:
                started.set()
        with admission():
            yield

    def slow_metadata(root):
        checked.append(root)
        admitted.set()
        assert release.wait(3)
        return content(root)

    monkeypatch.setattr(S, "_snapshot_read_admission", waiting)
    monkeypatch.setattr(S, "snapshot_content", slow_metadata)
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(S.verify_bundle_snapshot, workspace, bundle, repo=repo) for _ in range(4)]
        try:
            assert admitted.wait(3) and started.wait(3)
            # One real CPU is held by the active verifier; all other waiters
            # reached admission without taking the second exclusive slot.
            with C.selected_pool((cpus[-1],), slot_root=slots, timeout_s=0):
                assert os.sched_getaffinity(0) == {cpus[-1]}
        finally:
            release.set()
        assert all(future.result(timeout=3)["n_files"] == 1 for future in futures)
    assert len(checked) == 4  # independent actual payload verification, no reuse


def test_existing_lease_bypasses_waiting_admission_without_slot_cycle(frozen):
    workspace, bundle, repo, cpu, slots = frozen
    # A subprocess bounds a broken lock/slot implementation instead of leaving
    # deadlocked test threads in the test runner. No compiler or simulator runs.
    script = """
import json,sys,threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from merlin.common import cpu_pool as C, bundle_snapshot as S
workspace,repo=map(Path,sys.argv[1:3]); bundle=json.loads(sys.argv[3])
held,waiting=threading.Event(),threading.Event()
admission=S._snapshot_read_admission
def owner():
    with C.shared_compute_pool(timeout_s=2):
        held.set()
        assert waiting.wait(2)
        S.verify_bundle_snapshot(workspace,bundle,repo=repo)
def waiter():
    assert held.wait(2)
    with admission():
        waiting.set()
        with C.shared_compute_pool(timeout_s=2):
            S.verify_bundle_snapshot(workspace,bundle,repo=repo)
with ThreadPoolExecutor(max_workers=2) as executor:
    futures=[executor.submit(owner),executor.submit(waiter)]
    for future in futures: future.result(timeout=3)
print('both completed')
"""
    import json

    env = {**os.environ, "PYTHONPATH": str(Path(S.__file__).parents[2]), "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run(
        [sys.executable, "-c", script, str(workspace), str(repo), json.dumps(bundle)],
        env=env,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "both completed"
    with C.selected_pool((cpu,), slot_root=slots, timeout_s=0):
        pass


def test_admission_is_process_local_with_genuine_borrowed_cpu(frozen):
    workspace, bundle, repo, cpu, slots = frozen
    script = """
import json,sys
from pathlib import Path
from merlin.common import cpu_pool as C, bundle_snapshot as S
t=C.CompilerCPUTransfer(*(int(value) for value in sys.argv[1:4]))
with C.borrow_compiler_cpu(t,owner_sha256='a'*64):
    assert C.current_thread_owns_cpu_lease()
    S.verify_bundle_snapshot(Path(sys.argv[4]),json.loads(sys.argv[6]),repo=Path(sys.argv[5]))
print('borrowed verification completed')
"""
    import json

    env = {**os.environ, "PYTHONPATH": str(Path(S.__file__).parents[2]), "PYTHONDONTWRITEBYTECODE": "1"}
    # A live parent admission lock is never exported or consulted by its
    # independent broker process. Its exact CPU slot is privately transferred.
    with S._snapshot_read_admission(), C.shared_compute_pool(timeout_s=0):
        with C.transfer_compiler_cpu(owner_sha256="a" * 64) as transfer:
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    script,
                    *map(str, transfer.pass_fds),
                    str(workspace),
                    str(repo),
                    json.dumps(bundle),
                ],
                pass_fds=transfer.pass_fds,
                env=env,
                capture_output=True,
                text=True,
                timeout=5,
            )
            assert result.returncode == 0, result.stderr
            assert result.stdout.strip() == "borrowed verification completed"
    with C.selected_pool((cpu,), slot_root=slots, timeout_s=0):
        pass
