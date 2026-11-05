"""Two occupied slots admit their own trusted RPC children without a new lock."""

from __future__ import annotations

import contextvars
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from merlin.common import cpu_pool as C

OWNER = "a" * 64
CHILD = """
import json,os,sys
from merlin.common.cpu_pool import CompilerCPUTransfer,borrow_compiler_cpu,shared_compute_pool
transfer=CompilerCPUTransfer(*(int(value) for value in sys.argv[1:4]))
with borrow_compiler_cpu(transfer,owner_sha256=sys.argv[4]):
    with shared_compute_pool(timeout_s=.2):
        print(json.dumps(sorted(os.sched_getaffinity(0))),flush=True)
"""


def _environment(cpus, root):
    source = Path(C.__file__).parents[2]
    return {**os.environ, "PYTHONPATH": str(source), "PYTHONDONTWRITEBYTECODE": "1",
            "MERLIN_COMPUTE_CPUS": ",".join(map(str, cpus)), "MERLIN_COMPUTE_CPU_SLOTS": str(root)}


def _child(transfer, env, owner=OWNER, script=CHILD):
    return subprocess.Popen([sys.executable, "-c", script, *map(str, transfer.pass_fds), owner],
                            pass_fds=transfer.pass_fds, env=env, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def test_two_fully_occupied_slots_borrow_same_cpus_without_deadlock(tmp_path):
    available = sorted(os.sched_getaffinity(0))
    if len(available) < 2:
        pytest.skip("requires two available CPU slots")
    cpus = (available[0], available[-1])
    root, barrier = tmp_path / "slots", threading.Barrier(2)
    env = _environment(cpus, root)

    def request(_):
        before = os.sched_getaffinity(0)
        with C.selected_pool(cpus, slot_root=root, timeout_s=2):
            assigned = os.sched_getaffinity(0)
            barrier.wait(timeout=2)  # every slot is held before either RPC starts
            with C.transfer_compiler_cpu(owner_sha256=OWNER) as transfer:
                child = _child(transfer, env)
                stdout, stderr = child.communicate(timeout=3)
                assert child.returncode == 0, stderr
                assert set(json.loads(stdout)) == assigned
        assert os.sched_getaffinity(0) == before
        return assigned

    with ThreadPoolExecutor(max_workers=2) as executor:
        assigned = list(executor.map(request, range(2)))
    assert set.union(*assigned) == set(cpus)


def test_missing_foreign_and_copied_thread_issuers_refuse(tmp_path):
    with C.transfer_compiler_cpu(owner_sha256=OWNER) as transfer:
        assert transfer is None
    cpu = next(iter(os.sched_getaffinity(0)))
    with C.selected_pool((cpu,), slot_root=tmp_path / "slots"):
        copied = contextvars.copy_context()
        with ThreadPoolExecutor(max_workers=1) as executor:
            with pytest.raises(RuntimeError, match="original thread owner"):
                executor.submit(copied.run, lambda: C.transfer_compiler_cpu(owner_sha256=OWNER).__enter__()).result()
        with C.transfer_compiler_cpu(owner_sha256=OWNER) as transfer:
            child = _child(transfer, _environment((cpu,), tmp_path / "slots"), "b" * 64)
            _stdout, stderr = child.communicate(timeout=3)
            assert child.returncode != 0 and "original owner" in stderr


def test_borrower_refuses_after_original_transfer_lifetime_ends(tmp_path):
    cpu = next(iter(os.sched_getaffinity(0)))
    root = tmp_path / "slots"
    script = """
import os,sys
from merlin.common.cpu_pool import CompilerCPUTransfer,borrow_compiler_cpu,shared_compute_pool
t=CompilerCPUTransfer(*(int(value) for value in sys.argv[1:4]))
with borrow_compiler_cpu(t,owner_sha256=sys.argv[4]):
    print('ready',flush=True)
    sys.stdin.readline()
    with shared_compute_pool(timeout_s=.2):
        print('INCORRECT ACCEPTANCE',flush=True)
"""
    with C.selected_pool((cpu,), slot_root=root):
        with C.transfer_compiler_cpu(owner_sha256=OWNER) as transfer:
            child = _child(transfer, _environment((cpu,), root), script=script)
            assert child.stdout.readline().strip() == "ready"
        stdout, stderr = child.communicate("continue\n", timeout=3)
        assert child.returncode != 0 and "no longer live" in stderr
        assert "INCORRECT ACCEPTANCE" not in stdout


def test_replaced_slot_refuses_before_transfer_publication(tmp_path):
    cpu = next(iter(os.sched_getaffinity(0)))
    root = tmp_path / "slots"
    with C.selected_pool((cpu,), slot_root=root):
        slot = root / f"cpu-{cpu}.lock"
        slot.unlink()
        slot.touch()
        with pytest.raises(RuntimeError, match="exact ordinary slot"):
            with C.transfer_compiler_cpu(owner_sha256=OWNER):
                pytest.fail("replaced slot was loaned")


def test_missing_unsealed_and_foreign_descriptor_transfers_refuse(tmp_path):
    with pytest.raises(RuntimeError, match="exact original transfer"):
        with C.borrow_compiler_cpu(None, owner_sha256=OWNER):
            pytest.fail("missing transfer accepted")
    cpu = next(iter(os.sched_getaffinity(0)))
    root = tmp_path / "slots"
    with C.selected_pool((cpu,), slot_root=root):
        with C.transfer_compiler_cpu(owner_sha256=OWNER) as transfer:
            unsealed = os.memfd_create("unsealed-test", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
            os.write(unsealed, os.pread(transfer.metadata_fd, 8192, 0))
            try:
                forged = C.CompilerCPUTransfer(unsealed, transfer.slot_fd, transfer.lifetime_fd)
                child = _child(forged, _environment((cpu,), root))
                _stdout, stderr = child.communicate(timeout=3)
                assert child.returncode != 0 and "not immutable" in stderr
            finally:
                os.close(unsealed)


def test_unissued_and_inactive_parent_leases_refuse(tmp_path):
    cpu = next(iter(os.sched_getaffinity(0)))
    with C.selected_pool((cpu,), slot_root=tmp_path / "slots"):
        actual = C._ACTIVE_LEASE.get()
        fake = C._Lease(actual.pid, actual.thread, actual.root, actual.cpu, actual.fd)
        token = C._ACTIVE_LEASE.set(fake)
        try:
            with pytest.raises(RuntimeError, match="original thread owner"):
                with C.transfer_compiler_cpu(owner_sha256=OWNER):
                    pytest.fail("unissued owner accepted")
        finally:
            C._ACTIVE_LEASE.reset(token)
        actual.active = False
        with pytest.raises(RuntimeError, match="original thread owner"):
            with C.transfer_compiler_cpu(owner_sha256=OWNER):
                pytest.fail("inactive owner accepted")
