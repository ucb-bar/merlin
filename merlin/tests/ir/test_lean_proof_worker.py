"""Real spawned-worker lifecycle controls, including owned checker descendants."""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from merlin.verify.lean import proof_worker


def _success(value):
    return {"value": value, "pid": os.getpid(), "group": os.getpgrp()}


def _raise():
    raise ValueError("deliberate proof refusal")


def _crash():
    os._exit(17)


def _hung_checker(root):
    # A real child intentionally ignores SIGTERM, so killing only the proof
    # worker would leave the checker running. It belongs solely to this worker.
    checker = subprocess.Popen(
        [sys.executable, "-c", "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(3600)"]
    )
    (root / "checker.pid").write_text(str(checker.pid))
    (root / "worker.pid").write_text(str(os.getpid()))
    time.sleep(3600)


def _running(pid):
    stat = Path(f"/proc/{pid}/stat")
    try:
        state = stat.read_text().rpartition(")")[2].split()[0]
    except FileNotFoundError:
        return False
    # A reparented zombie has terminated; it cannot execute or retain resources.
    return state != "Z"


def test_spawned_success_uses_a_private_group(tmp_path):
    result = proof_worker.run(_success, "evidence", timeout=5, logs=tmp_path)
    assert result["value"] == "evidence"
    assert result["pid"] == result["group"]
    assert result["pid"] != os.getpid()
    assert not _running(result["pid"])


def test_worker_error_preserves_traceback_and_refuses(tmp_path):
    with pytest.raises(proof_worker.WorkerFailure, match="ValueError: deliberate proof refusal"):
        proof_worker.run(_raise, timeout=5, logs=tmp_path)
    assert "raise ValueError" in (tmp_path / "proof.stderr.log").read_text()


def test_worker_crash_never_returns_evidence(tmp_path):
    with pytest.raises(proof_worker.WorkerFailure, match="without a result"):
        proof_worker.run(_crash, timeout=5, logs=tmp_path)


def test_hung_checker_and_its_worker_are_terminated_on_deadline(tmp_path):
    sibling = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        with pytest.raises(TimeoutError, match="explicit deadline"):
            proof_worker.run(_hung_checker, tmp_path, timeout=3, logs=tmp_path)
        checker = int((tmp_path / "checker.pid").read_text())
        worker = int((tmp_path / "worker.pid").read_text())
        assert not _running(worker)
        deadline = time.monotonic() + 1
        while _running(checker) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not _running(checker)
        # This separately owned sibling must survive cleanup of the worker's group.
        assert sibling.poll() is None
    finally:
        sibling.terminate()
        sibling.wait(timeout=5)


def test_expired_deadline_does_not_start_a_worker(tmp_path):
    with pytest.raises(TimeoutError, match="before worker entry"):
        proof_worker.run(_success, None, timeout=0, logs=tmp_path)
    assert not list(tmp_path.iterdir())
