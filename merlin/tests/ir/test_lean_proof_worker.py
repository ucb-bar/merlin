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


def _mark_then_hang(root):
    (root / "callback-started").touch()
    time.sleep(3600)


def test_callback_cannot_start_without_parent_ownership_ack(tmp_path):
    directory, request = proof_worker._prepare(_mark_then_hang, (tmp_path,), tmp_path, time.monotonic() + 5)
    worker = proof_worker.start_owned(
        proof_worker._command(directory, request),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 5
        while not (directory / "ready.json").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert proof_worker._json(directory / "ready.json") == proof_worker._identity(request, worker.pid)
        assert not (tmp_path / "callback-started").exists()
        proof_worker._publish(directory / "go.json", {"kind": "not-go"})
        assert worker.wait(timeout=5) != 0
        assert not (tmp_path / "callback-started").exists()
    finally:
        proof_worker.stop_owned(worker)
        proof_worker.close_owned_pipes(worker)


def _partial_result(root, fault):
    directory = next(root.glob("proof-worker-*"))
    target = directory / "result.json"
    if fault == "pending":
        target = target.with_suffix(".json.pending")
        target.write_text('{"incomplete":')
    elif fault == "truncated":
        target.write_text('{"incomplete":')
    elif fault == "fifo":
        os.mkfifo(target)
    elif fault == "symlink":
        target.symlink_to(root / "callback-started")
        (root / "callback-started").touch()
    time.sleep(3600)


@pytest.mark.parametrize("fault", ["pending", "truncated", "fifo", "symlink"])
def test_partial_or_nonregular_results_cannot_block_the_supervisor(tmp_path, fault):
    expected = TimeoutError if fault == "pending" else proof_worker.WorkerFailure
    with pytest.raises(expected):
        proof_worker.run(_partial_result, tmp_path, fault, timeout=1, logs=tmp_path)
    cleanup = proof_worker._json(next(tmp_path.glob("proof-worker-*/cleanup.json")))
    assert cleanup["direct_child_reaped"]


def test_unreapable_child_and_result_wait_have_only_bounded_waits(tmp_path, monkeypatch):
    waits, signals = [], []

    class Unreapable:
        pid = 123456789
        returncode = None
        stdin = stdout = stderr = None

        def wait(self, *, timeout):
            waits.append(timeout)
            raise subprocess.TimeoutExpired("unreapable", timeout)

        def poll(self):
            return None

    def start(*args, **kwargs):
        directory = next(tmp_path.glob("proof-worker-*"))
        request = proof_worker._json(directory / "request.json")
        identity = proof_worker._identity(request, Unreapable.pid)
        proof_worker._publish(directory / "ready.json", identity)
        proof_worker._publish(directory / "result.json", {**identity, "kind": "result", "value": "untrusted"})
        return Unreapable()

    monkeypatch.setattr(proof_worker, "start_owned", start)
    monkeypatch.setattr(os, "killpg", lambda *args: signals.append(args))
    with pytest.raises(TimeoutError, match="did not exit after its result") as error:
        proof_worker.run(_success, "unused", timeout=5, logs=tmp_path)
    assert 0 < waits[0] <= 5
    assert waits[1:] == [0.2, 1.0]
    assert [pid for pid, _ in signals] == [Unreapable.pid, Unreapable.pid]
    cleanup = proof_worker._json(next(tmp_path.glob("proof-worker-*/cleanup.json")))
    assert cleanup["pid"] == cleanup["group"] == Unreapable.pid
    assert cleanup["direct_child_reaped"] is False
    assert "123456789" in "\n".join(error.value.__notes__)


def _nonjson():
    return object()


def test_returned_data_is_json_and_never_unpickled(tmp_path):
    with pytest.raises(proof_worker.WorkerFailure, match="not JSON serializable"):
        proof_worker.run(_nonjson, timeout=5, logs=tmp_path)


def test_bootstrap_uses_parent_worker_despite_foreign_pythonpath(tmp_path, monkeypatch):
    foreign = tmp_path / "foreign"
    fake = foreign / "merlin" / "verify" / "lean"
    fake.mkdir(parents=True)
    for directory in (foreign / "merlin", foreign / "merlin" / "verify", fake):
        (directory / "__init__.py").write_text("")
    (fake / "proof_worker.py").write_text("raise RuntimeError('foreign worker must never load')")
    monkeypatch.setenv("PYTHONPATH", str(foreign))
    result = proof_worker.run(_success, "parent-worker", timeout=5, logs=tmp_path)
    assert result["value"] == "parent-worker"


def test_bootstrap_source_pin_is_enforced_under_python_optimization(tmp_path, monkeypatch):
    command = proof_worker._command

    def wrong_pin(*args):
        argv = command(*args)
        argv[-2] = "0" * 64
        return argv

    monkeypatch.setattr(proof_worker, "_command", wrong_pin)
    monkeypatch.setenv("PYTHONOPTIMIZE", "1")
    with pytest.raises(proof_worker.WorkerFailure, match="without a result"):
        proof_worker.run(_mark_then_hang, tmp_path, timeout=5, logs=tmp_path)
    assert not (tmp_path / "callback-started").exists()
    assert "bootstrap source changed" in (tmp_path / "proof.stderr.log").read_text()
