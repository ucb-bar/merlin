"""Subprocess proof work with atomic bounded messages and owned-group cleanup.

Only a parent-authored data snapshot crosses this boundary, never live MLIR.
The input pickle is trusted local parent data; returned evidence is bounded JSON
and is independently revalidated by the caller. No multiprocessing finalizer,
framed pipe receive, or unbounded process wait participates in this protocol.
Ordinary filesystem IO remains subject to the host OS, as does pending SIGKILL.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import stat
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path

from merlin.common.owned_process import close_owned_pipes, start_owned, stop_owned

SCHEMA = "merlin.lean.proof_worker.v2"
MAX_MESSAGE = 8 * 1024 * 1024
MAX_INPUT = 256 * 1024 * 1024


class WorkerFailure(RuntimeError):
    """A proof worker crashed or reported a failure; its traceback is retained."""


def _canonical(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _publish(path: Path, value) -> None:
    payload = _canonical(value)
    if len(payload) > MAX_MESSAGE:
        raise WorkerFailure("proof worker message exceeds explicit byte quota")
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("xb") as stream:
        stream.write(payload)
    pending.replace(path)


def _read(path: Path, limit: int) -> bytes:
    # Nonblocking open + regular-file check rejects FIFO/symlink result tricks.
    # A published name is atomic; an incomplete *.pending file is never read.
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise WorkerFailure("proof worker message is not a bounded regular file")
        pieces, size = [], 0
        while True:
            piece = os.read(fd, min(65536, limit + 1 - size))
            if not piece:
                break
            pieces.append(piece)
            size += len(piece)
            if size > limit:
                raise WorkerFailure("proof worker message exceeds explicit byte quota")
        return b"".join(pieces)
    finally:
        os.close(fd)


def _json(path: Path):
    try:
        return json.loads(_read(path, MAX_MESSAGE))
    except (OSError, ValueError) as exc:
        raise WorkerFailure(f"invalid proof worker message {path.name}: {exc}") from exc


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("native tensor proof worker exceeded its explicit deadline")
    return remaining


def _prepare(callback, args, logs: Path, deadline: float) -> tuple[Path, dict]:
    directory = logs / f"proof-worker-{uuid.uuid4().hex}"
    directory.mkdir(mode=0o700)
    # This pickle is produced and consumed only within our private request. Never
    # unpickle a worker result or data supplied by a native consumer.
    payload = pickle.dumps((callback, args), protocol=pickle.HIGHEST_PROTOCOL)
    if len(payload) > MAX_INPUT:
        raise WorkerFailure("proof worker input snapshot exceeds explicit byte quota")
    (directory / "input.pickle").write_bytes(payload)
    request = {
        "schema": SCHEMA,
        "run": directory.name,
        "input_sha256": hashlib.sha256(payload).hexdigest(),
        "deadline": deadline,
        "sys_path": [str(Path(p or os.getcwd()).absolute()) for p in sys.path],
    }
    _publish(directory / "request.json", request)
    return directory, request


def _identity(request: dict, pid: int) -> dict:
    return {
        "schema": SCHEMA,
        "run": request["run"],
        "request_sha256": hashlib.sha256(_canonical(request)).hexdigest(),
        "pid": pid,
        "group": pid,
    }


def _serve(directory: Path) -> None:
    request = _json(directory / "request.json")
    if set(request) != {"schema", "run", "input_sha256", "deadline", "sys_path"} or request["schema"] != SCHEMA:
        raise WorkerFailure("invalid proof worker request")
    identity = _identity(request, os.getpid())
    if os.getpgrp() != identity["group"] or os.getsid(0) != identity["pid"]:
        raise WorkerFailure("proof worker does not own a private session")
    _publish(directory / "ready.json", identity)
    while not (directory / "go.json").exists():
        time.sleep(min(0.01, _remaining(request["deadline"])))
    if _json(directory / "go.json") != identity:
        raise WorkerFailure("native proof worker did not receive ownership acknowledgment")
    _remaining(request["deadline"])
    payload = _read(directory / "input.pickle", MAX_INPUT)
    if hashlib.sha256(payload).hexdigest() != request["input_sha256"]:
        raise WorkerFailure("proof worker input snapshot changed")
    sys.path[:] = request["sys_path"]
    try:
        callback, args = pickle.loads(payload)
        value = callback(*args)
        _publish(directory / "result.json", {**identity, "kind": "result", "value": value})
    except BaseException as exc:
        detail = traceback.format_exc()
        print(detail, file=sys.stderr, flush=True)
        _publish(
            directory / "result.json",
            {**identity, "kind": "error", "type": type(exc).__name__, "reason": str(exc), "traceback": detail},
        )


def _receive(process, directory: Path, request: dict, deadline: float):
    identity = _identity(request, process.pid)
    acknowledged = False
    while True:
        remaining = _remaining(deadline)
        if not acknowledged and (directory / "ready.json").exists():
            if _json(directory / "ready.json") != identity:
                raise WorkerFailure("native proof worker did not establish its private process group")
            # Session ownership exists from start_owned, even if cancellation
            # occurs before ready. Callback execution still requires this ack.
            _publish(directory / "go.json", identity)
            acknowledged = True
        if (directory / "result.json").exists():
            message = _json(directory / "result.json")
            if not acknowledged or not isinstance(message, dict):
                raise WorkerFailure("native proof worker returned an invalid lifecycle message")
            common = {key: message.get(key) for key in identity}
            if common != identity:
                raise WorkerFailure("native proof worker result identity changed")
            if message.get("kind") == "error" and set(message) == set(identity) | {
                "kind",
                "type",
                "reason",
                "traceback",
            }:
                raise WorkerFailure(f"{message['type']}: {message['reason']}\n{message['traceback']}")
            if message.get("kind") != "result" or set(message) != set(identity) | {"kind", "value"}:
                raise WorkerFailure("native proof worker returned an invalid lifecycle message")
            try:
                process.wait(timeout=_remaining(deadline))
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError("native tensor proof worker did not exit after its result") from exc
            if process.returncode != 0:
                raise WorkerFailure(f"native proof worker failed after its result (exit {process.returncode})")
            _remaining(deadline)
            return message["value"]
        if process.poll() is not None:
            raise WorkerFailure(f"native proof worker exited without a result (exit {process.returncode})")
        time.sleep(min(remaining, 0.01))


def _command(directory: Path, request: dict) -> list[str]:
    # Restore parent import paths before importing the worker, rather than using
    # -m with an independently resolved PYTHONPATH. Bind the actual script bytes.
    path = Path(__file__).resolve()
    bootstrap = (
        "import hashlib,json,runpy,sys; "
        "path,owner,digest,paths=sys.argv[1:]; sys.path[:]=json.loads(paths); "
        "actual=hashlib.sha256(open(path,'rb').read()).hexdigest()\n"
        "if actual!=digest: raise RuntimeError('proof worker bootstrap source changed')\n"
        "sys.argv=[path,'--serve',owner]; runpy.run_path(path,run_name='__main__')"
    )
    return [
        sys.executable,
        "-c",
        bootstrap,
        str(path),
        str(directory),
        hashlib.sha256(path.read_bytes()).hexdigest(),
        json.dumps(request["sys_path"]),
    ]


def run(callback, *args, timeout: float, logs: Path):
    """Run a data-only proof callback in a fresh interpreter under one deadline.

    Cancellation adds at most the finite TERM/KILL cleanup graces to this budget.
    A child unreapable even after KILL is recorded and refused, never joined
    indefinitely. A successful return is JSON data, not executable pickle data.
    """
    if not math.isfinite(timeout) or timeout <= 0:
        raise TimeoutError("native tensor proof deadline expired before worker entry")
    deadline = time.monotonic() + timeout
    directory, request = _prepare(callback, args, Path(logs), deadline)
    _remaining(deadline)
    with (Path(logs) / "proof.stdout.log").open("xb") as stdout, (Path(logs) / "proof.stderr.log").open("xb") as stderr:
        process = start_owned(
            _command(directory, request),
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
        )
        try:
            return _receive(process, directory, request, deadline)
        finally:
            original = sys.exc_info()[1]
            try:
                cleanup = stop_owned(process)
                _publish(directory / "cleanup.json", cleanup)
                if not cleanup["direct_child_reaped"] or cleanup["errors"]:
                    raise WorkerFailure(f"proof worker cleanup remains pending: {cleanup}")
            except BaseException as exc:
                if original is None:
                    raise
                original.add_note(f"proof worker PID/group {process.pid} cleanup: {exc}")
            finally:
                close_owned_pipes(process)


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--serve":
        raise SystemExit("expected --serve private-request-directory")
    _serve(Path(sys.argv[2]))
