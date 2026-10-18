"""Bounded spawned proof work with an invocation-owned process group.

No live native IR crosses this boundary. Killing a failed worker also terminates
its checker descendants, while other compiler or verification jobs are untouched.
"""

from __future__ import annotations

import contextlib
import multiprocessing
import os
import signal
import time
import traceback
from pathlib import Path


class WorkerFailure(RuntimeError):
    """A proof worker crashed or reported a failure; its traceback is retained."""


def _serve(callback, args, sender, logs: Path) -> None:
    os.setsid()
    sender.send({"kind": "ready", "pid": os.getpid(), "group": os.getpgrp()})
    with (logs / "proof.stdout.log").open("w") as stdout, (logs / "proof.stderr.log").open("w") as stderr:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                value = callback(*args)
                sender.send({"kind": "result", "value": value})
            except BaseException as exc:
                detail = traceback.format_exc()
                stderr.write(detail)
                stderr.flush()
                sender.send({"kind": "error", "type": type(exc).__name__, "reason": str(exc), "traceback": detail})
    sender.close()


def _stop(worker, owned_group: bool) -> None:
    if owned_group:
        try:
            os.killpg(worker.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    elif worker.is_alive():
        worker.terminate()
    worker.join(0.2)
    # The worker may already have exited while a descendant ignores SIGTERM.
    # Always clean the specifically acknowledged group, never the parent's group.
    if owned_group:
        try:
            os.killpg(worker.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if worker.is_alive():
        worker.kill()
    worker.join()


def run(callback, *args, timeout: float, logs: Path):
    """Run a picklable proof callback in a fresh interpreter under one total deadline."""
    if timeout <= 0:
        raise TimeoutError("native tensor proof deadline expired before worker entry")
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    worker = context.Process(target=_serve, args=(callback, args, sender, logs))
    deadline = time.monotonic() + timeout
    owned_group = False
    try:
        worker.start()
        sender.close()
        result = None
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("native tensor proof worker exceeded its explicit deadline")
            if receiver.poll(min(remaining, 0.05)):
                try:
                    message = receiver.recv()
                except EOFError as exc:
                    raise WorkerFailure(
                        f"native proof worker exited without a result (exit {worker.exitcode})"
                    ) from exc
                if not owned_group:
                    if message != {"kind": "ready", "pid": worker.pid, "group": worker.pid}:
                        raise WorkerFailure("native proof worker did not establish its private process group")
                    owned_group = True
                elif message.get("kind") == "error":
                    raise WorkerFailure(f"{message['type']}: {message['reason']}\n{message['traceback']}")
                elif message.get("kind") == "result" and set(message) == {"kind", "value"}:
                    result = message["value"]
                    worker.join(max(0, deadline - time.monotonic()))
                    if worker.is_alive():
                        raise TimeoutError("native tensor proof worker did not exit after its result")
                    if worker.exitcode != 0:
                        raise WorkerFailure(f"native proof worker failed after its result (exit {worker.exitcode})")
                    return result
                else:
                    raise WorkerFailure("native proof worker returned an invalid lifecycle message")
            if not worker.is_alive() and not receiver.poll():
                raise WorkerFailure(f"native proof worker exited without a result (exit {worker.exitcode})")
    finally:
        if worker.pid is not None:
            _stop(worker, owned_group)
            worker.close()
        receiver.close()
        sender.close()
