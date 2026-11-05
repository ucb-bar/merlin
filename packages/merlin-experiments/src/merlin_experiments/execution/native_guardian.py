"""Single-invocation Linux subreaper. Launched only by the managed supervisor."""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time

from ..frozen_python import active_source_identity
from . import _protocol as protocol
from .owned_children import become_child_subreaper as _subreaper
from .owned_children import reap_owned_children as _cleanup


def supervise(control: socket.socket, worker_fd: int, driver_fd: int) -> dict:
    process = None
    started = False
    reason = "setup_failure"
    error = None
    stop = False

    def stopping(*_args):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, stopping)
    try:
        _subreaper()
        protocol.send(control, {"state": "ready", "source": active_source_identity()})
        deadline = time.monotonic() + 10
        while True:
            if stop or protocol.exited(worker_fd) or protocol.exited(driver_fd):
                reason = "owner_lost"
                break
            if not started and time.monotonic() >= deadline:
                reason = "admission_expired"
                break
            if process is not None and process.poll() is not None:
                reason = "leader_exit"
                break
            if not protocol.readable([control], 0.02):
                continue
            try:
                command, fds = protocol.receive(control)
            except EOFError:
                reason = "control_lost"
                break
            try:
                if command.get("operation") == "cancel":
                    reason = "cancelled"
                    break
                if started or command.get("operation") != "start" or len(fds) != 3:
                    raise ValueError("invalid native admission")
                argv, cwd, environment = command["argv"], command["cwd"], command["env"]
                if not isinstance(argv, list) or not argv or not all(isinstance(item, str) for item in argv):
                    raise ValueError("native argv must be a nonempty string list")
                if not isinstance(cwd, str) or not isinstance(environment, dict):
                    raise ValueError("native cwd and environment must be explicit")
                # All control and owner handles are close-on-exec, and only the
                # three explicit standard streams enter the native process.
                process = subprocess.Popen(
                    argv, cwd=cwd, env=environment, stdin=fds[0], stdout=fds[1], stderr=fds[2], close_fds=True
                )
                started = True
            finally:
                protocol.close_fds(fds)
    except BaseException as exc:
        error = type(exc).__name__
    cleanup = _cleanup(process, 4)
    return {
        "native_started": started,
        "returncode": process.returncode if process is not None else None,
        "reason": reason,
        "error": error,
        "native_cleanup": cleanup,
    }


def main() -> int:
    control_fd, worker_fd, driver_fd = map(int, sys.argv[1:])
    for fd in (control_fd, worker_fd, driver_fd):
        os.set_inheritable(fd, False)
    with socket.socket(fileno=control_fd) as control:
        receipt = supervise(control, worker_fd, driver_fd)
        try:
            protocol.send(control, {"state": "finished", "receipt": receipt})
        except OSError:
            pass
    protocol.close_fds((worker_fd, driver_fd))
    return 0 if receipt["native_cleanup"]["complete"] and receipt["error"] is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
