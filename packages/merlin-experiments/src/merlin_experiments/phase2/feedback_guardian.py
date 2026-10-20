"""Independent per-call callback child owner; no daemon or serialized authority.

Fork preserves the live issued evaluator context. Only this new guardian becomes
a subreaper. Its private socket and owner pidfd never enter the callback worker.
The guardian keeps a custodial lease descriptor until actual ECHILD cleanup, so
a lost coordinator cannot release the lease while descendant work is still live.
"""
from __future__ import annotations

import os
import signal
import time
from types import SimpleNamespace

from merlin_experiments.execution import _protocol as protocol
from merlin_experiments.execution.owned_children import become_child_subreaper, reap_owned_children

from .contracts import write_json

_WORKER_LEASE = None


def has_parent_feedback_lease(path):
    """Recognize only a live exact fork delegation in this callback process.

    This is a host scheduling observation, never a target measurement capability.
    A delegated child's own fork has a different PID and cannot inherit this role.
    """
    return _WORKER_LEASE == (os.getpid(), path)


def _callback_worker(callback, kwargs, destination, limits, control, owner_fd, lease_fd, lease_path):
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    control.close()
    os.close(owner_fd)
    if lease_fd is not None:
        os.close(lease_fd)
        global _WORKER_LEASE
        _WORKER_LEASE = (os.getpid(), lease_path)
    from .supervised_feedback import _worker

    _worker(callback, kwargs, destination, limits)


def run_feedback_guardian(callback, kwargs, destination, limits, deadline, control, other_control,
                          owner_fd, lease_fd, lease_path):
    """Own all direct/adopted descendants until reaped, even after owner loss."""
    other_control.close()
    process, child_fd, error, reason = None, None, None, "setup_failure"
    stop = False

    def stopping(*_args):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, stopping)
    try:
        os.setsid()
        become_child_subreaper()
        protocol.send(control, {"state": "ready", "guardian_pid": os.getpid()})
        while not protocol.readable([control, owner_fd], 0.02):
            if stop or time.monotonic() >= deadline:
                raise TimeoutError("callback admission expired")
        if protocol.exited(owner_fd):
            raise EOFError("callback owner exited before admission")
        command, fds = protocol.receive(control)
        protocol.close_fds(fds)
        if command != {"operation": "start"} or fds:
            raise ValueError("callback guardian requires exact private start admission")
        pid = os.fork()
        if pid == 0:
            try:
                _callback_worker(callback, kwargs, destination, limits, control, owner_fd, lease_fd, lease_path)
            finally:
                os._exit(1)
        process = SimpleNamespace(pid=pid, returncode=None)
        child_fd = protocol.pidfd_open(pid)
        protocol.send(control, {"state": "started", "worker_pid": pid})
        announced = False
        while True:
            if stop or protocol.exited(owner_fd):
                reason = "owner_lost" if protocol.exited(owner_fd) else "cancelled"
                break
            if time.monotonic() >= deadline:
                reason = "worker_deadline"
                break
            if protocol.exited(child_fd):
                reason = "worker_exit"
                break
            if destination.exists() and not announced:
                protocol.send(control, {"state": "result_ready"})
                announced = True
            if not protocol.readable([control], 0.02):
                continue
            command, fds = protocol.receive(control)
            protocol.close_fds(fds)
            if command != {"operation": "cancel"} or fds:
                raise ValueError("callback guardian requires exact private cancellation")
            reason = "cancelled"
            break
    except BaseException as exc:
        error = type(exc).__name__
    finally:
        if child_fd is not None:
            os.close(child_fd)
    attempt = 0
    while True:
        cleanup = reap_owned_children(process, 4)
        receipt = {
            "schema": "merlin.feedback_guardian_cleanup.v1", "guardian_pid": os.getpid(),
            "worker_pid": None if process is None else process.pid,
            "worker_exitcode": None if process is None else process.returncode,
            "reason": reason, "error": error, "cleanup": cleanup,
            "custodial_lease": lease_fd is not None,
            "scope": "invocation descendants only; delegated processes and OS hard deadline UNKNOWN",
        }
        path = destination.parent / ("guardian_cleanup_" + str(attempt) + ".json")
        write_json(path, receipt)
        path.chmod(0o400)
        try:
            protocol.send(control, {"state": "finished", "receipt": receipt})
            # Do not close a socket with an unread cancellation and discard its
            # final packet. The coordinator acknowledges bounded cleanup data.
            acknowledged_until = time.monotonic() + 1
            while time.monotonic() < acknowledged_until and not protocol.exited(owner_fd):
                if not protocol.readable([control], 0.02):
                    continue
                command, fds = protocol.receive(control)
                protocol.close_fds(fds)
                if command == {"operation": "cleanup_received"} and not fds:
                    break
                if command != {"operation": "cancel"} or fds:
                    break
        except (OSError, EOFError):
            pass
        if cleanup["complete"]:
            break
        # Retain the child owner and custodial lock when termination is unresolved.
        # A later actual ECHILD may close the tree; no snapshot grants release.
        attempt += 1
        time.sleep(0.05)
    if lease_fd is not None:
        os.close(lease_fd)
    os.close(owner_fd)
    control.close()
    os._exit(0 if error is None else 1)
