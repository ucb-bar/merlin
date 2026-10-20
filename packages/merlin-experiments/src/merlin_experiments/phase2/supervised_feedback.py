"""Closed bounded callback transport with an independent invocation child owner.

Only the guardian owns/reaps callback descendants. Parent reception rejects late
results; local OS calls and decoding are not interruptible hard OS deadlines.
Leases stay held on unresolved cleanup. Delegated non-descendant work is UNKNOWN.
"""
from __future__ import annotations

import os
import signal
import socket
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

from merlin.benchharness import hash_tree
from merlin_experiments.execution import _protocol as protocol
from merlin_experiments.execution import owned_children

from .contracts import StageGateError, mapping_file, sha256_file, write_json
from .feedback_guardian import run_feedback_guardian
from .feedback_protocol import FeedbackValueLimits, check_deadline, encode_feedback_value, receive_feedback_value
from .portfolio_launch import acquire_host_resource_lease

# Keep parent handles when no actual cleanup/reaping proof closes their lifetime.
# These are not a release capability or a background service.
_UNRESOLVED = []


def _worker(callback, kwargs, destination, limits):
    os.setsid()
    try:
        data = encode_feedback_value({"result": callback(**kwargs)}, limits=limits)
    except BaseException as exc:
        data = encode_feedback_value({"failure": [type(exc).__name__, str(exc)[:512]]},
                                     limits=FeedbackValueLimits())
    temporary = destination.with_suffix(".partial")
    with temporary.open("xb") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(data)
    temporary.replace(destination)
    # The guardian owns and reaps this process, including adopted double forks.
    while True:
        signal.pause()


def _packet(control, guardian_fd, deadline):
    def receive():
        try:
            value, fds = protocol.receive(control)
        except (OSError, EOFError) as error:
            raise StageGateError("component callback guardian control lost; cleanup UNKNOWN") from error
        protocol.close_fds(fds)
        if fds:
            raise StageGateError("component callback control has foreign descriptors")
        return value

    while True:
        check_deadline(deadline)
        ready = protocol.readable([control, guardian_fd], min(0.02, deadline - time.monotonic()))
        if control in ready:
            return receive()
        if guardian_fd in ready:
            # Exit readiness can appear after select sampled the socket. Drain a
            # final atomic packet before treating the owner as evidence-less.
            if protocol.readable([control], 0):
                return receive()
            raise StageGateError("component callback guardian exited without cleanup evidence")


def _close_guardian(control, guardian_pid, guardian_fd, receipt=None):
    """Request cancellation; require actual guardian receipt and parent reaping."""
    if receipt is None and not protocol.exited(guardian_fd):
        protocol.send_signal(guardian_fd, signal.SIGCONT)
        try:
            protocol.send(control, {"operation": "cancel"})
        except OSError:
            pass
    deadline = time.monotonic() + 5
    try:
        while receipt is None:
            value = _packet(control, guardian_fd, deadline)
            if value.get("state") == "finished":
                receipt = value.get("receipt")
        protocol.send(control, {"operation": "cleanup_received"})
    except (StageGateError, OSError, EOFError):
        pass
    while time.monotonic() < deadline:
        # The PID cannot recycle before this sole parent reaps its child. A
        # multiprocessing pipe sentinel can be inherited by escaped children;
        # actual pidfd exit and waitpid, rather than pipe EOF, close this owner.
        if protocol.exited(guardian_fd):
            waited, status = os.waitpid(guardian_pid, os.WNOHANG)
            if waited:
                return receipt, True, os.waitstatus_to_exitcode(status)
        time.sleep(0.01)
    return receipt, False, None


def _complete_cleanup(receipt, process_pid, root):
    if not isinstance(receipt, dict) or receipt.get("guardian_pid") != process_pid:
        return False
    retained = root / "guardian_cleanup_0.json"
    try:
        if not retained.is_file() or retained.is_symlink() or mapping_file(retained) != receipt:
            return False
    except (OSError, ValueError, StageGateError):
        return False
    return receipt.get("cleanup", {}).get("complete") is True and receipt["cleanup"].get("error") is None


def bounded_feedback(callback, *, timeout_s, output, kwargs, limits=FeedbackValueLimits(), lease_path=None):
    """Refuse late data; release a parent lease only after actual descendant closure."""
    if not isinstance(timeout_s, (int, float)) or isinstance(timeout_s, bool) or not 0 < timeout_s <= 2400:
        raise StageGateError("component feedback needs a bounded positive wall budget")
    started, primary, result, received_bytes = time.monotonic(), None, None, None
    deadline = started + timeout_s
    if type(limits) is not FeedbackValueLimits:
        raise StageGateError("component feedback requires typed result limits")
    limits.verify()
    output = Path(output)
    if not output.is_absolute() or output.resolve() != output or output.is_symlink():
        raise StageGateError("component feedback requires canonical private output")
    output.mkdir(parents=True, mode=0o700, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="component_worker_", dir=output))
    destination = root / "result.json"
    lease = None
    if lease_path is not None:
        lease_path = Path(lease_path)
        if not lease_path.is_absolute() or lease_path.resolve() != lease_path or lease_path.is_symlink():
            raise StageGateError("component feedback requires an exact parent-owned lease path")
        lease = acquire_host_resource_lease(root, lease_path=lease_path)
        if lease is None:
            raise StageGateError("component feedback parent resource lease is busy")
    control = child_control = owner_fd = None
    try:
        # Probe the actual kernel before any guardian or callback starts.
        owner_fd = protocol.self_pidfd()
        control, child_control = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    except BaseException:
        if lease is not None:
            lease.close()
        if control is not None:
            control.close()
        if child_control is not None:
            child_control.close()
        if owner_fd is not None:
            os.close(owner_fd)
        raise
    guardian_fd, cleanup, reaped, exitcode, worker_pid, guardian_pid = None, None, False, None, None, None
    try:
        guardian_pid = os.fork()
        if guardian_pid == 0:
            try:
                run_feedback_guardian(callback, kwargs, destination, limits, deadline, child_control, control,
                                      owner_fd, None if lease is None else lease.fileno(), lease_path)
            finally:
                os._exit(1)
        guardian_fd = protocol.pidfd_open(guardian_pid)
        child_control.close()
        os.close(owner_fd)
        owner_fd = None
        ready = _packet(control, guardian_fd, deadline)
        if ready != {"state": "ready", "guardian_pid": guardian_pid}:
            raise StageGateError("component callback guardian failed independent child-owner setup")
        protocol.send(control, {"operation": "start"})
        launched = _packet(control, guardian_fd, deadline)
        if set(launched) != {"state", "worker_pid"} or launched["state"] != "started":
            raise StageGateError("component callback guardian failed worker admission")
        worker_pid = launched["worker_pid"]
        while not destination.exists():
            if destination.is_symlink():
                raise StageGateError("component feedback worker result is not a plain file")
            if time.monotonic() >= deadline:
                raise StageGateError("component feedback exceeded worker deadline")
            if protocol.readable([control, guardian_fd], 0.02):
                packet = _packet(control, guardian_fd, deadline)
                if packet.get("state") == "finished":
                    cleanup = packet.get("receipt")
                    detail = ("exceeded worker deadline" if cleanup.get("reason") == "worker_deadline"
                              else "ended before a complete result")
                    raise StageGateError("component feedback worker " + detail)
        check_deadline(deadline)
        result, received_bytes = receive_feedback_value(destination, limits=limits, deadline=deadline)
        if type(result) is not dict or set(result) not in ({"result"}, {"failure"}):
            raise StageGateError("component feedback worker omitted its closed outcome envelope")
        if "failure" in result:
            failure = result["failure"]
            if type(failure) is not list or len(failure) != 2 or any(type(s) is not str for s in failure):
                raise StageGateError("component feedback worker failure frame is malformed")
            raise StageGateError("component feedback worker refused: " + ": ".join(failure))
    except BaseException as error:
        primary = error
    finally:
        if guardian_fd is not None:
            observed, reaped, exitcode = _close_guardian(control, guardian_pid, guardian_fd, cleanup)
            cleanup = observed if observed is not None else cleanup
        closed = reaped and _complete_cleanup(cleanup, guardian_pid, root)
        if closed and worker_pid is None:
            # A callback can pause its coordinator before STARTED is received.
            # The actual closed guardian product still names the reaped worker.
            worker_pid = cleanup.get("worker_pid")
        if lease is not None and (closed or guardian_pid is None):
            lease.close()
        if guardian_pid is not None and not closed:
            _UNRESOLVED.append((guardian_pid, guardian_fd, lease, root))
        elif guardian_fd is not None:
            os.close(guardian_fd)
        if owner_fd is not None:
            os.close(owner_fd)
        control.close()
        child_control.close()
        if primary is None and not closed:
            primary = StageGateError("component feedback descendant cleanup is unresolved; resource lease retained")
        if primary is None and (exitcode != 0 or cleanup.get("error") is not None):
            primary = StageGateError("component feedback guardian failed its private lifecycle protocol")
        if primary is None:
            try:
                check_deadline(deadline)
            except StageGateError as error:
                primary = error
        sources = (Path(__file__).resolve(), Path(run_feedback_guardian.__code__.co_filename).resolve(),
                   Path(owned_children.__file__).resolve(),
                   Path(owned_children.reap_owned_children.__code__.co_filename).resolve(),
                   Path(protocol.__file__).resolve(),
                   Path(protocol.readable.__code__.co_filename).resolve(),
                   Path(receive_feedback_value.__code__.co_filename).resolve(),
                   Path(acquire_host_resource_lease.__code__.co_filename).resolve())
        write_json(root / "lifecycle.json", {
            "schema": "merlin.feedback_worker_lifecycle.v2", "status": "refused" if primary else "returned",
            "limits": asdict(limits), "elapsed_seconds": time.monotonic() - started,
            "deadline_exceeded": time.monotonic() >= deadline, "received_bytes": received_bytes,
            "worker": {"pid": worker_pid, "reaped": closed and worker_pid is not None,
                       "exitcode": None if cleanup is None else cleanup.get("worker_exitcode")},
            "guardian": {"pid": guardian_pid,
                         "reaped": reaped, "exitcode": exitcode},
            "failure": None if primary is None else {"type": type(primary).__name__, "detail": str(primary)[:512]},
            "cleanup": {"status": "COMPLETE" if closed else "UNKNOWN", "method": "subreaper and waitpid ECHILD",
                        "receipt_sha256": sha256_file(root / "guardian_cleanup_0.json") if closed else None},
            "lease_release": {"status": "NOT_REQUESTED" if lease is None else "RELEASED" if closed else "UNKNOWN",
                              "owner": "parent with guardian custodial descriptor",
                              "path": None if lease_path is None else str(lease_path)},
            "sources": [(str(path), sha256_file(path)) for path in sources],
            "scope": ("owned invocation descendants only; "
                      "external delegation/namespace isolation/OS hard deadline UNKNOWN"),
        })
        (root / "lifecycle.json").chmod(0o400)
    if primary is not None:
        raise primary
    return result["result"]


def record_feedback_refusal(*, output, action, call_index, candidate, expected_candidate_sha256, error):
    """Retain actual mutation/refusal evidence privately; no restoration authority."""
    output = Path(output)
    if not output.is_absolute() or output.resolve() != output or output.is_symlink():
        raise StageGateError("component feedback refusal requires canonical private output")
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    root = Path(tempfile.mkdtemp(prefix="component_refusal_", dir=output))
    try:
        observed = str(hash_tree(candidate)["sha256"])
    except Exception:  # noqa: BLE001 - an unreadable mutated candidate remains unknown
        observed = None
    write_json(root / "refusal.json", {
        "schema": "merlin.component_feedback_refusal.v1", "action": action, "call_index": call_index,
        "failure": {"type": type(error).__name__, "detail": str(error)[:512]},
        "candidate_before_sha256": expected_candidate_sha256, "candidate_after_sha256": observed,
        "candidate_changed": (observed != expected_candidate_sha256
                              if observed is not None and expected_candidate_sha256 is not None else "UNKNOWN"),
        "descendant_cleanup": "UNKNOWN", "lease_release": "UNKNOWN",
        "scope": "private mutation/refusal observation; no rollback, cleanup or target authority",
    })
    (root / "refusal.json").chmod(0o400)
