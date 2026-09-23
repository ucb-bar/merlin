"""Explicitly provisioned node-local guardian parent. Never started on import.

Run in the foreground under the worker host's service manager. The service, not
the Chia worker or driver, constructs and reaps guardians. A restarted service
cannot attest to an earlier session. No daemonization or cluster mutation occurs.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import select
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

from ..frozen_python import active_source_identity, inherited_python_command
from . import _protocol as protocol


class Supervisor:
    def __init__(self, receipt_root: Path):
        self.identity = secrets.token_hex(16)
        self.source = active_source_identity()
        self.sessions = {}
        self.lock = threading.Lock()
        self.stopping = threading.Event()
        self.receipt_root = receipt_root / self.identity
        self.receipt_root.mkdir(mode=0o700)

    def _publish(self, invocation, result):
        result["source"] = self.source
        path = self.receipt_root / (result["invocation"] + ".json")
        descriptor = os.open(path.with_suffix(".pending"), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            json.dump(result, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(path.with_suffix(".pending"), path)
        invocation["receipt"] = result

    def _session(self, request):
        session = self.sessions.get(request.get("session"))
        if session is None:
            raise PermissionError("unknown supervisor session")
        return session

    def _request(self, connection):
        request, fds = protocol.receive(connection)
        try:
            with self.lock:
                operation = request.get("operation")
                if operation == "session":
                    if request.get("source") != self.source:
                        raise PermissionError("supervisor frozen source identity mismatch")
                    if request.get("host") != protocol.host_identity() or len(fds) != 1:
                        raise PermissionError("supervisor requires the same host and PID namespace")
                    protocol.verify_peer(connection, fds[0])
                    if len(self.sessions) >= 128:
                        raise RuntimeError("supervisor session capacity reached")
                    token = secrets.token_hex(32)
                    self.sessions[token] = {"driver": fds.pop(), "invocations": {}}
                    protocol.send(connection, {"session": token, "supervisor": self.identity, "source": self.source})
                    return
                session = self._session(request)
                if operation == "release":
                    if fds or any(item["receipt"] is None for item in session["invocations"].values()):
                        raise RuntimeError("cannot release an unresolved managed session")
                    os.close(session["driver"])
                    del self.sessions[request["session"]]
                    protocol.send(connection, {"released": True})
                    return
                if operation == "reserve":
                    if fds or protocol.exited(session["driver"]):
                        raise RuntimeError("driver session is no longer live")
                    if len(session["invocations"]) >= 1024:
                        raise RuntimeError("supervisor invocation capacity reached")
                    identifier = secrets.token_hex(16)
                    session["invocations"][identifier] = {
                        "cancel": threading.Event(),
                        "attached": False,
                        "receipt": None,
                    }
                    protocol.send(connection, {"invocation": identifier})
                    return
                invocation = session["invocations"].get(request.get("invocation"))
                if invocation is None:
                    raise PermissionError("unknown invocation")
                if operation == "receipt":
                    if fds:
                        raise ValueError("unexpected receipt descriptors")
                    protocol.send(connection, {"receipt": invocation["receipt"]})
                    return
                if operation == "cancel":
                    invocation["cancel"].set()
                    if not invocation["attached"] and invocation["receipt"] is None:
                        self._publish(
                            invocation,
                            {
                                "schema": "merlin.native-lifecycle.v1",
                                "supervisor": self.identity,
                                "invocation": request["invocation"],
                                "native_started": False,
                                "guardian_created": False,
                                "cleanup_complete": True,
                            },
                        )
                    protocol.send(connection, {"cancel_requested": True})
                    return
                if operation != "attach" or len(fds) != 1 or invocation["attached"] or invocation["cancel"].is_set():
                    raise ValueError("invalid or repeated native attachment")
                protocol.verify_peer(connection, fds[0])
                if request.get("host") != protocol.host_identity() or protocol.exited(session["driver"]):
                    raise PermissionError("native attachment owner is unavailable or foreign")
                invocation["attached"] = True
            self._guardian(connection, request["invocation"], invocation, fds[0], session["driver"])
        finally:
            protocol.close_fds(fds)

    def _guardian(self, worker, identifier, invocation, worker_fd, driver_fd):
        control, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        process = None
        guardian_fd = None
        receipt = None
        error = None
        ready = False
        started = False
        cancelled = False
        deadline = time.monotonic() + 12
        try:
            process = subprocess.Popen(
                inherited_python_command(
                    [
                        sys.executable,
                        "-m",
                        "merlin_experiments.execution.native_guardian",
                        str(child.fileno()),
                        str(worker_fd),
                        str(driver_fd),
                    ]
                ),
                pass_fds=(child.fileno(), worker_fd, driver_fd),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            guardian_fd = protocol.pidfd_open(process.pid)
            child.close()
            while True:
                if not cancelled and (
                    self.stopping.is_set()
                    or invocation["cancel"].is_set()
                    or protocol.exited(worker_fd)
                    or protocol.exited(driver_fd)
                ):
                    protocol.send(control, {"operation": "cancel"})
                    cancelled = True
                    deadline = time.monotonic() + 6
                if time.monotonic() >= deadline:
                    raise TimeoutError("guardian did not complete its admission or cleanup")
                readers = [control] if cancelled else [control, worker]
                available = select.select(readers, [], [], 0.02)[0]
                if control in available:
                    response, attachments = protocol.receive(control)
                    protocol.close_fds(attachments)
                    if response.get("state") == "ready" and not ready:
                        if response.get("source") != self.source:
                            raise PermissionError("guardian frozen source identity mismatch")
                        ready = True
                        try:
                            protocol.send(
                                worker, {"state": "ready", "supervisor": self.identity, "source": self.source}
                            )
                        except OSError:
                            invocation["cancel"].set()
                    elif response.get("state") == "finished":
                        receipt = response["receipt"]
                        break
                    else:
                        raise ValueError("invalid guardian transition")
                if worker in available:
                    try:
                        request, attachments = protocol.receive(worker)
                    except EOFError:
                        invocation["cancel"].set()
                        continue
                    try:
                        if request.get("operation") == "cancel":
                            invocation["cancel"].set()
                        elif ready and not started and request.get("operation") == "start" and len(attachments) == 3:
                            protocol.send(control, request, attachments)
                            started = True
                            deadline = float("inf")  # Native runtime belongs to the calling phase's budget.
                        else:
                            raise ValueError("invalid worker transition")
                    finally:
                        protocol.close_fds(attachments)
        except BaseException as exc:
            error = type(exc).__name__
        finally:
            child.close()
            control.close()  # EOF asks the guardian to clean its tree on protocol errors.
            reaped = False
            guardian_status = None
            if process is not None:
                try:
                    guardian_status = process.wait(timeout=6)
                    reaped = True
                except subprocess.TimeoutExpired:
                    # Never assert native cleanup after forced guardian termination.
                    if guardian_fd is not None:
                        protocol.send_signal(guardian_fd, signal.SIGKILL)
                    try:
                        guardian_status = process.wait(timeout=1)
                        reaped = True
                    except subprocess.TimeoutExpired:
                        pass
            if guardian_fd is not None:
                os.close(guardian_fd)
            result = {
                "schema": "merlin.native-lifecycle.v1",
                "supervisor": self.identity,
                "invocation": identifier,
                "guardian_created": process is not None,
                "guardian_reaped": reaped,
                "guardian_returncode": guardian_status,
                "guardian": receipt,
                "supervision_error": error,
                "cleanup_complete": bool(
                    error is None
                    and reaped
                    and guardian_status == 0
                    and receipt
                    and receipt.get("native_cleanup", {}).get("complete")
                ),
            }
            with self.lock:
                self._publish(invocation, result)
            try:
                protocol.send(worker, {"state": "finished", "receipt": result})
            except OSError:
                pass

    def handle(self, connection):
        with connection:
            connection.settimeout(10)
            try:
                self._request(connection)
            except BaseException as exc:
                try:
                    protocol.send(connection, {"error": type(exc).__name__})
                except OSError:
                    pass

    def expire_drivers(self):
        with self.lock:
            for token, session in list(self.sessions.items()):
                if not protocol.exited(session["driver"]):
                    continue
                for identifier, invocation in session["invocations"].items():
                    invocation["cancel"].set()
                    if not invocation["attached"] and invocation["receipt"] is None:
                        self._publish(
                            invocation,
                            {
                                "schema": "merlin.native-lifecycle.v1",
                                "supervisor": self.identity,
                                "invocation": identifier,
                                "native_started": False,
                                "guardian_created": False,
                                "cleanup_complete": True,
                            },
                        )
                if all(item["receipt"] is not None for item in session["invocations"].values()):
                    os.close(session["driver"])
                    del self.sessions[token]


def serve(endpoint: Path) -> int:
    endpoint = endpoint.absolute()
    parent = endpoint.parent.stat()
    if parent.st_uid != os.getuid() or parent.st_mode & 0o077 or endpoint.exists() or endpoint.is_symlink():
        raise PermissionError("supply a fresh socket in an owner-only managed directory")
    probe = protocol.self_pidfd()
    os.close(probe)
    receipts = endpoint.parent / "receipts"
    receipts.mkdir(mode=0o700, exist_ok=True)
    if receipts.is_symlink() or receipts.stat().st_uid != os.getuid() or receipts.stat().st_mode & 0o077:
        raise PermissionError("managed receipt directory must be owner-only")
    supervisor = Supervisor(receipts)
    threads = []
    signal.signal(signal.SIGTERM, lambda *_: supervisor.stopping.set())
    signal.signal(signal.SIGINT, lambda *_: supervisor.stopping.set())
    with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as listener:
        listener.bind(str(endpoint))
        endpoint.chmod(0o600)
        listener.listen(32)
        listener.settimeout(0.2)
        try:
            while not supervisor.stopping.is_set():
                supervisor.expire_drivers()
                try:
                    connection, _ = listener.accept()
                except TimeoutError:
                    continue
                thread = threading.Thread(target=supervisor.handle, args=(connection,), daemon=True)
                thread.start()
                threads = [old for old in threads if old.is_alive()]
                threads.append(thread)
        finally:
            supervisor.stopping.set()
            deadline = time.monotonic() + 8
            for thread in threads:
                thread.join(max(0, deadline - time.monotonic()))
            for session in supervisor.sessions.values():
                os.close(session["driver"])
            endpoint.unlink()
    return int(any(thread.is_alive() for thread in threads))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True, type=Path)
    return serve(parser.parse_args().endpoint)


if __name__ == "__main__":
    raise SystemExit(main())
