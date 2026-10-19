"""Capture one external command only after its dedicated child owner reaps the tree.

The guardian alone becomes a Linux subreaper. Descendants that create their own
sessions are still its actual adopted children; a process-group snapshot is not
used as evidence of cleanup. This supplies no sandbox or execution qualification.
"""

from __future__ import annotations

import json
import locale
import os
import selectors
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

if __package__:
    from . import owned_children as children
else:
    # The isolated guardian imports only its exact sibling, never a checkout
    # discovered through PYTHONPATH, cwd or optional project installations.
    import importlib.util

    _spec = importlib.util.spec_from_file_location(
        "merlin_owned_children", Path(__file__).with_name("owned_children.py")
    )
    children = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(children)


class CleanupUnknown(RuntimeError):
    """The dedicated owner remains responsible for children whose cleanup is unknown."""


def _readable(readers, timeout):
    """Wait on actual reader identities without select's platform FD_SETSIZE cap."""
    with selectors.DefaultSelector() as selector:
        for reader in readers:
            selector.register(reader, selectors.EVENT_READ)
        return [key.fileobj for key, _events in selector.select(timeout)]


def _close_other_fds(kept):
    start = 3
    for descriptor in sorted(value for value in set(kept) if value >= 3):
        os.closerange(start, descriptor)
        start = descriptor + 1
    os.closerange(start, 1 << 20)


def _guardian(argv, environment, cwd, control, owner_fd, out_fd, err_fd):
    process, child_fd, error, stopped, exec_error_fd, exec_error_write = None, None, None, False, None, None

    def stopping(*_args):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stopping)
    try:
        # A concurrent invocation must not retain another job's pipe writers.
        # Descriptor enumeration is unnecessary: preserve only this owner's fds.
        _close_other_fds((control.fileno(), owner_fd, out_fd, err_fd))
        children.become_child_subreaper()
        exec_error_fd, exec_error_write = os.pipe2(os.O_CLOEXEC)
        pid = os.fork()
        if pid == 0:
            try:
                signal.signal(signal.SIGTERM, signal.SIG_DFL)
                signal.signal(signal.SIGINT, signal.SIG_DFL)
                os.dup2(out_fd, 1)
                os.dup2(err_fd, 2)
                _close_other_fds((exec_error_write,))
                if cwd is not None:
                    os.chdir(cwd)
                os.execvpe(argv[0], argv, environment)
            except OSError as caught:
                os.write(
                    exec_error_write,
                    json.dumps(
                        {"errno": caught.errno, "strerror": caught.strerror, "filename": caught.filename}
                    ).encode(),
                )
            finally:
                os._exit(126)
        os.close(exec_error_write)
        exec_error_write = None
        process = SimpleNamespace(pid=pid, returncode=None)
        os.close(out_fd)
        os.close(err_fd)
        child_fd = children.pidfd_open(pid)
        while not stopped:
            ready = _readable((control, owner_fd, child_fd), 0.02)
            if child_fd in ready or owner_fd in ready:
                break
            if control in ready:
                if control.recv(16) not in (b"cancel", b""):
                    error = "invalid cancellation"
                break
    except BaseException as caught:
        error = type(caught).__name__
    finally:
        if child_fd is not None:
            os.close(child_fd)
        if exec_error_write is not None:
            os.close(exec_error_write)
    # Never send completion while any direct/adopted child is unaccounted for.
    while True:
        cleanup = children.reap_owned_children(process, 4)
        if cleanup["complete"]:
            break
        time.sleep(0.02)
    exec_error = None
    if exec_error_fd is not None:
        raw = os.read(exec_error_fd, 4096)
        os.close(exec_error_fd)
        if raw:
            exec_error = json.loads(raw)
    packet = {
        "returncode": process.returncode if process else None,
        "cleanup": cleanup,
        "error": error,
        "exec_error": exec_error,
    }
    try:
        control.send(json.dumps(packet, sort_keys=True).encode())
        # Drain a racing cancellation before closing the seqpacket channel.
        until = time.monotonic() + 1
        while time.monotonic() < until and not _readable((owner_fd,), 0):
            if _readable((control,), 0.02):
                if control.recv(16) in (b"ack", b""):
                    break
    except OSError:
        pass
    control.close()
    os.close(owner_fd)
    os._exit(0 if error is None else 1)


def run(argv, *, cwd=None, env=None, timeout=None, capture_output=True, text=False, encoding=None, errors=None):
    """A captured command with descendant cleanup before return or TimeoutExpired.

    Only the explicit captured-output form is supported. Stdin retains the caller's
    inherited descriptor, as subprocess.run does without an input argument. Unknown
    cleanup raises CleanupUnknown and leaves its dedicated guardian responsible;
    callers must retain the surrounding job owner rather than claim completion.
    """
    if capture_output is not True:
        raise ValueError("owned command requires captured stdout and stderr")
    argv = [str(value) for value in argv]
    if not argv:
        raise ValueError("owned command requires argv")
    environment = dict(os.environ if env is None else env)
    request = json.dumps({"argv": argv, "environment": environment, "cwd": os.fspath(cwd) if cwd else None}).encode()
    owner_fd = children.pidfd_open(os.getpid())
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    out_read, out_write = os.pipe2(os.O_CLOEXEC)
    err_read, err_write = os.pipe2(os.O_CLOEXEC)
    guardian_process = None
    try:
        # Exec a fresh single-threaded guardian. Forking the compiler's Python
        # process and running Python there can inherit unrelated thread locks.
        guardian_process = subprocess.Popen(
            [sys.executable, "-I", "-B", str(Path(__file__).resolve()), str(child.fileno()), str(owner_fd)],
            stdout=out_write,
            stderr=err_write,
            pass_fds=(child.fileno(), owner_fd),
            start_new_session=True,
        )
        guardian = guardian_process.pid
        if parent.send(request) != len(request):
            raise RuntimeError("owned command request was not sent completely")
    except BaseException:
        parent.close()
        child.close()
        for descriptor in (owner_fd, out_read, out_write, err_read, err_write):
            os.close(descriptor)
        if guardian_process is not None:
            # No complete command request was admitted. The guardian cannot
            # have spawned the command, and EOF closes its initial receive.
            guardian_process.wait(timeout=2)
        raise
    child.close()
    for descriptor in (owner_fd, out_write, err_write):
        os.close(descriptor)
    chunks = {"stdout": [], "stderr": []}
    packet = failure = None
    cancelled = False
    deadline = None if timeout is None else time.monotonic() + timeout
    cleanup_deadline = None
    try:
        with selectors.DefaultSelector() as selector:
            for fd, name in ((out_read, "stdout"), (err_read, "stderr")):
                os.set_blocking(fd, False)
                selector.register(fd, selectors.EVENT_READ, name)
            selector.register(parent, selectors.EVENT_READ, "control")
            while selector.get_map():
                if not cancelled and deadline is not None and time.monotonic() >= deadline:
                    failure = subprocess.TimeoutExpired(argv, timeout)
                    parent.send(b"cancel")
                    cancelled, cleanup_deadline = True, time.monotonic() + 6
                if cancelled and time.monotonic() >= cleanup_deadline:
                    raise CleanupUnknown(f"owned command guardian {guardian} has not confirmed child cleanup")
                for key, _ in selector.select(0.02):
                    if key.data == "control":
                        raw = parent.recv(4096)
                        if raw:
                            packet = json.loads(raw)
                            parent.send(b"ack")
                        selector.unregister(parent)
                    else:
                        data = os.read(key.fd, 65536)
                        if data:
                            chunks[key.data].append(data)
                        else:
                            selector.unregister(key.fd)
        waited, status = os.waitpid(guardian, 0)
        guardian_process.returncode = os.waitstatus_to_exitcode(status)
        if (
            waited != guardian
            or os.waitstatus_to_exitcode(status) != 0
            or not packet
            or packet.get("error") is not None
            or not packet.get("cleanup", {}).get("complete")
        ):
            raise CleanupUnknown(f"owned command guardian {guardian} did not complete cleanly")
        stdout, stderr = (b"".join(chunks[name]) for name in ("stdout", "stderr"))
        cleanup = {**packet["cleanup"], "guardian_reaped": True}
        if failure is not None:
            failure.output, failure.stderr = stdout, stderr
            failure.owned_process_cleanup = cleanup
            raise failure
        if packet.get("exec_error"):
            error = packet["exec_error"]
            raise OSError(error["errno"], error["strerror"], error["filename"])
        if text or encoding is not None or errors is not None:
            selected_encoding, selected_errors = encoding or locale.getencoding(), errors or "strict"
            stdout, stderr = (
                data.decode(selected_encoding, selected_errors).replace("\r\n", "\n").replace("\r", "\n")
                for data in (stdout, stderr)
            )
        result = subprocess.CompletedProcess(argv, packet["returncode"], stdout, stderr)
        result.owned_process_cleanup = cleanup
        return result
    except BaseException as caught:
        if not cancelled and packet is None:
            try:
                parent.send(b"cancel")
            except OSError:
                pass
            # An interrupted caller still waits for actual child ownership to
            # close. The dedicated guardian also watches the caller's pidfd.
            until = time.monotonic() + 6
            while time.monotonic() < until:
                if not _readable((parent,), 0.02):
                    continue
                raw = parent.recv(4096)
                if raw:
                    packet = json.loads(raw)
                    parent.send(b"ack")
                break
            if packet and packet.get("cleanup", {}).get("complete"):
                _, status = os.waitpid(guardian, 0)
                guardian_process.returncode = os.waitstatus_to_exitcode(status)
            else:
                raise CleanupUnknown(f"owned command guardian {guardian} has not confirmed child cleanup") from caught
        raise
    finally:
        parent.close()
        os.close(out_read)
        os.close(err_read)


if __name__ == "__main__":
    _control = socket.socket(fileno=int(sys.argv[1]))
    _request = json.loads(_control.recv(1 << 20))
    _guardian(_request["argv"], _request["environment"], _request["cwd"], _control, int(sys.argv[2]), 1, 2)
