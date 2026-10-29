"""Capture one external command only after its dedicated child owner reaps the tree.

The guardian alone becomes a Linux subreaper. Descendants that create their own
sessions are still its actual adopted children; a process-group snapshot is not
used as evidence of cleanup. This supplies no sandbox or execution qualification.
"""

from __future__ import annotations

import json
import math
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
from typing import Any

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




def start_owned(command, **kwargs: Any) -> subprocess.Popen:
    """Start one private session (PGID == PID), established before user code runs."""
    if os.name != "posix":
        raise NotImplementedError("owned subprocess groups require a POSIX host")
    if kwargs.pop("start_new_session", True) is not True or kwargs.get("process_group") not in (None, -1):
        raise ValueError("owned subprocess execution requires its own new session")
    if kwargs.pop("bufsize", 0) != 0:
        raise ValueError("owned subprocess pipes require unbuffered single-owner IO")
    return subprocess.Popen(command, start_new_session=True, bufsize=0, **kwargs)



def stop_owned(process: subprocess.Popen, *, terminate_grace: float = 0.2, kill_grace: float = 1.0) -> dict:
    """Request TERM then KILL of a start_owned group, using only bounded waits.

    Reaping the leader does not establish that all descendants exited. Always
    request KILL for its specifically owned group; never discover another group
    or signal the application's session. The returned record preserves PID and
    PGID even if the direct child remains unreaped after the kill grace period.
    """
    if not all(math.isfinite(value) and value >= 0 for value in (terminate_grace, kill_grace)):
        raise ValueError("owned subprocess cleanup grace periods must be nonnegative")
    errors = []
    kill_sent = False

    def request(sig, grace):
        nonlocal kill_sent
        try:
            os.killpg(process.pid, sig)
            if sig == signal.SIGKILL:
                kill_sent = True
        except ProcessLookupError:
            pass
        except OSError as exc:
            errors.append(f"{sig.name}: {exc}")
        try:
            process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            pass
        except OSError as exc:
            errors.append(f"bounded wait: {exc}")

    try:
        request(signal.SIGTERM, terminate_grace)
    finally:
        # A second cancellation during TERM grace must still request KILL.
        request(signal.SIGKILL, kill_grace)
    return {
        "pid": process.pid,
        "group": process.pid,
        "direct_child_reaped": process.returncode is not None,
        "group_kill_attempted": True,
        "group_kill_sent": kill_sent,
        "descendant_exit_confirmed": False,
        "errors": errors,
    }



def close_owned_pipes(process: subprocess.Popen) -> None:
    """Close each unbuffered start_owned pipe once, with no wait or input flush.

    The caller owns these streams on one thread. POSIX communicate uses selectors,
    not background readers; start_owned enforces raw pipes and write-through text
    input. Never raw-close an FD and then close its wrapper: another thread could
    reuse that FD between the two closes.
    """
    for stream in (process.stdin, process.stdout, process.stderr):
        if stream is not None and not stream.closed:
            try:
                stream.close()
            except OSError:
                pass



def run_owned(
    command,
    *,
    input=None,
    capture_output: bool = False,
    timeout: float | None = None,
    check: bool = False,
    terminate_grace: float = 0.2,
    kill_grace: float = 1.0,
    **kwargs: Any,
) -> subprocess.CompletedProcess:
    """subprocess.run capture semantics with bounded owned-group cancellation.

    Commands, output and TimeoutExpired/CalledProcessError contracts are kept.
    Cleanup grace periods are separate from the command's evaluation timeout;
    interrupted invocations retain the original exception and cleanup identity.
    """
    if not all(math.isfinite(value) and value >= 0 for value in (terminate_grace, kill_grace)):
        raise ValueError("owned subprocess cleanup grace periods must be nonnegative")
    if input is not None:
        if kwargs.get("stdin") is not None:
            raise ValueError("stdin and input arguments may not both be used.")
        kwargs["stdin"] = subprocess.PIPE
    if capture_output:
        if kwargs.get("stdout") is not None or kwargs.get("stderr") is not None:
            raise ValueError("stdout and stderr arguments may not be used with capture_output.")
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
    process = start_owned(command, **kwargs)
    try:
        try:
            stdout, stderr = process.communicate(input=input, timeout=timeout)
        except BaseException as exc:
            try:
                cleanup = stop_owned(process, terminate_grace=terminate_grace, kill_grace=kill_grace)
                exc.add_note(f"owned subprocess cleanup: {cleanup}")
            except BaseException as cleanup_error:
                exc.add_note(f"owned subprocess PID/group {process.pid} cleanup failed: {cleanup_error}")
            raise
        result = subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr)
        if check:
            result.check_returncode()
        return result
    finally:
        # Popen.__exit__ would wait indefinitely for an uninterruptible child.
        close_owned_pipes(process)



if __name__ == "__main__":
    _control = socket.socket(fileno=int(sys.argv[1]))
    _request = json.loads(_control.recv(1 << 20))
    _guardian(_request["argv"], _request["environment"], _request["cwd"], _control, int(sys.argv[2]), 1, 2)
