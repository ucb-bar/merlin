"""Own a POSIX subprocess group without unbounded cleanup waits.

Descendants that deliberately create another session leave this ownership
boundary. A pending SIGKILL is reported honestly: the OS may prevent reaping a
child, and cleanup never waits indefinitely for it.
"""

from __future__ import annotations

import math
import os
import signal
import subprocess
from typing import Any


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
