"""Linux direct/adopted child ownership for an invocation's sole subreaper.

Only a dedicated child becomes a subreaper. Actual waitpid/ECHILD closes its
descendant tree; a /proc roster is used only to open owned child pidfds.
"""

from __future__ import annotations

import ctypes
import os
import signal
import time
from pathlib import Path


def _libc(name, arguments):
    function = getattr(ctypes.CDLL(None, use_errno=True), name)
    function.argtypes, function.restype = arguments, ctypes.c_int
    return function


def pidfd_open(pid):
    descriptor = (
        os.pidfd_open(pid) if hasattr(os, "pidfd_open") else _libc("pidfd_open", [ctypes.c_int, ctypes.c_uint])(pid, 0)
    )
    if descriptor < 0:
        raise OSError(ctypes.get_errno(), "Linux kernel pidfd_open unavailable")
    try:
        os.set_inheritable(descriptor, False)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def send_signal(descriptor, signum):
    if hasattr(signal, "pidfd_send_signal"):
        signal.pidfd_send_signal(descriptor, signum)
    elif (
        _libc("pidfd_send_signal", [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint])(
            descriptor, signum, None, 0
        )
        < 0
    ):
        raise OSError(ctypes.get_errno(), "Linux kernel pidfd_send_signal unavailable")


def become_child_subreaper():
    libc = ctypes.CDLL(None, use_errno=True)
    flag = ctypes.c_int()
    if libc.prctl(36, 1, 0, 0, 0) != 0 or libc.prctl(37, ctypes.byref(flag), 0, 0, 0) != 0 or flag.value != 1:
        raise RuntimeError("Linux child subreaper unavailable")


def reap_owned_children(process, budget):
    deadline = time.monotonic() + budget
    escalation = time.monotonic() + min(1.0, budget / 2)
    owned, reaped, complete, error = {}, 0, False, None
    try:
        while time.monotonic() < deadline:
            children = Path(f"/proc/self/task/{os.getpid()}/children").read_text().split()
            for child in map(int, children):
                if child not in owned:
                    owned[child] = pidfd_open(child)
            for child, fd in list(owned.items()):
                try:
                    send_signal(fd, signal.SIGKILL if time.monotonic() >= escalation else signal.SIGTERM)
                except ProcessLookupError:
                    pass
                waited, status = os.waitpid(child, os.WNOHANG)
                if waited:
                    if process is not None and child == process.pid:
                        process.returncode = os.waitstatus_to_exitcode(status)
                    os.close(owned.pop(child))
                    reaped += 1
            if not owned:
                try:
                    os.waitid(os.P_ALL, 0, os.WEXITED | os.WNOHANG | os.WNOWAIT)
                except ChildProcessError:
                    complete = True
                    break
            time.sleep(0.02)
    except BaseException as exc:
        error = type(exc).__name__
    finally:
        for descriptor in owned.values():
            os.close(descriptor)
    return {"complete": complete, "reaped_children": reaped, "error": error}
