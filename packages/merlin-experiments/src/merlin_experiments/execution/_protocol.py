"""Private Linux-local packet transport and kernel-owned process identities."""

from __future__ import annotations

import array
import ctypes
import json
import os
import select
import signal
import socket
import struct
from pathlib import Path

MAX_PACKET = 256 * 1024


def host_identity() -> dict:
    return {
        "boot": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "pid_namespace": os.stat("/proc/self/ns/pid").st_ino,
    }


def _libc_function(name, arguments):
    try:
        function = getattr(ctypes.CDLL(None, use_errno=True), name)
    except AttributeError as exc:
        raise RuntimeError(f"managed native execution requires Linux libc {name}") from exc
    function.argtypes = arguments
    function.restype = ctypes.c_int
    return function


def pidfd_open(pid: int) -> int:
    if hasattr(os, "pidfd_open"):
        descriptor = os.pidfd_open(pid)
    else:
        descriptor = _libc_function("pidfd_open", [ctypes.c_int, ctypes.c_uint])(pid, 0)
        if descriptor < 0:
            raise OSError(ctypes.get_errno(), "managed native execution requires kernel pidfd_open")
    try:
        os.set_inheritable(descriptor, False)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def send_signal(descriptor: int, signum: int) -> None:
    if hasattr(signal, "pidfd_send_signal"):
        signal.pidfd_send_signal(descriptor, signum)
    elif (
        _libc_function("pidfd_send_signal", [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint])(
            descriptor, signum, None, 0
        )
        < 0
    ):
        raise OSError(ctypes.get_errno(), "managed native execution requires kernel pidfd_send_signal")


def self_pidfd() -> int:
    descriptor = pidfd_open(os.getpid())
    try:
        send_signal(descriptor, 0)  # Capability check without delivering a signal.
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def exited(fd: int) -> bool:
    return bool(select.select([fd], [], [], 0)[0])


def send(connection: socket.socket, message: dict, fds=()) -> None:
    payload = json.dumps(message, separators=(",", ":"), allow_nan=False).encode()
    if len(payload) > MAX_PACKET:
        raise ValueError("managed native control packet exceeds limit")
    ancillary = [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", fds))] if fds else []
    if connection.sendmsg([payload], ancillary) != len(payload):
        raise RuntimeError("incomplete managed native packet")


def receive(connection: socket.socket) -> tuple[dict, list[int]]:
    payload, ancillary, flags, _ = connection.recvmsg(MAX_PACKET, socket.CMSG_SPACE(8 * 4), socket.MSG_CMSG_CLOEXEC)
    fds = []
    try:
        for level, kind, data in ancillary:
            if (level, kind) != (socket.SOL_SOCKET, socket.SCM_RIGHTS):
                raise ValueError("unexpected control attachment")
            values = array.array("i")
            values.frombytes(data[: len(data) - len(data) % values.itemsize])
            fds.extend(values)
        if not payload:
            raise EOFError("managed native control connection closed")
        if flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC):
            raise ValueError("truncated managed native packet")
        message = json.loads(payload)
        if not isinstance(message, dict):
            raise ValueError("managed native packet must be an object")
        return message, fds
    except BaseException:
        close_fds(fds)
        raise


def close_fds(fds) -> None:
    for fd in fds:
        os.close(fd)


def verify_peer(connection: socket.socket, pidfd: int) -> None:
    pid, uid, _ = struct.unpack("3i", connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
    fields = dict(line.split(":", 1) for line in Path(f"/proc/self/fdinfo/{pidfd}").read_text().splitlines())
    if uid != os.getuid() or fields.get("Pid", "").strip() != str(pid) or exited(pidfd):
        raise PermissionError("pidfd does not identify the live local socket peer")


def connect(endpoint: str | Path, *, timeout: float = 10) -> socket.socket:
    path = Path(endpoint)
    parent = path.parent.stat()
    own = path.stat()
    if path.is_symlink() or parent.st_uid != os.getuid() or parent.st_mode & 0o077:
        raise PermissionError("supervisor endpoint requires an owner-only directory")
    if own.st_uid != os.getuid() or own.st_mode & 0o077:
        raise PermissionError("supervisor endpoint must be owner-only")
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    try:
        connection.settimeout(timeout)
        connection.connect(str(path))
        _, uid, _ = struct.unpack("3i", connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
        if uid != os.getuid():
            raise PermissionError("foreign supervisor peer")
        return connection
    except BaseException:
        connection.close()
        raise
