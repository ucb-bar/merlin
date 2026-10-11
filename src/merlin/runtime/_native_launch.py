"""Fixed closed-view native launch and kernel-derived adoption observation."""

from __future__ import annotations

import contextlib
import os
import resource
import selectors
import shutil
import signal
import socket
import struct
import subprocess
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

from merlin.common import invocation_record as I


def _stack():
    try:
        resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    except (OSError, ValueError):
        pass  # Preserve the original backend's stack policy.


def _mount_path(value):
    for encoded, decoded in (("\\040", " "), ("\\011", "\t"), ("\\012", "\n"), ("\\134", "\\")):
        value = value.replace(encoded, decoded)
    return value


def _same_inode(left, right):
    a, b = os.stat(left), os.stat(right)
    return (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino)


def verify_adoption(pid, leader, image, work, temporary, elf):
    """Inspect the blocked real child; a planned bind list cannot satisfy this."""
    child = Path("/proc") / str(pid)
    _closed_status((child / "status").read_text())
    ancestor = pid
    for _ in range(32):
        if ancestor == leader:
            break
        data = (Path("/proc") / str(ancestor) / "status").read_text()
        ancestor = next(int(row.partition(":")[2]) for row in data.splitlines() if row.startswith("PPid:"))
    else:
        raise ValueError("native READY sender is not the original launched child")
    if os.stat(child / "ns/mnt").st_ino == os.stat("/proc/self/ns/mnt").st_ino:
        raise ValueError("native process did not adopt a private runtime mount namespace")
    root = child / "root"
    expected = {"/run/work": work, "/tmp": temporary}
    if not _same_inode(root, image) or not _same_inode(root / "run/merlin/input.elf", elf):
        raise ValueError("native process did not adopt the exact runtime or requested ELF")
    seen, inventory = set(), []
    for row in (child / "mountinfo").read_text().splitlines():
        left, separator, right = row.partition(" - ")
        fields, backing = left.split(), right.split()
        if not separator or len(fields) < 6 or len(backing) < 3:
            raise ValueError("native runtime mount observation is unsupported")
        target, options = _mount_path(fields[4]), fields[5].split(",")
        if target in expected:
            if "rw" not in options or not _same_inode(root / target.lstrip("/"), expected[target]):
                raise ValueError("native writable backing differs from its fresh private scratch")
        elif target in {"/", "/run/merlin/input.elf"}:
            if "ro" not in options:
                raise ValueError("native runtime or input mount is writable")
        else:
            raise ValueError("native runtime exposed an unrelated mount")
        seen.add(target)
        inventory.append({"target": target, "options": options, "device": fields[2], "filesystem": backing[0]})
    if seen != {"/", "/run/merlin/input.elf", *expected}:
        raise ValueError("native process omitted an original closed-view mount")
    return {
        "pid": pid,
        "mount_namespace_inode": os.stat(child / "ns/mnt").st_ino,
        "mounts": sorted(inventory, key=lambda row: row["target"]),
    }


def _closed_status(text):
    observed = dict(row.split(":", 1) for row in text.splitlines() if ":" in row)
    if (
        int(observed["CapEff"].strip(), 16)
        or int(observed["CapPrm"].strip(), 16)
        or observed["NoNewPrivs"].strip() != "1"
        or tuple(map(int, observed["NSpid"].split()))[-1] != 1
    ):
        raise ValueError("native runtime lacks an unprivileged private namespace leader")


def _ready(control, process, *, timeout):
    control.settimeout(timeout)
    packet, ancillary, flags, _ = control.recvmsg(32, socket.CMSG_SPACE(struct.calcsize("3i")))
    credentials = [
        struct.unpack("3i", data)
        for level, kind, data in ancillary
        if level == socket.SOL_SOCKET and kind == socket.SCM_CREDENTIALS
    ]
    if packet != b"READY" or flags or len(credentials) != 1 or credentials[0][1] != os.getuid():
        raise ValueError("native runtime has no kernel-authenticated READY process")
    pid = credentials[0][0]
    if pid <= 0 or process.poll() is not None:
        raise ValueError("native runtime exited before adoption verification")
    return pid


def _streams(process, *, deadline, max_bytes):
    data = {"stdout": bytearray(), "stderr": bytearray()}
    with selectors.DefaultSelector() as poll:
        poll.register(process.stdout, selectors.EVENT_READ, "stdout")
        poll.register(process.stderr, selectors.EVENT_READ, "stderr")
        while poll.get_map():
            if time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(
                    process.args, 0, output=bytes(data["stdout"]), stderr=bytes(data["stderr"])
                )
            for key, _ in poll.select(min(0.1, max(0, deadline - time.monotonic()))):
                total = sum(map(len, data.values()))
                chunk = os.read(key.fileobj.fileno(), min(65536, max_bytes + 1 - total))
                if not chunk:
                    poll.unregister(key.fileobj)
                else:
                    data[key.data].extend(chunk)
                    if sum(map(len, data.values())) > max_bytes:
                        raise ValueError("native captured streams exceed their original byte budget")
    process.wait(timeout=max(0.001, deadline - time.monotonic()))
    return subprocess.CompletedProcess(process.args, process.returncode, bytes(data["stdout"]), bytes(data["stderr"]))


@contextlib.contextmanager
def _retained_scratch(directory):
    owner = SimpleNamespace(path=Path(tempfile.mkdtemp(prefix="native-call-", dir=directory)), closed=False)
    try:
        yield owner
    finally:
        if owner.closed:
            shutil.rmtree(owner.path)
        # Unknown process closure retains the original writable backing. The
        # enclosing original guardian still owns and reaps the invocation.


def run(owner, selection, elf, *, timeout, directory, unlimited_stack):
    """No caller callbacks or shell payloads: the bootstrap only gates and execs."""
    with _retained_scratch(directory) as retained:
        scratch = retained.path
        work, temporary = scratch / "work", scratch / "tmp"
        work.mkdir()
        temporary.mkdir()
        control, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        control.setsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED, 1)
        descriptor = child.fileno()
        if not 3 <= descriptor <= 512:
            control.close()
            child.close()
            raise ValueError("native private channel exceeds its fixed descriptor bound")
        script = (
            f"printf READY >&{descriptor}; read -r native_go <&{descriptor}; "
            f'[[ $native_go == GO ]] || exit 126; exec {descriptor}>&-; exec /usr/bin/env -- "$@"'
        )
        argv = [
            str(owner.bwrap),
            "--unshare-all",
            "--unshare-user",
            "--as-pid-1",
            "--cap-drop",
            "ALL",
            "--disable-userns",
            "--assert-userns-disabled",
            "--die-with-parent",
            "--new-session",
            "--clearenv",
            "--preserve-fds",
            str(descriptor - 2),
            "--ro-bind",
            str(owner.image),
            "/",
            "--ro-bind",
            str(elf),
            "/run/merlin/input.elf",
            "--bind",
            str(work),
            "/run/work",
            "--bind",
            str(temporary),
            "/tmp",
            "--chdir",
            selection["cwd"],
            "--",
            "/bin/bash",
            "--noprofile",
            "--norc",
            "-c",
            script,
            "native-runtime",
            *(f"{key}={value}" for key, value in sorted(selection["environment"].items())),
            *selection["guest_argv"],
        ]
        process, child_fd, adoption = None, None, None
        deadline = time.monotonic() + timeout
        try:
            with I.observe(
                directory,
                stage="native_runtime_process",
                argv=argv,
                cwd=directory,
                env={},
                inputs=(elf,),
                dependencies=owner.observation_sources(),
            ) as observed:
                process = subprocess.Popen(
                    argv,
                    cwd=directory,
                    env={},
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    pass_fds=(descriptor,),
                    start_new_session=True,
                    preexec_fn=_stack if unlimited_stack else None,
                )
                child.close()
                pid = _ready(control, process, timeout=min(10, max(0.001, deadline - time.monotonic())))
                child_fd = os.pidfd_open(pid)
                adoption = verify_adoption(pid, process.pid, owner.image, work, temporary, elf)
                owner.verify()
                control.send(b"GO\n")
                result = _streams(process, deadline=deadline, max_bytes=owner.max_console_bytes)
                # The child is PID 1 of its private PID namespace. Its actual
                # exit closes that namespace's descendants; the wrapper is waited.
                with selectors.DefaultSelector() as exited:
                    exited.register(child_fd, selectors.EVENT_READ)
                    if not exited.select(0):
                        raise ValueError("native namespace leader has not actually exited")
                retained.closed = True
                observed.complete(result)
                owner.verify()
                return result, observed.path, adoption
        finally:
            child.close()
            control.close()
            if child_fd is not None:
                try:
                    signal.pidfd_send_signal(child_fd, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if process is not None and process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
            if child_fd is not None:
                with selectors.DefaultSelector() as exited:
                    exited.register(child_fd, selectors.EVENT_READ)
                    retained.closed = bool(exited.select(5)) and process.poll() is not None
                os.close(child_fd)
