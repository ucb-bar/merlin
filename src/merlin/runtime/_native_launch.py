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
    return verify_process_adoption(pid, leader, image, work, temporary, ((elf, "/run/merlin/input.elf"),))


def verify_process_adoption(pid, leader, image, work, temporary, readonly_mounts):
    """Reopen each exact readonly input bind and the two private writable binds."""
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
    readonly = {str(target): source for source, target in readonly_mounts}
    if len(readonly) != len(readonly_mounts) or not _same_inode(root, image):
        raise ValueError("native process did not adopt its exact runtime and readonly input roster")
    for target, source in readonly.items():
        if not _same_inode(root / target.lstrip("/"), source):
            raise ValueError("native process did not adopt an exact readonly input")
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
        elif target in {"/", *readonly}:
            if "ro" not in options:
                raise ValueError("native runtime or input mount is writable")
        else:
            raise ValueError("native runtime exposed an unrelated mount")
        seen.add(target)
        inventory.append({"target": target, "options": options, "device": fields[2], "filesystem": backing[0]})
    if seen != {"/", *readonly, *expected}:
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


def _streams(process, *, deadline, max_bytes, writable_budget=None):
    data = {"stdout": bytearray(), "stderr": bytearray()}
    with selectors.DefaultSelector() as poll:
        poll.register(process.stdout, selectors.EVENT_READ, "stdout")
        poll.register(process.stderr, selectors.EVENT_READ, "stderr")
        while poll.get_map():
            if writable_budget is not None:
                _measure_writable(*writable_budget)
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


def _measure_writable(roots, max_files, max_bytes):
    """Bound the observed private products; this is not an aggregate disk quota."""
    from . import _native_files as F

    files, size = 0, 0
    for root in roots:
        for path in F.members(root, max_files=max_files):
            files += 1
            if files > max_files or path.is_symlink():
                raise ValueError("native private products exceed their closed file budget")
            if path.is_file():
                size += path.stat().st_size
                if size > max_bytes:
                    raise ValueError("native private products exceed their byte budget")


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
    return run_process(
        owner,
        selection,
        readonly_mounts=((elf, "/run/merlin/input.elf"),),
        inputs=(elf,),
        timeout=timeout,
        directory=directory,
        unlimited_stack=unlimited_stack,
    )


def run_process(
    owner, selection, *, readonly_mounts, inputs, timeout, directory, unlimited_stack=False, outputs=(), build=False
):
    """One shared kernel-verified namespace launch for native and build owners."""
    stage = "native_runtime_process"
    if build:
        # Reopen the live owner's complete current request before preserving
        # the original compiler step in the actual process record. Saved stage
        # claims cannot select this boundary independently of that request.
        selection = owner.reopen_selection(selection)
        stage = selection["identity"]["stage"]
        selection = {
            "guest_argv": selection["identity"]["guest_argv"],
            "cwd": selection["identity"]["guest_cwd"],
            "environment": selection["child_environment"],
            "storage_directories": selection["identity"]["storage_directories"],
            **selection,
        }
    with _retained_scratch(directory) as retained:
        scratch = retained.path
        work, temporary = scratch / "work", scratch / "tmp"
        work.mkdir()
        temporary.mkdir()
        if build:
            for selected_path in selection.get("storage_directories", ()):
                relative = Path(selected_path).relative_to("/tmp")
                if len(relative.parts) != 1 or relative.name in {".", ".."}:
                    raise ValueError("build storage selection escaped private temporary backing")
                (temporary / relative).mkdir()
        # Bash read consumes bytes; SEQPACKET would discard the remainder of
        # a GO packet after its first byte and hang before engine exec.
        control, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        control.setsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED, 1)
        # Source snapshot leases can retain thousands of descriptors. Only this
        # socket is inherited via Popen(pass_fds); bubblewrap forwards inherited
        # descriptors and has no --preserve-fds option. Never pass a whole range.
        descriptor = child.fileno()
        if not 3 <= descriptor <= 65535:
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
            "--ro-bind",
            str(owner.image),
            "/",
            *(token for source, target in readonly_mounts for token in ("--ro-bind", str(source), str(target))),
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
                stage=stage,
                argv=argv,
                cwd=directory,
                env={},
                inputs=inputs,
                outputs=outputs,
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
                adoption = (
                    verify_process_adoption(pid, process.pid, owner.image, work, temporary, readonly_mounts)
                    if build
                    else verify_adoption(pid, process.pid, owner.image, work, temporary, inputs[0])
                )
                owner.verify()
                if build:
                    resource.prlimit(pid, resource.RLIMIT_FSIZE, (owner.max_output_bytes, owner.max_output_bytes))
                control.send(b"GO\n")
                result = _streams(
                    process,
                    deadline=deadline,
                    max_bytes=owner.max_console_bytes,
                    **(
                        {"writable_budget": ((work, temporary), owner.max_output_files, owner.max_output_bytes)}
                        if build
                        else {}
                    ),
                )
                # The child is PID 1 of its private PID namespace. Its actual
                # exit closes that namespace's descendants; the wrapper is waited.
                with selectors.DefaultSelector() as exited:
                    exited.register(child_fd, selectors.EVENT_READ)
                    if not exited.select(0):
                        raise ValueError("native namespace leader has not actually exited")
                retained.closed = True
                if build and result.returncode == 0:
                    _measure_writable((work, temporary), owner.max_output_files, owner.max_output_bytes)
                    owner.collect_products(selection, work)
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
