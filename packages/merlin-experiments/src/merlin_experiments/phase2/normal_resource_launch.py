"""Bound original normal Phase 1/2 namespace launches to a live resource owner.

This adapts already admitted argv; it supplies no tools, source or scientific
authority. The default path is exact. Unsupported hard-mode options refuse.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path

from merlin.targetgen.sandbox import preflight

from . import external_resources as resources
from .contracts import StageGateError


@contextmanager
def selected_boundary(boundary, *, writable_roots=()):
    """Explicitly bind the issued owner, including in broker worker threads."""
    with resources.executing_resource_boundary(boundary):
        if boundary is not None:
            boundary.verify(writable_roots=writable_roots)
        yield
        if boundary is not None:
            boundary.verify(writable_roots=writable_roots)


def _destination(value):
    path = Path(value)
    if not path.is_absolute() or str(path) != value or ".." in path.parts:
        raise StageGateError("normal resource launch has a noncanonical namespace destination")
    return path


def _scratch_plan(argv):
    """Parse only the original AW prefix; a payload or FD language is not a grant."""
    if not argv or any(not isinstance(value, str) or not value or "\0" in value for value in argv):
        raise StageGateError("normal resource launch has malformed original argv")
    argv, positions, writable = list(argv), [], []
    index, device, proc = 1, 0, 0
    switches = {"--die-with-parent", "--unshare-pid", "--clearenv", "--new-session"}
    while index < len(argv):
        flag = argv[index]
        if flag in switches:
            index += 1
        elif flag in {"--setenv", "--ro-bind", "--ro-bind-try", "--bind", "--bind-try"} and index + 2 < len(argv):
            if flag != "--setenv":
                _destination(argv[index + 2])
                if flag in {"--bind", "--bind-try"}:
                    writable.append(Path(argv[index + 1]))
            index += 3
        elif flag in {"--unsetenv", "--chdir", "--dir"} and index + 1 < len(argv):
            if flag != "--unsetenv":
                _destination(argv[index + 1])
            index += 2
        elif flag == "--tmpfs" and index + 1 < len(argv):
            _destination(argv[index + 1])
            positions.append(index)
            index += 2
        elif flag in {"--dev", "--proc"} and index + 1 < len(argv):
            if argv[index + 1] != {"--dev": "/dev", "--proc": "/proc"}[flag]:
                raise StageGateError("normal resource launch changed its original device/process namespace")
            device += flag == "--dev"
            proc += flag == "--proc"
            index += 2
        else:
            raise StageGateError("normal resource launch encountered an unsupported original namespace option")
    if device != 1 or proc != 1 or not positions or sum(argv[i + 1] == "/tmp" for i in positions) != 1:
        raise StageGateError("normal resource launch lost its original temporary/device/process namespace")
    return argv, tuple(positions), tuple(writable)


@contextmanager
def bounded_normal_scratch(argv, *, owner, boundary=None):
    """Keep scratch until the original real process returns or is killed and waited.

    Every original tmpfs mask remains empty but uses selected capped backing.
    Only namespace root/device metadata becomes readonly; original binds retain
    their permissions. Disable nested user namespaces so payloads cannot mount
    an unrelated tmpfs. Missing kernel support makes the actual launch refuse.
    """
    if boundary is None:
        yield argv
        return
    adjusted, positions, writable = _scratch_plan(argv)
    owner = resources.kernel.canonical(owner)
    with selected_boundary(boundary, writable_roots=(owner, *writable)):
        owner.mkdir(parents=True, mode=0o700, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="bounded-normal-", dir=owner) as temporary:
            root = Path(temporary)
            for number in range(len(positions)):
                (root / str(number)).mkdir(mode=0o700)
            (root / "shm").mkdir(mode=0o700)
            boundary.verify(writable_roots=(root,))
            for number, index in reversed(tuple(enumerate(positions))):
                adjusted[index : index + 2] = ["--bind", str(root / str(number)), adjusted[index + 1]]
            adjusted.extend(
                [
                    "--bind",
                    str(root / "shm"),
                    "/dev/shm",
                    "--remount-ro",
                    "/dev",
                    "--remount-ro",
                    "/",
                    "--unshare-user",
                    "--disable-userns",
                    "--assert-userns-disabled",
                ]
            )
            yield adjusted
            boundary.verify(writable_roots=(root,))


def require_bounded_sandbox(binary, *, owner, boundary):
    """Run the original fixed operability payload with the same bounded adapter."""
    original = preflight._probe_argv(str(binary))
    end = original.index("--")
    with bounded_normal_scratch(original[:end], owner=owner, boundary=boundary) as prefix:
        command = [*prefix, *original[end:]]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=20.0,
                check=False,
                env={**os.environ, "LC_ALL": "C"},
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise StageGateError("bounded original sandbox operability probe failed: " + type(exc).__name__) from exc
        probe = preflight.SandboxProbe(
            status=preflight.SANDBOX_OK if completed.returncode == 0 else preflight.SANDBOX_INOPERABLE,
            reason="probe_command_ran"
            if completed.returncode == 0
            else preflight._classify_stderr(completed.stderr or ""),
            binary=str(binary),
            returncode=completed.returncode,
            stderr=completed.stderr or "",
            argv=tuple(command),
        )
        if not probe.usable:
            raise preflight.SandboxUnavailable(probe, "bounded performance authoring sandbox operability preflight")
        return probe
