"""Kernel-issued resource verification and private scratch on capped backing storage.

No process launch, cgroup creation, quota configuration or scientific authority.
Launchers must inventory every original writable owner and verify any delegated
process before it starts work. Container services remain unsupported here.
"""

from __future__ import annotations

import tempfile
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path

from . import _kernel_resources as kernel
from .contracts import StageGateError, document_sha256

_ISSUED = {}
_ACTIVE = ContextVar("external_resource_boundary", default=None)


@dataclass(frozen=True)
class ExternalResourceBoundary:
    cpu_ids: tuple[int, ...]
    ram_bytes: int
    disk_bytes: int
    writable_roots: tuple[Path, ...]
    _state: tuple = field(repr=False)
    _issuer: object = field(repr=False, compare=False)

    def verify(self, *, cpu_ids=None, ram_bytes=None, disk_bytes=None, writable_roots=(), owner_pids=()):
        if _ISSUED.get(self._issuer) is not self:
            raise StageGateError("external resource boundary was not issued from live kernel controls")
        if any(
            selected is not None and selected != actual
            for selected, actual in (
                (cpu_ids, self.cpu_ids),
                (ram_bytes, self.ram_bytes),
                (disk_bytes, self.disk_bytes),
            )
        ):
            raise StageGateError("assigned resources differ from their selected external boundary")
        for path in writable_roots:
            path = kernel.canonical(path)
            if not any(path.is_relative_to(root) for root in self.writable_roots):
                raise StageGateError("external resource boundary omitted an original writable owner")
        state = kernel.observe(
            cpu_ids=self.cpu_ids,
            ram_bytes=self.ram_bytes,
            disk_bytes=self.disk_bytes,
            writable_roots=(*self.writable_roots, *writable_roots),
            owner_pids=owner_pids,
        )
        if state != self._state:
            raise StageGateError("external resource membership, ceilings or backing filesystem changed")
        return document_sha256(self._state)

    def receipt(self):
        return {
            "boundary_sha256": self.verify(),
            "cpu": "shared kernel cgroup v2 cpu.max and effective cpuset",
            "ram": "shared kernel cgroup v2 memory.max; swap disabled; normal kernel OOM semantics",
            "disk": "readonly enclosing namespace and capped writable backing; sparse logical byte quota UNKNOWN",
            "hard_aggregate_limits": "VERIFIED_KERNEL_BOUNDARY",
            "external_executors": "unbound native supervisors and container services unsupported",
        }


def prepare_external_resource_boundary(*, cpu_ids, ram_bytes, disk_bytes, writable_roots):
    """Bind actual externally provisioned cgroup/readonly namespace/capped backing.

    The whole stage and ordinary native executors must already run inside that
    namespace. Never accept a planned mount list or serialized observations.
    """
    if (
        type(cpu_ids) is not tuple
        or not cpu_ids
        or len(set(cpu_ids)) != len(cpu_ids)
        or any(type(cpu) is not int or cpu < 0 for cpu in cpu_ids)
        or any(type(value) is not int or value <= 0 for value in (ram_bytes, disk_bytes))
        or type(writable_roots) is not tuple
        or not writable_roots
    ):
        raise StageGateError("external resources require explicit CPU/RAM/disk and writable-root selections")
    roots = tuple(kernel.canonical(path) for path in writable_roots)
    state = kernel.observe(cpu_ids=cpu_ids, ram_bytes=ram_bytes, disk_bytes=disk_bytes, writable_roots=roots)
    issuer = object()
    boundary = ExternalResourceBoundary(cpu_ids, ram_bytes, disk_bytes, roots, state, issuer)
    _ISSUED[issuer] = boundary
    boundary.verify()
    return boundary


@contextmanager
def executing_resource_boundary(boundary):
    """Resource-only context, inherited by ordinary local children and fixed forks."""
    if boundary is not None:
        if type(boundary) is not ExternalResourceBoundary:
            raise StageGateError("execution requires the exact issued external resource owner")
        boundary.verify()
    token = _ACTIVE.set(boundary)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def active_resource_boundary():
    return _ACTIVE.get()


def refuse_unbound_native_executor():
    if active_resource_boundary() is not None:
        raise StageGateError("hard resource boundary cannot admit an unbound external native supervisor")


def _scratch_plan(argv):
    """Parse existing grants only; this pure plan supplies no resource authority."""
    argv = list(argv)
    end = argv.index("--") if "--" in argv else len(argv)
    positions, writable = [], []
    index, device = 1, False
    switches = {"--unshare-all", "--die-with-parent", "--new-session", "--clearenv"}
    while index < end:
        flag = argv[index]
        if flag in switches:
            index += 1
        elif flag in {"--ro-bind", "--bind", "--setenv"} and index + 2 < end:
            if flag == "--bind":
                writable.append(Path(argv[index + 1]))
            index += 3
        elif flag == "--chdir" and index + 1 < end:
            index += 2
        elif flag in {"--tmpfs", "--dev", "--proc"} and index + 1 < end:
            expected = {"--tmpfs": "/tmp", "--dev": "/dev", "--proc": "/proc"}[flag]
            if argv[index + 1] != expected:
                raise StageGateError("resource scratch encountered another unbounded writable namespace")
            if flag == "--tmpfs":
                positions.append(index)
            device |= flag == "--dev"
            index += 2
        else:
            raise StageGateError("resource scratch encountered an unsupported original namespace option")
    if len(positions) != 1 or not device or end == len(argv):
        raise StageGateError("resource scratch requires the original strict temporary/device namespace")
    return argv, positions[0], tuple(writable)


@contextmanager
def bounded_tool_scratch(argv, *, owner):
    """Keep original namespace/tool grants and bound every temporary writable mount.

    The caller already owns its evidence directory. Only fresh empty temporary
    directories are added there. Root/device directory metadata becomes readonly;
    original bind mounts retain their ordinary permissions. Legacy argv is exact.
    """
    boundary = active_resource_boundary()
    if boundary is None:
        yield argv
        return
    owner = kernel.canonical(owner)
    adjusted, index, writable = _scratch_plan(argv)
    boundary.verify(writable_roots=(owner, *writable))
    owner.mkdir(parents=True, mode=0o700, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="bounded-scratch-", dir=owner) as temporary:
        root = Path(temporary)
        for name in ("tmp", "shm"):
            (root / name).mkdir(mode=0o700)
        boundary.verify(writable_roots=(root,))
        adjusted[index : index + 2] = ["--bind", str(root / "tmp"), "/tmp"]
        end = adjusted.index("--")
        adjusted[end:end] = ["--bind", str(root / "shm"), "/dev/shm", "--remount-ro", "/dev", "--remount-ro", "/"]
        yield adjusted
        boundary.verify(writable_roots=(root,))
