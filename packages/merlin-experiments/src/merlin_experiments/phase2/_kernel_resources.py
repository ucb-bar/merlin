"""Read Linux resource controls; never create, configure or migrate a resource owner."""

from __future__ import annotations

import os
import stat
from fractions import Fraction
from pathlib import Path

from .contracts import StageGateError

# Linux device ABI: https://docs.kernel.org/admin-guide/devices.html
_DEVICES = {
    "null": (1, 3),
    "zero": (1, 5),
    "full": (1, 7),
    "random": (1, 8),
    "urandom": (1, 9),
    "tty": (5, 0),
    "ptmx": (5, 2),
}


def cpu_set(value):
    cpus = set()
    for item in value.strip().split(","):
        lower, separator, upper = item.partition("-")
        if not lower.isdecimal() or separator and not upper.isdecimal():
            raise StageGateError("kernel CPU set is unsupported")
        first, last = int(lower), int(upper) if separator else int(lower)
        if not 0 <= first <= last <= 1_000_000:
            raise StageGateError("kernel CPU set exceeds its parser bound")
        cpus.update(range(first, last + 1))
    return tuple(sorted(cpus))


def _unescape(value):
    for escaped, plain in (("\\040", " "), ("\\011", "\t"), ("\\012", "\n"), ("\\134", "\\")):
        value = value.replace(escaped, plain)
    return value


def mounts(text):
    result = []
    for line in text.splitlines():
        before, separator, after = line.partition(" - ")
        left, right = before.split(), after.split()
        if not separator or len(left) < 6 or len(right) != 3:
            raise StageGateError("kernel mount inventory is unsupported")
        result.append((int(left[0]), _unescape(left[3]), _unescape(left[4]), right[0], left[5], right[2]))
    return tuple(result)


def canonical(path):
    path = Path(path)
    if not path.is_absolute() or path.resolve() != path:
        raise StageGateError("resource writable root is noncanonical or linked")
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise StageGateError("resource writable root has a linked ancestor")
    return path


def _membership(pid):
    value = (Path("/proc") / str(pid) / "cgroup").read_text().splitlines()
    if len(value) != 1 or not value[0].startswith("0::/"):
        raise StageGateError("resource enforcement requires an unambiguous unified cgroup v2 membership")
    path = Path(value[0][3:])
    if str(path) != value[0][3:] or ".." in path.parts:
        raise StageGateError("kernel cgroup membership is noncanonical")
    return path


def _process(pid):
    root = Path("/proc") / str(pid)
    stat = (root / "stat").read_text().rpartition(") ")[2].split()
    status = dict(line.partition(":")[::2] for line in (root / "status").read_text().splitlines() if ":" in line)
    if (
        stat[0] == "Z"
        or any(int(status[name].strip(), 16) for name in ("CapEff", "CapPrm", "CapAmb"))
        or status["NoNewPrivs"].strip() != "1"
        or set(map(int, status["Uid"].split())) != {os.getuid()}
    ):
        raise StageGateError("resource owner is dead, privileged or has foreign credentials")
    return stat[19], _membership(pid), (root / "ns" / "mnt").stat().st_ino


def _ceiling(value):
    return None if value == "max" else int(value)


def _cpu_ceiling(value):
    quota, period = value.split()
    if not period.isdecimal() or int(period) <= 0:
        raise StageGateError("kernel CPU bandwidth period is unsupported")
    return None if quota == "max" else Fraction(int(quota), int(period))


def _cgroup(inventory, cpu_ids, ram_bytes):
    selected = [row for row in inventory if row[3] == "cgroup2"]
    if len(selected) != 1:
        raise StageGateError("resource enforcement requires one visible cgroup v2 mount")
    mount = selected[0]
    membership = _membership(os.getpid())
    if not membership.is_relative_to(Path(mount[1])):
        raise StageGateError("cgroup namespace hides the selected resource controls")
    root = canonical(Path(mount[2]))
    current = canonical(root / membership.relative_to(mount[1]))
    rows, memories, swaps, cpus = [], [], [], []
    for directory in (current, *current.parents):
        if not directory.is_relative_to(root):
            break
        # Migration needs write access to the common ancestor. Refuse delegation
        # that permits the experiment itself to escape or change its ceiling.
        if os.access(directory, os.W_OK) or os.access(directory / "cgroup.procs", os.W_OK):
            raise StageGateError("resource controls permit caller migration or modification")
        values = []
        for name in ("memory.max", "memory.swap.max", "cpu.max", "cpuset.cpus"):
            control = directory / name
            if not control.exists() and directory == root:
                continue
            if control.is_symlink() or os.access(control, os.W_OK):
                raise StageGateError("resource ceiling is linked or writable by the caller")
            value = control.read_text().strip()
            values.append((name, value, control.stat().st_ino))
            if name == "cpuset.cpus":
                continue
            limit = _cpu_ceiling(value) if name == "cpu.max" else _ceiling(value)
            if limit is not None:
                (cpus if name == "cpu.max" else swaps if name == "memory.swap.max" else memories).append(limit)
        rows.append((str(directory), directory.stat().st_ino, tuple(values)))
    # Require a ceiling on the actual experiment group, not only a shared host ancestor.
    local = dict((name, value) for name, value, _inode in rows[0][2])
    if any(local.get(name, "max").split()[0] == "max" for name in ("memory.max", "memory.swap.max", "cpu.max")):
        raise StageGateError("current experiment cgroup lacks its own CPU/RAM/swap ceilings")
    effective = cpu_set((current / "cpuset.cpus.effective").read_text())
    if (
        effective != tuple(sorted(cpu_ids))
        or not memories
        or not 0 < min(memories) <= ram_bytes
        or not swaps
        or min(swaps) != 0
        or not cpus
        or not 0 < min(cpus) <= len(cpu_ids)
    ):
        raise StageGateError("kernel cgroup ceilings do not contain the selected CPU/RAM budget")
    return membership, (mount, tuple(rows), effective, min(memories), str(min(cpus)))


def _filesystem(inventory, writable_roots, disk_bytes):
    selected = None
    roots = tuple(canonical(path) for path in writable_roots)
    if not roots:
        raise StageGateError("resource boundary omitted its writable-root inventory")
    for path in roots:
        if any(Path(row[2]) != path and Path(row[2]).is_relative_to(path) for row in inventory):
            raise StageGateError("selected writable root contains another mounted filesystem")
        candidates = [row for row in inventory if path.is_relative_to(Path(row[2]))]
        mount = max(candidates, key=lambda row: len(Path(row[2]).parts))
        if selected is not None and mount != selected:
            raise StageGateError("selected writable roots do not share one capped backing filesystem")
        selected = mount
        if mount[1] != "/" or mount[3] not in {"ext4", "xfs", "tmpfs"} or "ro" in mount[4].split(","):
            raise StageGateError(
                "bounded storage needs a whole supported writable filesystem; project quotas unsupported"
            )
    root = canonical(selected[2])
    capacity = os.statvfs(root)
    total = capacity.f_blocks * capacity.f_frsize
    if not 0 < total <= disk_bytes:
        raise StageGateError(
            "backing filesystem capacity exceeds the selected disk budget; quota metadata is insufficient"
        )
    return selected, root.stat().st_dev, total, capacity.f_frsize


def _readonly_namespace(inventory, filesystem):
    """An enclosing stage namespace must contain every writable backing mount.

    Native same-Phase1 executors inherit this namespace. Metadata or an intended
    bind plan cannot substitute for the actual mounted readonly root and backing.
    Only procfs and the kernel's terminal interface have no filesystem data store.
    Device-directory metadata must be readonly; /dev/shm needs capped backing.
    """
    _selected, device, total, fragment = filesystem
    root_seen = False
    for row in inventory:
        path = canonical(row[2])
        readonly = "ro" in row[4].split(",") or "ro" in row[5].split(",")
        if path == Path("/"):
            root_seen = True
            if not readonly:
                raise StageGateError("hard resource launch requires an enclosing readonly filesystem root")
        if readonly or (row[3], str(path)) in {("proc", "/proc"), ("devpts", "/dev/pts")}:
            continue
        if path == Path("/dev") or row[3] not in {"ext4", "xfs", "tmpfs"}:
            raise StageGateError("hard resource namespace contains an unsupported writable mount")
        capacity = os.statvfs(path)
        if (
            path.stat().st_dev != device
            or capacity.f_blocks * capacity.f_frsize != total
            or capacity.f_frsize != fragment
        ):
            raise StageGateError("hard resource namespace contains writable backing outside its capped filesystem")
    if not root_seen:
        raise StageGateError("hard resource namespace has no visible readonly filesystem root")
    return inventory


def _device_namespace(directory=Path("/dev")):
    """Require the usual minimal device namespace, without raw/FUSE device access."""
    links = {
        "fd": "/proc/self/fd",
        "stdin": "/proc/self/fd/0",
        "stdout": "/proc/self/fd/1",
        "stderr": "/proc/self/fd/2",
        "core": "/proc/kcore",
        "ptmx": "pts/ptmx",
    }
    rows = []
    with os.scandir(directory) as entries:
        for entry in entries:
            if len(rows) >= 32:
                raise StageGateError("resource device namespace exceeds its minimal inventory")
            value = entry.stat(follow_symlinks=False)
            if entry.is_symlink():
                target = os.readlink(entry.path)
                if links.get(entry.name) != target:
                    raise StageGateError("resource device namespace contains an unsupported link")
            elif entry.name in {"pts", "shm"} and stat.S_ISDIR(value.st_mode):
                target = "directory"
            elif stat.S_ISCHR(value.st_mode) and _DEVICES.get(entry.name) == (
                os.major(value.st_rdev),
                os.minor(value.st_rdev),
            ):
                target = "kernel device"
            else:
                raise StageGateError("resource device namespace exposes a nonminimal device or file")
            rows.append((entry.name, target, value.st_ino, value.st_rdev))
    return tuple(sorted(rows))


def _writable_descriptors(pid, filesystem):
    """Readonly mounts cannot revoke inherited writable regular-file handles."""
    process = Path("/proc") / str(pid)
    for path in (process / "fd").iterdir():
        try:
            _descriptor(path, process / "fdinfo" / path.name, filesystem)
        except FileNotFoundError:
            continue  # A closed descriptor has no remaining write authority.


def _descriptor(path, information_path, filesystem):
    _selected, device, total, fragment = filesystem
    information = information_path.read_text().splitlines()
    flags = [line.partition(":")[2].strip() for line in information if line.startswith("flags:")]
    if len(flags) != 1:
        raise StageGateError("resource descriptor has no exact kernel access mode")
    before = path.stat()
    if stat.S_ISBLK(before.st_mode):
        raise StageGateError("resource process has a raw block-device descriptor")
    if stat.S_ISCHR(before.st_mode):
        numbers = os.major(before.st_rdev), os.minor(before.st_rdev)
        if numbers not in _DEVICES.values() and not 128 <= numbers[0] <= 143:
            raise StageGateError("resource process inherited a nonminimal character-device descriptor")
    if int(flags[0], 8) & os.O_ACCMODE != os.O_RDONLY:
        if stat.S_ISREG(before.st_mode):
            capacity = os.statvfs(path)
            if (
                before.st_dev != device
                or capacity.f_blocks * capacity.f_frsize != total
                or capacity.f_frsize != fragment
            ):
                raise StageGateError("inherited writable file descriptor escapes the capped backing filesystem")
    after = path.stat()
    if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
        raise StageGateError("resource descriptor changed identity during verification")


def observe(*, cpu_ids, ram_bytes, disk_bytes, writable_roots, owner_pids=()):
    """Reopen actual membership, immutable ceilings and whole-filesystem capacity.

    The filesystem limit bounds allocated backing storage, including unlinked
    open files. It does not promise a logical byte quota for sparse files.
    """
    try:
        inventory = mounts(Path("/proc/self/mountinfo").read_text())
        membership, cgroup = _cgroup(inventory, cpu_ids, ram_bytes)
        namespace_id = Path("/proc/self/ns/mnt").stat().st_ino
        filesystem = _filesystem(inventory, writable_roots, disk_bytes)
        namespace = _readonly_namespace(inventory, filesystem)
        devices = _device_namespace()
        for pid in (os.getpid(), *owner_pids):
            if type(pid) is not int or pid <= 0:
                raise StageGateError("resource process owner requires an actual positive PID")
            before = _process(pid)
            _writable_descriptors(pid, filesystem)
            if not before[1].is_relative_to(membership) or before[2] != namespace_id or _process(pid) != before:
                raise StageGateError("resource process is outside the experiment cgroup or changed identity")
        return (
            Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            cgroup,
            filesystem,
            (namespace_id, namespace, devices),
        )
    except (OSError, ValueError, IndexError, KeyError) as failure:
        raise StageGateError("live external resource boundary unavailable: " + type(failure).__name__) from failure
