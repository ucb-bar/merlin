"""Make sealed evidence read-only, and immutable where the host permits, and say which it got.

A seal is a promise that bytes will still be there and still be the same when something re-reads
them. Clearing write bits keeps a careless writer out but not a cleanup: deleting a file only needs
write access to its DIRECTORY, and a cleanup that meets a read-only tree restores the bits and
deletes anyway (that is the documented procedure for build trees, and it was applied to sealed
evidence on 2026-10-24). The Linux immutable attribute is the only flag an owner cannot clear
without ``CAP_LINUX_IMMUTABLE``, so a seal asks for it too -- and records whether the host granted
it, because "we tried" and "it is immutable" are different claims a verifier must not conflate.

Shared content-store files are left alone: their inode is every other snapshot's file too, so
locking one name would lock (and later refuse to link) all of them.

``MERLIN_SEAL_IMMUTABLE`` selects the attempt: ``attempt`` (default, plain ``chattr``), ``sudo``
(``sudo -n chattr``, never prompting) or ``off``. A caller may pass ``mode=`` explicitly instead.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from collections.abc import Iterable
from pathlib import Path

ENV = "MERLIN_SEAL_IMMUTABLE"
MODES = ("attempt", "sudo", "off")
_BATCH = 128


def selected_mode(mode: str | None = None) -> str:
    chosen = (mode or os.environ.get(ENV) or "attempt").strip()
    if chosen not in MODES:
        raise ValueError(f"{ENV} must be one of {MODES}, not {chosen!r}")
    return chosen


def _members(roots: Iterable[Path]) -> tuple[list[Path], list[Path]]:
    """Regular files and directories under ``roots``, deepest first; symlinks are never followed."""
    files: list[Path] = []
    dirs: list[Path] = []
    for root in roots:
        root = Path(root)
        if root.is_symlink():
            continue
        if root.is_file():
            files.append(root)
            continue
        if not root.is_dir():
            continue
        dirs.append(root)
        for current, subdirs, names in os.walk(root, followlinks=False):
            base = Path(current)
            for name in names:
                path = base / name
                if not path.is_symlink() and path.is_file():
                    files.append(path)
            for name in subdirs:
                path = base / name
                if not path.is_symlink():
                    dirs.append(path)
    dirs.sort(key=lambda p: len(p.parts), reverse=True)
    return files, dirs


def _shared(path: Path) -> bool:
    try:
        return path.stat(follow_symlinks=False).st_nlink > 1
    except OSError:
        return False


def read_only(roots: Iterable[Path]) -> dict:
    """Clear every write bit under ``roots`` (files first, then directories, deepest first)."""
    files, dirs = _members(roots)
    changed = shared = 0
    for path in files:
        if _shared(path):
            shared += 1
            continue
        mode = stat.S_IMODE(path.stat(follow_symlinks=False).st_mode)
        if mode & 0o222:
            path.chmod(mode & ~0o222)
            changed += 1
    for path in dirs:
        mode = stat.S_IMODE(path.stat(follow_symlinks=False).st_mode)
        if mode & 0o222:
            path.chmod(mode & ~0o222)
            changed += 1
    return {"files": len(files) - shared, "directories": len(dirs), "changed": changed, "shared_skipped": shared}


def _chattr_command(mode: str) -> list[str] | None:
    chattr = shutil.which("chattr")
    if chattr is None:
        return None
    if mode == "sudo":
        sudo = shutil.which("sudo")
        return None if sudo is None else [sudo, "-n", chattr]
    return [chattr]


def immutable_flags(paths: Iterable[Path]) -> dict[Path, bool | None]:
    """Whether each path carries the immutable attribute; ``None`` where the host cannot say."""
    paths = [Path(p) for p in paths]
    out: dict[Path, bool | None] = dict.fromkeys(paths)
    lsattr = shutil.which("lsattr")
    if lsattr is None:
        return out
    for index in range(0, len(paths), _BATCH):
        batch = paths[index : index + _BATCH]
        try:
            done = subprocess.run(
                [lsattr, "-d", *map(str, batch)], capture_output=True, text=True, timeout=300, check=False
            )
        except (OSError, subprocess.SubprocessError):
            continue
        for line in done.stdout.splitlines():
            flags, _, name = line.partition(" ")
            if name.strip():
                out[Path(name.strip())] = "i" in flags
    return out


def is_immutable(path: Path) -> bool | None:
    """Whether the immutable attribute is set, or ``None`` when the host cannot say."""
    return immutable_flags([path])[Path(path)]


def immutable(roots: Iterable[Path], *, mode: str | None = None) -> dict:
    """Ask for the immutable attribute on every unshared member; report what the host granted."""
    chosen = selected_mode(mode)
    files, dirs = _members(roots)
    targets = [p for p in files if not _shared(p)] + dirs
    record = {"mode": chosen, "attempted": False, "requested": len(targets), "granted": 0, "error": None}
    if chosen == "off" or not targets:
        return record
    command = _chattr_command(chosen)
    if command is None:
        record["error"] = "chattr" if chosen == "attempt" else "chattr or sudo"
        record["error"] += " is not available on this host"
        return record
    record["attempted"] = True
    errors: list[str] = []
    for index in range(0, len(targets), _BATCH):
        batch = [str(p) for p in targets[index : index + _BATCH]]
        try:
            done = subprocess.run([*command, "+i", *batch], capture_output=True, text=True, timeout=300, check=False)
        except (OSError, subprocess.SubprocessError) as exc:
            errors.append(str(exc))
            continue
        if done.returncode != 0:
            lines = (done.stderr or done.stdout or "").strip().splitlines()
            errors.append(lines[-1] if lines else f"exit status {done.returncode}")
    record["granted"] = sum(1 for flag in immutable_flags(targets).values() if flag)
    if errors:
        record["error"] = errors[0]
    return record


def seal_tree(roots: Iterable[Path], *, mode: str | None = None) -> dict:
    """Read-only first (always possible for the owner), then the immutable attribute where permitted."""
    roots = [Path(root) for root in roots]
    protected = read_only(roots)
    locked = immutable(roots, mode=mode)
    locked["complete"] = locked["attempted"] and locked["granted"] == locked["requested"]
    return {"read_only": protected, "immutable": locked}
