"""Bounded file snapshots for an original native runtime view."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def document(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical(path, *, exists=True):
    path = Path(path)
    if not path.is_absolute() or path.resolve() != path or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("native owner path must be canonical and unlinked")
    if exists and not path.exists():
        raise ValueError("native owner path is unavailable")
    return path


def pin(path, *, max_bytes):
    path = canonical(path)
    before = path.stat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > max_bytes:
        raise ValueError("native observation member is special or exceeds its byte bound")
    result = {"path": str(path), "sha256": digest(path)}
    if identity(before) != identity(path.stat()):
        raise ValueError("native observation member changed during its read")
    return result


def identity(value):
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns, value.st_mode)


def children(path, *, limit):
    names = []
    with os.scandir(path) as entries:
        for entry in entries:
            if len(names) >= limit:
                raise ValueError("native runtime directory exceeds its file bound")
            names.append(entry.name)
    return sorted(names)


def members(root, *, max_files):
    remaining = max_files

    def visit(path):
        nonlocal remaining
        remaining -= 1
        if remaining < 0:
            raise ValueError("native runtime view exceeds its file bound")
        yield path
        if path.is_dir() and not path.is_symlink():
            for name in children(path, limit=remaining):
                yield from visit(path / name)

    yield from visit(root)


def stamps(root, *, max_files):
    return tuple(
        (path.relative_to(root).as_posix(), identity(path.lstat())) for path in members(root, max_files=max_files)
    )


def inventory(root, *, max_files, max_bytes):
    """Include lookup membership, modes and links; bound reads before hashing."""
    root = canonical(root)
    rows, size = [], 0
    for path in members(root, max_files=max_files):
        before = path.lstat()
        name = path.relative_to(root).as_posix()
        if stat.S_ISLNK(before.st_mode):
            target = os.readlink(path)
            resolved = root / target.lstrip("/") if target.startswith("/") else path.parent / target
            # Guest absolute links resolve inside the guest root, never the host.
            if ".." in Path(target).parts and not resolved.absolute().is_relative_to(root):
                raise ValueError("native runtime link escapes its closed view")
            row = {"name": name, "kind": "symlink", "target": target}
        elif stat.S_ISDIR(before.st_mode):
            row = {"name": name, "kind": "directory", "mode": stat.S_IMODE(before.st_mode)}
        elif stat.S_ISREG(before.st_mode):
            size += before.st_size
            if size > max_bytes:
                raise ValueError("native runtime view exceeds its byte bound")
            row = {
                "name": name,
                "kind": "file",
                "mode": stat.S_IMODE(before.st_mode),
                "bytes": before.st_size,
                "sha256": digest(path),
            }
        else:
            raise ValueError("native runtime view contains a special file")
        if identity(before) != identity(path.lstat()):
            raise ValueError("native runtime view changed during inventory")
        rows.append(row)
    return {"sha256": hashlib.sha256(document(rows).encode()).hexdigest(), "files": len(rows), "bytes": size}


def copy_member(source, destination, *, budget):
    """Copy bytes, never mutable hard links; preserve guest lookup names."""
    source, destination = Path(source), Path(destination)
    before = source.lstat()
    budget[0] -= 1
    if budget[0] < 0:
        raise ValueError("native snapshot exceeds its file bound")
    if stat.S_ISLNK(before.st_mode):
        destination.symlink_to(os.readlink(source))
    elif stat.S_ISDIR(before.st_mode):
        destination.mkdir()
        names = children(source, limit=budget[0])
        for name in names:
            copy_member(source / name, destination / name, budget=budget)
        if names != children(source, limit=budget[0] + len(names)):
            raise ValueError("native source directory changed during snapshot")
    elif stat.S_ISREG(before.st_mode):
        budget[1] -= before.st_size
        if budget[1] < 0:
            raise ValueError("native snapshot exceeds its byte bound")
        source_hash = hashlib.sha256()
        fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as incoming, destination.open("xb") as outgoing:
            while chunk := incoming.read(1024 * 1024):
                outgoing.write(chunk)
                source_hash.update(chunk)
            outgoing.flush()
            os.fsync(outgoing.fileno())
            if identity(before) != identity(os.fstat(incoming.fileno())):
                raise ValueError("native source file changed during snapshot")
        if source_hash.hexdigest() != digest(destination):
            raise ValueError("native copied runtime differs from its selected bytes")
        destination.chmod(0o444 | (stat.S_IMODE(before.st_mode) & 0o111))
    else:
        raise ValueError("native snapshot source is special")
    if identity(before) != identity(source.lstat()):
        raise ValueError("native snapshot source changed during copy")


def seal_directories(root, *, max_files):
    for path in sorted(members(root, max_files=max_files), key=lambda p: len(p.parts), reverse=True):
        if path.is_dir() and not path.is_symlink():
            path.chmod(0o555)
    root.chmod(0o555)
