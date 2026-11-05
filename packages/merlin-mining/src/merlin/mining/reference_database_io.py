"""Pinned input staging and exact-byte storage for offline reference databases."""

from __future__ import annotations

import hashlib
import json
import shutil
import stat
import subprocess
import tarfile
import zipfile
from pathlib import Path, PurePosixPath


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def safe_member(name):
    relative = PurePosixPath(name)
    if not name or relative.is_absolute() or ".." in relative.parts or "\\" in name:
        raise ValueError("unsafe archive member")
    return relative


def copy_checked(source, destination, expected):
    source, destination = Path(source), Path(destination)
    if source.is_symlink() or not source.is_file() or digest(source) != expected:
        raise ValueError("source differs from its selected hash")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.is_symlink() or digest(destination) != expected:
            raise ValueError("existing destination differs")
    else:
        shutil.copyfile(source, destination)
    if digest(source) != expected or digest(destination) != expected:
        raise ValueError("input changed during staging")


def unpack_selected(archive, destination, members, *, archive_format):
    """Extract only explicitly selected ordinary members; never follow archive links."""
    archive, destination = Path(archive), Path(destination)
    selected = {name: str(safe_member(name)) for name in members}
    if len(selected) != len(members):
        raise ValueError("duplicate member selection")
    found = set()

    def consume(name, size, stream):
        if name in found or not 0 <= size <= 4 << 30:
            raise ValueError("duplicate or excessive selected archive member")
        found.add(name)
        path = destination / selected[name]
        path.parent.mkdir(parents=True, exist_ok=True)
        if any(parent.is_symlink() for parent in (path, *path.parents)):
            raise ValueError("indirect extraction destination")
        temporary = path.with_name(path.name + ".partial")
        if temporary.exists():
            raise ValueError("unfinished extraction already exists")
        with temporary.open("xb") as output:
            shutil.copyfileobj(stream, output, 1 << 20)
        if temporary.stat().st_size != size:
            raise ValueError("truncated archive member")
        if path.exists():
            if digest(path) != digest(temporary):
                raise ValueError("existing extracted input differs")
            temporary.unlink()
        else:
            temporary.replace(path)

    if archive_format == "zip":
        with zipfile.ZipFile(archive) as source:
            for item in source.infolist():
                safe_member(item.filename)
                if item.filename not in selected:
                    continue
                if item.is_dir() or stat.S_ISLNK(item.external_attr >> 16):
                    raise ValueError("selected member is not ordinary")
                with source.open(item) as stream:
                    consume(item.filename, item.file_size, stream)
    elif archive_format == "tar.zst":
        with subprocess.Popen(["zstd", "-d", "-c", str(archive)], stdout=subprocess.PIPE) as process:
            with tarfile.open(fileobj=process.stdout, mode="r|") as source:
                for item in source:
                    safe_member(item.name)
                    if item.name not in selected:
                        continue
                    if not item.isfile():
                        raise ValueError("selected member is not ordinary")
                    with source.extractfile(item) as stream:
                        consume(item.name, item.size, stream)
            process.stdout.close()
            if process.wait():
                raise ValueError("archive decompression failed")
    else:
        raise ValueError("unsupported archive format")
    if found != set(selected):
        raise ValueError("selected archive member is absent")


def store_body(root, kind, payload, suffix):
    """Same bytes deduplicate only within one representation, never by semantic guesses."""
    sha = hashlib.sha256(payload).hexdigest()
    identity = kind + "-" + sha
    relative = Path("kernels") / kind / sha / ("body" + suffix)
    path = Path(root) / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError("body identity collision")
    else:
        path.write_bytes(payload)
    return dict(id=identity, kind=kind, body_sha256=sha, path=relative.as_posix(), size_bytes=len(payload))
