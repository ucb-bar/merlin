"""Checked compiler input bytes and bounded tensor-archive identities."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

from merlin.capture.safetensors import read_header


@dataclass(frozen=True)
class _ArchiveInfo:
    header: dict
    payload_start: int
    file_bytes: int
    checked_sha256: str | None


def _archive_info(
    directory: Path,
    name: str,
    *,
    expected_sha256: str | None = None,
    typed: bool = True,
) -> _ArchiveInfo:
    """Read bounded tensor metadata and, when requested, hash without copying payloads."""
    member = directory / name
    if member.is_symlink() or not member.is_file():
        raise ValueError(f"capture member is absent or a symlink: {name}")
    with member.open("rb") as stream:
        before = os.fstat(stream.fileno())
        header, start = read_header(stream) if typed else ({}, 0)
        if start > before.st_size:
            raise ValueError("tensor archive header extends past its file")
        checked_sha256 = None
        if expected_sha256 is not None:
            stream.seek(0)
            checked_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
            if checked_sha256 != expected_sha256:
                raise ValueError(f"compiler input member differs from manifest: {name}")
        after = os.fstat(stream.fileno())
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ValueError("tensor archive changed while checking its metadata")
    return _ArchiveInfo(header, start, before.st_size, checked_sha256)


def _sha(data: bytes | _ArchiveInfo) -> str:
    if isinstance(data, _ArchiveInfo):
        if data.checked_sha256 is None:
            raise ValueError("tensor archive identity has not been checked")
        return data.checked_sha256
    return hashlib.sha256(data).hexdigest()


def _copy_archive(source: Path, destination: Path, info: _ArchiveInfo) -> None:
    """Copy and hash the same bounded chunks before publishing a staged archive."""
    digest, size = hashlib.sha256(), 0
    with source.open("rb") as reader, destination.open("xb") as writer:
        while chunk := reader.read(1024 * 1024):
            writer.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    if size != info.file_bytes or digest.hexdigest() != _sha(info):
        raise ValueError("capture archive changed during staging")


def _read_member(directory: Path, name: str) -> bytes:
    member = directory / name
    if member.is_symlink() or not member.is_file():
        raise ValueError(f"capture member is absent or a symlink: {name}")
    return member.read_bytes()


def _checked_member(directory: Path, receipt: dict, name: str) -> bytes:
    entry = (receipt.get("artifacts") or {}).get(name)
    if not isinstance(entry, dict):
        raise ValueError(f"capture receipt lacks {name}")
    raw = _read_member(directory, name)
    if entry.get("bytes") != len(raw) or entry.get("sha256") != _sha(raw):
        raise ValueError(f"capture member differs from receipt: {name}")
    return raw
