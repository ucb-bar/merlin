"""Exact file handoffs for explicitly selected execution inputs.

Streaming private copies prevent mutable source paths from selecting new bytes
after dispatch. Callers own parent-directory protection, resource locking,
complete input membership and consumer correspondence. These helpers confer
no execution, hardware, numerical or timing authority.
"""

from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path


class FileHandoffError(ValueError):
    """An explicitly selected file or private snapshot changed."""


@dataclass(frozen=True)
class PinnedFile:
    path: Path
    sha256: str

    def __post_init__(self):
        if (
            not isinstance(self.path, Path)
            or not self.path.is_absolute()
            or self.path.resolve() != self.path
            or self.path.is_symlink()
            or type(self.sha256) is not str
            or len(self.sha256) != 64
            or any(char not in "0123456789abcdef" for char in self.sha256)
        ):
            raise FileHandoffError("file selection requires a canonical path and explicit SHA-256")


def _read_digest(path: Path) -> str:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise FileHandoffError("execution input is not a regular file")
            digest = hashlib.sha256()
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise FileHandoffError("execution input cannot be reopened") from error
    return digest.hexdigest()


def verify_files(files: tuple[PinnedFile, ...]) -> None:
    """Reopen every exact selected regular file; this is a byte check only."""
    if type(files) is not tuple or not files or any(type(item) is not PinnedFile for item in files):
        raise FileHandoffError("file handoff requires a nonempty exact selection")
    if len({item.path for item in files}) != len(files):
        raise FileHandoffError("file handoff repeats a selected path")
    for item in files:
        # Check again: a parent may have been replaced after declaration.
        if item.path.resolve() != item.path or item.path.is_symlink() or _read_digest(item.path) != item.sha256:
            raise FileHandoffError("selected execution file changed")


def snapshot_files(files: tuple[PinnedFile, ...], destination: Path) -> tuple[PinnedFile, ...]:
    """Copy complete selected bytes into a new owner-private directory.

    Creation is exclusive. Failed attempts remain caller-owned diagnostic
    artifacts and return no snapshot selection. The caller must protect the
    enclosing directory against rename and retain the relevant resource lock.
    """
    verify_files(files)
    if not isinstance(destination, Path) or not destination.is_absolute() or destination.resolve() != destination:
        raise FileHandoffError("snapshot destination must be an explicit canonical path")
    try:
        destination.mkdir(mode=0o700, parents=False, exist_ok=False)
        observed = destination.stat()
        if observed.st_uid != os.geteuid() or stat.S_IMODE(observed.st_mode) != 0o700:
            raise FileHandoffError("snapshot directory is not owner-private")
        result = []
        for index, item in enumerate(files):
            output = destination / f"input-{index:04d}-{item.path.name}"
            descriptor = os.open(item.path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            digest = hashlib.sha256()
            with os.fdopen(descriptor, "rb") as source, output.open("xb") as target:
                if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                    raise FileHandoffError("execution input is not a regular file")
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
                    target.write(chunk)
                target.flush()
                os.fsync(target.fileno())
            output.chmod(0o444)
            if digest.hexdigest() != item.sha256:
                raise FileHandoffError("execution source changed while copying")
            result.append(PinnedFile(output, item.sha256))
    except OSError as error:
        raise FileHandoffError("private execution snapshot could not be created") from error
    verify_files(files)
    snapshots = tuple(result)
    verify_files(snapshots)
    return snapshots
