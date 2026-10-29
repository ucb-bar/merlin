"""A target's declared RTL checkout revision and submodule gitlinks, checked before any read.

A descriptor may pin the elaboration it selects to an exact parent commit and to the gitlinks
recorded in that commit. Both are optional: a declaration with neither is accepted unchanged. When
present they are verified against the checkout by content -- branch names move and forks share
them -- and any disagreement refuses the source rather than reading a different elaboration.
"""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .introspect import RtlSourceInvalid


def verify_declared_revision(root: Path, source_revision: str | None, gitlinks: tuple[tuple[str, str], ...]) -> None:
    """Check optional target-authored git identities before reading an elaboration.

    A gitlink is read from the parent commit, so this check does not depend on
    whether the potentially large submodule has been initialized locally.
    """
    if source_revision is None and not gitlinks:
        return
    if gitlinks and source_revision is None:
        raise RtlSourceInvalid("selected RTL gitlinks require a parent source_revision")

    def git(*args: str) -> str:
        try:
            result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=False)
        except OSError as exc:
            raise RtlSourceInvalid(f"selected RTL checkout cannot run git at {root}") from exc
        if result.returncode:
            raise RtlSourceInvalid(f"selected RTL checkout cannot read git identity at {root}")
        return result.stdout.strip()

    if source_revision is not None:
        if len(source_revision) != 40 or any(c not in "0123456789abcdef" for c in source_revision):
            raise RtlSourceInvalid("selected RTL source_revision must be a full lowercase git commit")
        if git("rev-parse", "HEAD") != source_revision:
            raise RtlSourceInvalid(f"selected RTL checkout revision differs from {source_revision}")
        if git("status", "--porcelain", "--untracked-files=no", "--ignore-submodules=all"):
            raise RtlSourceInvalid("selected RTL checkout has modified tracked source")
    for path, expected in gitlinks:
        parts = Path(path).parts
        if (
            not parts
            or Path(path).is_absolute()
            or any(part in (".", "..") for part in parts)
            or len(expected) != 40
            or any(c not in "0123456789abcdef" for c in expected)
        ):
            raise RtlSourceInvalid("selected RTL gitlink declaration is invalid")
        row = git("ls-tree", "HEAD", "--", path)
        fields = row.partition("\t")
        if fields[2] != path or fields[0] != f"160000 commit {expected}":
            raise RtlSourceInvalid(f"selected RTL gitlink {path} differs from {expected}")


def declared_revision(
    block: Mapping[str, Any], root: Path, *, where: str
) -> tuple[str | None, tuple[tuple[str, str], ...]]:
    """Parse ``source_revision`` / ``gitlinks`` from an elaboration block and verify them at ``root``."""
    gitlink_rows = block.get("gitlinks", {})
    if not isinstance(gitlink_rows, dict) or any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in gitlink_rows.items()
    ):
        raise RtlSourceInvalid(f"{where} declares invalid RTL gitlinks")
    revision = block.get("source_revision")
    if revision is not None and not isinstance(revision, str):
        raise RtlSourceInvalid(f"{where} declares invalid RTL source_revision")
    gitlinks = tuple(sorted(gitlink_rows.items()))
    verify_declared_revision(root, revision, gitlinks)
    return revision, gitlinks


#: Verified (path, size, mtime_ns) -> sha256, so a pinned 80 MB FIRRTL is hashed once per change.
_DIGESTS: dict[tuple[str, int, int], str] = {}


def _content_sha256(path: Path) -> str:
    import hashlib

    st = path.stat()
    key = (str(path), st.st_size, st.st_mtime_ns)
    if key not in _DIGESTS:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
        _DIGESTS[key] = digest.hexdigest()
    return _DIGESTS[key]


def verify_artifact_digests(source) -> None:
    """Refuse a declared elaboration whose PRESENT artifacts are not the pinned bytes.

    An absent artifact is left to the caller (absence is reported as such elsewhere); an artifact that
    exists with different bytes is a different elaboration and is never read in place of the pinned one.
    """
    if not source.artifact_sha256:
        return
    found = source.artifacts()
    for role, expected in source.artifact_sha256:
        path = Path(found[role])
        if path.is_file() and _content_sha256(path) != expected:
            raise RtlSourceInvalid(
                f"{source.target}: {path} has sha256 {_content_sha256(path)}, but {source.origin} pins the "
                f"{role} of {source.config} to {expected}; refusing a different elaboration"
            )


def declared_artifact_pins(block: Mapping[str, Any], *, where: str) -> tuple[tuple[str, str], ...]:
    """Parse an elaboration block's optional ``artifact_sha256`` (``fir``/``hierarchy`` -> sha256)."""
    pins = block.get("artifact_sha256") or {}
    if not isinstance(pins, Mapping) or any(
        role not in ("fir", "hierarchy")
        or not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
        for role, value in pins.items()
    ):
        raise RtlSourceInvalid(f"{where} artifact_sha256 must map fir/hierarchy to lowercase sha256")
    return tuple(sorted(pins.items()))
