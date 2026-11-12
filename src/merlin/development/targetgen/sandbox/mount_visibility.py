"""Ordered bubblewrap mount replay and per-call path visibility indexing."""

from __future__ import annotations

import os
from pathlib import Path

_EXPOSE_OPS = ("--ro-bind", "--bind", "--dev-bind", "--ro-bind-try", "--bind-try")
_HIDE_DEST_OPS = ("--tmpfs",)  # single-arg-dest hide ops
_DEVNULL = "/dev/null"


def _mounts(argv: list[str]) -> list[tuple[str, str, str]]:
    """Parse the argv into ordered (state, source, dest) mount ops, where state is 'expose' or 'hide'.
    Only the ops that affect path visibility are kept; everything else (flags, --chdir, --unsetenv…) is
    ignored. The SOURCE is carried because a bind's dest is not always its own host path (see
    :func:`is_exposed`); a hide op has no meaningful source and reports ``""``."""
    ops: list[tuple[str, str, str]] = []
    i = 0
    n = len(argv)
    while i < n:
        a = argv[i]
        if a in _EXPOSE_OPS and i + 2 < n:
            src, dest = argv[i + 1], argv[i + 2]
            ops.append(("hide", "", dest) if src == _DEVNULL else ("expose", src, dest))
            i += 3
            continue
        if a in _HIDE_DEST_OPS and i + 1 < n:
            ops.append(("hide", "", argv[i + 1]))
            i += 2
            continue
        if a in ("--dev", "--proc") and i + 1 < n:
            ops.append(("hide", "", argv[i + 1]))  # devtmpfs/procfs — hides host content under dest
            i += 2
            continue
        i += 1
    return ops


class _MountVisibility:
    """One-call mount index; never caches filesystem visibility or mutable argv.

    Path-equivalent destinations share a slot, but precedence still uses the
    original destination STRING length and ordered mount index, not path depth.
    Guest-visible symlinks can redirect to a later, deeper frozen mount.
    """

    def __init__(self, argv: list[str], mounts: list[tuple[str, str, str]] | None = None):
        self._destinations: dict[tuple[str, ...], tuple[int, int, str, str, str]] = {}
        self._next_index = 0
        for state, src, dest in _mounts(argv) if mounts is None else mounts:
            self.append(state, src, dest)

    def append(self, state: str, src: str, dest: str) -> None:
        record = (len(dest), self._next_index, state, src, dest)
        self._next_index += 1
        key = Path(dest).parts
        previous = self._destinations.get(key)
        if previous is None or record[:2] >= previous[:2]:
            self._destinations[key] = record

    @staticmethod
    def _source_alias(source: Path, destination: Path, relative: Path) -> Path | None:
        """Rewrite the first guest-visible symlink inside this bind, excluding its source root.

        This handles a live ``lib64 -> lib`` alias whose ``lib`` child is replaced
        by a pycache-free frozen grant; a late host pyc is not guest-visible.
        """
        cursor = source
        parts = relative.parts
        for index, part in enumerate(parts):
            cursor /= part
            if not cursor.is_symlink():
                continue
            try:
                link = os.readlink(cursor)
            except OSError:
                return destination / relative  # changed while observed: fail closed
            target = Path(link)
            guest_parent = destination.joinpath(*parts[:index])
            rewritten = target if target.is_absolute() else guest_parent / target
            return Path(os.path.normpath(str(rewritten.joinpath(*parts[index + 1 :]))))
        return None

    def is_exposed(self, path: Path) -> bool:
        current = path
        seen: set[Path] = set()
        # The same bounded traversal as filesystem symlink resolution; cycles
        # and unstable paths fail closed as exposed, never as a clean mask.
        for _ in range(41):
            if current in seen:
                return True
            seen.add(current)
            best = None
            parts = current.parts
            for length in range(len(parts), 0 if current.anchor else -1, -1):
                record = self._destinations.get(parts[:length])
                if record is not None and (best is None or record[:2] >= best[:2]):
                    best = record
            if best is None or best[2] == "hide":
                return False
            _, _, _, src, dest = best
            relative = current.relative_to(dest)
            alias = self._source_alias(Path(src), Path(dest), relative)
            if alias is not None:
                current = alias
                continue
            # A frozen snapshot may be mounted over a live tree. Test the
            # effective source bytes, not the destination's host contents.
            mapped = Path(src) / relative
            try:
                return mapped.exists()
            except PermissionError:
                return True  # locked-but-present still exposes content
            except OSError:
                return False
        return True


def path_kind(p: Path) -> str:
    """``dir`` / ``file`` / ``missing`` -- permission-safe.

    A chmod-000 lock makes ``stat()`` raise; treat that as a present dir so a locked answer surface
    is still masked rather than crashing the binder.
    """
    try:
        if p.is_dir():
            return "dir"
        if p.exists():
            return "file"
        return "missing"
    except PermissionError:
        return "dir"
