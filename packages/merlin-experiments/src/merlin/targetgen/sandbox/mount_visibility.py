"""Ordered bubblewrap mount replay and per-call path visibility indexing."""

from __future__ import annotations

import os
import stat
from pathlib import Path

_EXPOSE_OPS = ("--ro-bind", "--bind", "--dev-bind", "--ro-bind-try", "--bind-try", "--dev-bind-try")
_HIDE_DEST_OPS = ("--tmpfs",)  # single-arg-dest hide ops
_DEVNULL = "/dev/null"


def _skipped_try(op: str, source: str) -> bool:
    """Whether bwrap skips this bind: a ``*-try`` bind whose source is missing.

    bwrap stat()s the source, following symlinks (a dangling link is missing too), and skips the op
    only on ENOENT; any other failure aborts the launch. A source this process cannot stat for another
    reason (EACCES) may still be mounted or skipped by bwrap, which holds capabilities in its own user
    namespace, so it stays in the plan: a path it serves then either tests as exposed
    (:meth:`_MountVisibility.is_exposed`), keeping its mask whichever way bwrap decides, or cannot be
    resolved at all and the plan is refused.
    """
    if not op.endswith("-try"):
        return False
    try:
        os.stat(source)
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return False


def _mounts(argv: list[str], *, present_only: bool = True) -> list[tuple[str, str, str]]:
    """Parse the argv into ordered (state, source, dest) mount ops, where state is 'expose' or 'hide'.
    Only the ops that affect path visibility are kept; everything else (flags, --chdir, --unsetenv…) is
    ignored. The SOURCE is carried because a bind's dest is not always its own host path (see
    :func:`is_exposed`); a hide op has no meaningful source and reports ``""``.

    A ``*-try`` bind bwrap would skip (:func:`_skipped_try`) mounts nothing, so the earlier mount keeps
    serving its destination; ``present_only=False`` keeps it, for callers asking what could ever be
    bound once the source appears."""
    ops: list[tuple[str, str, str]] = []
    i = 0
    n = len(argv)
    while i < n:
        a = argv[i]
        if a in _EXPOSE_OPS and i + 2 < n:
            src, dest = argv[i + 1], argv[i + 2]
            if not (present_only and _skipped_try(a, src)):
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


class _UnsettledPath(RuntimeError):
    """Guest symlink traversal did not settle on one serving bind."""


class _MountVisibility:
    """One-call mount index; never caches filesystem visibility or mutable argv.

    Precedence is bwrap's own: mount operations apply in argv order and each
    one stacks over its destination, so a path is served by the LAST operation
    whose destination equals the path or is one of its ancestors. A later,
    shallower bind therefore shadows every earlier, deeper mount below it,
    overlays included. Path-equivalent destinations share a slot holding the
    latest operation. Guest-visible symlinks can redirect to a later, deeper
    frozen mount.
    """

    def __init__(self, argv: list[str], mounts: list[tuple[str, str, str]] | None = None):
        self._destinations: dict[tuple[str, ...], tuple[int, str, str, str]] = {}
        self._next_index = 0
        for state, src, dest in _mounts(argv) if mounts is None else mounts:
            self.append(state, src, dest)

    def append(self, state: str, src: str, dest: str) -> None:
        # Indices only grow, so the latest operation on a destination always replaces its slot.
        self._destinations[Path(dest).parts] = (self._next_index, state, src, dest)
        self._next_index += 1

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

    def serving(self, path: Path) -> tuple[Path, Path] | None:
        """``(bind source, source path)`` of the bind that serves ``path``; ``None`` when hidden.

        Raises :class:`_UnsettledPath` when guest symlinks cycle or do not settle.
        """
        current = path
        seen: set[Path] = set()
        # The same bounded traversal as filesystem symlink resolution.
        for _ in range(41):
            if current in seen:
                break
            seen.add(current)
            best = None
            parts = current.parts
            for length in range(len(parts), 0 if current.anchor else -1, -1):
                record = self._destinations.get(parts[:length])
                if record is not None and (best is None or record[0] > best[0]):
                    best = record
            if best is None or best[1] == "hide":
                return None
            _, _, src, dest = best
            relative = current.relative_to(dest)
            alias = self._source_alias(Path(src), Path(dest), relative)
            if alias is not None:
                current = alias
                continue
            return Path(src), Path(src) / relative
        raise _UnsettledPath(path)

    def is_exposed(self, path: Path) -> bool:
        try:
            served = self.serving(path)
        except _UnsettledPath:
            return True  # cycles and unstable paths fail closed as exposed, never as a clean mask
        if served is None:
            return False
        # A frozen snapshot may be mounted over a live tree. Test the
        # effective source bytes, not the destination's host contents.
        try:
            return served[1].exists()
        except PermissionError:
            return True  # locked-but-present still exposes content
        except OSError:
            return False


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


def _mount_op_args(argv: list[str]) -> list[tuple[list[str], tuple[str, str, str]]]:
    """Each visibility-affecting op's raw arguments beside its parsed form (the walk of ``_mounts``,
    which also drops a skipped ``*-try`` bind)."""
    ops: list[tuple[list[str], tuple[str, str, str]]] = []
    i = 0
    while i < len(argv):
        if argv[i] in _EXPOSE_OPS and i + 2 < len(argv):
            args = argv[i : i + 3]
        elif (argv[i] in _HIDE_DEST_OPS or argv[i] in ("--dev", "--proc")) and i + 1 < len(argv):
            args = argv[i : i + 2]
        else:
            i += 1
            continue
        ops += [(args, op) for op in _mounts(args)]
        i += len(args)
    return ops


def _shadowed(destination: Path, later: list[tuple[list[str], tuple[str, str, str]]]) -> bool:
    """Whether a later op on ``destination`` or one of its ancestors stacks over a mount there."""
    return any(Path(dest) == destination or Path(dest) in destination.parents for _, (_, _, dest) in later)


class _CallPathResolution:
    """Resolve shared ancestors once in one stable mount-plan observation.

    Nothing survives this call. Symlinks and unusual filesystem errors use the
    ordinary resolver, including its loop checks. This does not memoize file
    existence, kinds, mount visibility, or a completed sandbox policy.
    """

    def __init__(self):
        self._resolved: dict[str, str] = {}

    def _resolve(self, path: str) -> str:
        cached = self._resolved.get(path)
        if cached is not None:
            return cached
        parent, name = os.path.split(path)
        if not name:
            result = str(Path(path).resolve())
        else:
            resolved_parent = self._resolve(parent)
            if name == "..":
                result = os.path.dirname(resolved_parent)
            else:
                result = os.path.join(resolved_parent, name)
                try:
                    mode = os.lstat(result).st_mode
                except OSError:
                    result = str(Path(path).resolve())
                else:
                    if stat.S_ISLNK(mode):
                        result = str(Path(path).resolve())
        self._resolved[path] = result
        return result

    def resolve(self, path: Path) -> Path:
        # Long declarations share ancestors. String keys avoid reconstructing
        # and hashing Path component tuples at every recursive prefix lookup.
        return Path(self._resolve(str(path.absolute())))
