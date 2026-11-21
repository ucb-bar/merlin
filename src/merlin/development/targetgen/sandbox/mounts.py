"""Per-observation lexical and resolved path indexes for ordered bind mounts."""

from __future__ import annotations

import bisect
import contextvars
import os
import stat
import threading
import time
from bisect import bisect_left
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

# ---- mount_paths -------------------------------------------------------------------------------


class _MountPathIndex:
    """Lexical overlap candidates, retaining the original ordered mount IDs."""

    def __init__(self, groups):
        self.exact = {}
        for mount_id, paths in enumerate(groups):
            for path in paths:
                self.exact.setdefault(path.parts, set()).add(mount_id)
        self.ordered = sorted(self.exact)

    def descendants(self, path):
        found = set()
        prefix = path.parts
        start = bisect.bisect_left(self.ordered, prefix)
        for index in range(start, len(self.ordered)):
            parts = self.ordered[index]
            if parts[: len(prefix)] != prefix:
                break
            found.update(self.exact[parts])
        return found

    def related(self, path):
        found = self.descendants(path)
        parts = path.parts
        # Tuple prefixes preserve pathlib's lexical parent relation, including
        # relative '.' and '..', without allocating every ancestor Path.
        for length in range(len(parts), 0 if path.anchor else -1, -1):
            found.update(self.exact.get(parts[:length], ()))
        return found


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


def _resolved_bind_mounts(
    mounts: list[tuple[str, str, str]], resolution: _CallPathResolution | None = None
) -> list[tuple[Path, Path, set[Path]]]:
    """Resolve each source once within the trusted, stable launch namespace.

    Retain every ordered exposed mount and both lexical/canonical source names.
    This cache ends with the call; later calls must observe changed aliases.
    """
    resolution = resolution or _CallPathResolution()
    sources: dict[Path, set[Path]] = {}
    result = []
    for state, raw_source, raw_destination in mounts:
        if state == "expose":
            source = Path(raw_source).absolute()
            if source not in sources:
                sources[source] = {source, resolution.resolve(source)}
            result.append((source, Path(raw_destination), sources[source]))
    return result


# ---- mount_visibility --------------------------------------------------------------------------
# Ordered bubblewrap mount replay and per-call path visibility indexing.

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


# ---- guarded_inventory -------------------------------------------------------------------------
# Request-owned filesystem facts under the existing kernel roster guard.
#
# No mask, grant, import policy or qualification decision is retained. Every
# cache hit requires unchanged watched names/inodes and mount topology; unsupported
# or indirect trees use the caller's original fresh traversal. One lock coalesces
# concurrent discovery within the owner, never across requests or processes.


def _alias_members(root):
    """Lexical alias inventory: record symlinks, never traverse their referents.

    Parent watches invalidate replacements of skipped symlinks. Regular-file
    watches still detect aliases created outside the selected tree. This reader
    is for runtime mounts only; private-input discovery retains its strict reader.
    """
    if root.resolve() != root:
        raise RuntimeError("alias inventory root is indirect")
    pending = [(root, ())]
    members = []
    while pending:
        directory, prefix = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                parts = (*prefix, entry.name)
                info = entry.stat(follow_symlinks=False)
                if stat.S_ISLNK(info.st_mode):
                    kind, size = "symlink", info.st_size
                elif stat.S_ISREG(info.st_mode):
                    kind, size = "file", info.st_size
                elif stat.S_ISDIR(info.st_mode):
                    kind, size = "directory", None
                    pending.append((Path(entry.path), parts))
                else:
                    raise RuntimeError("alias inventory contains a special file")
                members.append((parts, Path(entry.path), kind, size))
    members.sort(key=lambda member: member[0])
    return members


@dataclass(frozen=True)
class TreeFacts:
    members: tuple
    linked_inodes: frozenset
    linked_members: tuple

    @cached_property
    def _member_index(self):
        rows = tuple(sorted(self.members, key=lambda row: row[0]))
        return tuple(row[0] for row in rows), rows

    @cached_property
    def _linked_index(self):
        rows = tuple(sorted(self.linked_members, key=lambda row: row[0].parts))
        return tuple(row[0].parts for row in rows), rows


def _descendants(index, prefix):
    """Visit only a lexical subtree of already guarded facts, never the disk.

    Tuple components keep sibling prefixes distinct. The caller still checks
    its live kernel guard before and after projection; this is an index of the
    same immutable observation, not a new custody or filesystem cache.
    """
    keys, rows = index
    position = bisect_left(keys, prefix)
    width = len(prefix)
    while position < len(keys):
        key = keys[position]
        if key[:width] != prefix:
            break
        if len(key) > width:
            yield rows[position]
        position += 1


class GuardedInventory:
    """Bounded filesystem inventory, enabled with original snapshot read leases."""

    def __init__(self, max_roots=128, max_members=262144):
        self._pid = os.getpid()
        self._lock = threading.RLock()
        self._entries = {}
        self._views = {}
        self._view_members = 0
        self._unsupported = set()
        self._resource_retry_after = {}
        self.max_roots, self.max_members = max_roots, max_members
        self._members = 0
        self._closed = False
        self._scratch = contextvars.ContextVar("policy_inventory_scratch", default=())

    @contextmanager
    def temporary_root(self, root):
        """Small per-command writable trees are inspected afresh, never watched."""
        token = self._scratch.set((*self._scratch.get(), Path(root)))
        try:
            yield
        finally:
            self._scratch.reset(token)

    def adopt_snapshot(self, entry):
        """Borrow a live original snapshot's watches without owning their lifetime."""
        from merlin.common.storage import bundle_snapshots as B

        if type(entry) is not B.SharedEntry:
            return False
        with self._lock:
            if self._closed or self._pid != os.getpid():
                return False
            root = entry.root
            if not entry.reusable(root, entry.repo, entry.policy_bytes):
                return False
            current = self._entries.get(root)
            if current is not None and getattr(current[0], "_snapshot_owner", None) is entry:
                return current[0].unchanged()
            members = tuple(entry.guard.scanned)
            descendants = [key for key in self._entries if self._root(key).is_relative_to(root)]
            replaced = sum(len(self._entries[key][1].members) for key in descendants)
            if len(members) + self._members - replaced > self.max_members:
                return False
            linked = []
            for _, path, kind, _ in members:
                if kind == "file":
                    info = path.stat(follow_symlinks=False)
                    if not stat.S_ISREG(info.st_mode):
                        return False
                    if info.st_nlink > 1:
                        linked.append((path, (info.st_dev, info.st_ino)))
            if not entry.reusable(root, entry.repo, entry.policy_bytes):
                return False
            view = B._GuardView(entry.guard)
            view._snapshot_owner = entry
            for key in descendants:
                guard, facts = self._entries.pop(key)
                guard.close()
                self._members -= len(facts.members)
                self._drop_views(key)
            self._entries[root] = view, TreeFacts(members, frozenset(x[1] for x in linked), tuple(linked))
            self._members += len(members)
            return True

    def close(self):
        with self._lock:
            for guard, _ in self._entries.values():
                guard.close()
            self._entries.clear()
            self._views.clear()
            self._view_members = 0
            self._resource_retry_after.clear()
            self._members = 0
            self._closed = True

    def __del__(self):
        self.close()

    def tree(self, root: Path):
        return self._tree(root, nofollow=False)

    def tree_nofollow(self, root: Path):
        """Runtime alias facts; skipped symlinks confer no input authority."""
        return self._tree(root, nofollow=True)

    @staticmethod
    def _root(key):
        return key[0] if isinstance(key, tuple) else key

    @staticmethod
    def _project(facts, parent, root, *, nofollow):
        prefix = root.relative_to(parent).parts
        members = tuple(
            (parts[len(prefix) :], path, kind, size)
            for parts, path, kind, size in _descendants(facts._member_index, prefix)
        )
        # A lexical runtime inventory must never authorize strict private-input
        # traversal through a symlink. Link-free subtrees have identical readers.
        if not nofollow and any(kind == "symlink" for _, _, kind, _ in members):
            return None
        linked = tuple(_descendants(facts._linked_index, root.parts))
        return TreeFacts(members, frozenset(inode for _, inode in linked), linked)

    def _drop_views(self, source):
        for key, (owner, facts) in list(self._views.items()):
            if owner == source:
                self._view_members -= len(facts.members)
                del self._views[key]

    def _ancestor(self, key, root, *, nofollow):
        view = self._views.get(key)
        if view is not None:
            owner, facts = view
            entry = self._entries.get(owner)
            if entry is not None and entry[0].unchanged():
                return facts
            self._view_members -= len(facts.members)
            del self._views[key]
        for owner, (guard, facts) in sorted(self._entries.items(), key=lambda item: -len(self._root(item[0]).parts)):
            parent = self._root(owner)
            if not root.is_relative_to(parent) or not guard.unchanged():
                continue
            projected = self._project(facts, parent, root, nofollow=nofollow)
            if projected is None or not guard.unchanged():
                continue
            if len(self._views) < self.max_roots and self._view_members + len(projected.members) <= self.max_members:
                self._views[key] = owner, projected
                self._view_members += len(projected.members)
            return projected
        return None

    def _tree(self, root: Path, *, nofollow):
        from merlin.common.storage import bundle_snapshots as snapshot
        from merlin.common.storage import read_leases as read_lease_digest

        if not read_lease_digest.enabled():
            return None
        root = Path(root)
        if any(root.is_relative_to(scratch) for scratch in self._scratch.get()):
            return None
        with self._lock:
            if self._closed or self._pid != os.getpid():
                return None
            # Revalidate canonical spelling on every hit, including parent aliases.
            if not root.is_absolute() or root.resolve() != root or not root.is_dir():
                return None
            key = (root, "nofollow") if nofollow else root
            current = self._entries.get(key)
            if current is not None:
                guard, facts = current
                if guard.unchanged():
                    return facts
                guard.close()
                self._members -= len(facts.members)
                del self._entries[key]
                self._drop_views(key)
            inherited = self._ancestor(key, root, nofollow=nofollow)
            if inherited is not None:
                return inherited
            if key in self._unsupported or len(self._entries) >= self.max_roots:
                return None
            retry_after = self._resource_retry_after.get(key)
            if retry_after is not None:
                if time.monotonic() < retry_after:
                    return None
                del self._resource_retry_after[key]
            guard = None
            try:
                # Strict private discovery and no-follow runtime aliases have
                # separate cache identities and race-closing rescans.
                scanner = _alias_members if nofollow else snapshot._scanned_members
                members = scanner(root)
                descendants = [owner for owner in self._entries if self._root(owner).is_relative_to(root)]
                replaced_members = sum(len(self._entries[owner][1].members) for owner in descendants)
                if len(members) + self._members - replaced_members > self.max_members:
                    if len(self._unsupported) < self.max_roots:
                        self._unsupported.add(key)
                    return None
                guard = snapshot._RosterGuard(root, members, scanner=scanner)
                if not guard.unchanged():
                    guard.close()
                    if guard.resource_errno is not None:
                        # No facts survived the unwatched interval. A later call
                        # may rescan and establish a NEW guard after capacity
                        # returns; coalesce retries under this owner's lock.
                        if len(self._resource_retry_after) < self.max_roots:
                            self._resource_retry_after[key] = time.monotonic() + 30.0
                    elif len(self._unsupported) < self.max_roots:
                        self._unsupported.add(key)
                    return None
                linked = set()
                linked_members = []
                for _, path, kind, _ in members:
                    if kind == "file":
                        info = path.stat(follow_symlinks=False)
                        if not stat.S_ISREG(info.st_mode):
                            guard.close()
                            return None
                        if info.st_nlink > 1:
                            linked.add((info.st_dev, info.st_ino))
                            linked_members.append((path, (info.st_dev, info.st_ino)))
                # File watches include attrib: outside hardlinks invalidate too.
                if not guard.unchanged():
                    guard.close()
                    return None
                facts = TreeFacts(tuple(members), frozenset(linked), tuple(linked_members))
                # Only after the complete replacement guard is armed and checked
                # may its covered child guards close. No unwatched fact survives.
                for owner in descendants:
                    previous_guard, previous_facts = self._entries.pop(owner)
                    previous_guard.close()
                    self._members -= len(previous_facts.members)
                    self._drop_views(owner)
                self._entries[key] = guard, facts
                self._members += len(members)
                return facts
            except (OSError, RuntimeError):
                if guard is not None:
                    guard.close()
                if len(self._unsupported) < self.max_roots:
                    self._unsupported.add(key)
                return None
