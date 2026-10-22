"""Request-owned filesystem facts under the existing kernel roster guard.

No mask, grant, import policy or qualification decision is retained. Every
cache hit requires unchanged watched names/inodes and mount topology; unsupported
or indirect trees use the caller's original fresh traversal. One lock coalesces
concurrent discovery within the owner, never across requests or processes.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import stat
import threading
import time


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


class GuardedInventory:
    """Bounded filesystem inventory, enabled with original snapshot read leases."""

    def __init__(self, max_roots=128, max_members=262144):
        self._pid = os.getpid()
        self._lock = threading.RLock()
        self._entries = {}
        self._unsupported = set()
        self._resource_retry_after = {}
        self.max_roots, self.max_members = max_roots, max_members
        self._members = 0
        self._closed = False

    def close(self):
        with self._lock:
            for guard, _ in self._entries.values():
                guard.close()
            self._entries.clear()
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

    def _tree(self, root: Path, *, nofollow):
        from merlin.common import bundle_snapshot as snapshot
        from merlin.common import read_lease_digest

        if not read_lease_digest.enabled():
            return None
        root = Path(root)
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
                if len(members) + self._members > self.max_members:
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
                    if kind == 'file':
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
                self._entries[key] = guard, facts
                self._members += len(members)
                return facts
            except (OSError, RuntimeError):
                if guard is not None:
                    guard.close()
                if len(self._unsupported) < self.max_roots:
                    self._unsupported.add(key)
                return None
