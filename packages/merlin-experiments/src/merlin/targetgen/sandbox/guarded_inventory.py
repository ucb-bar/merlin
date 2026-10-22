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
        self.max_roots, self.max_members = max_roots, max_members
        self._members = 0
        self._closed = False

    def close(self):
        with self._lock:
            for guard, _ in self._entries.values():
                guard.close()
            self._entries.clear()
            self._members = 0
            self._closed = True

    def __del__(self):
        self.close()

    def tree(self, root: Path):
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
            current = self._entries.get(root)
            if current is not None:
                guard, facts = current
                if guard.unchanged():
                    return facts
                guard.close()
                self._members -= len(facts.members)
                del self._entries[root]
            if root in self._unsupported or len(self._entries) >= self.max_roots:
                return None
            guard = None
            try:
                # The existing scanner refuses symlinks/special files. Its caller
                # retains historical leaf-referent behavior through the fallback.
                members = snapshot._scanned_members(root)
                if len(members) + self._members > self.max_members:
                    if len(self._unsupported) < self.max_roots:
                        self._unsupported.add(root)
                    return None
                guard = snapshot._RosterGuard(root, members)
                if not guard.unchanged():
                    guard.close()
                    if len(self._unsupported) < self.max_roots:
                        self._unsupported.add(root)
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
                self._entries[root] = guard, facts
                self._members += len(members)
                return facts
            except (OSError, RuntimeError):
                if guard is not None:
                    guard.close()
                if len(self._unsupported) < self.max_roots:
                    self._unsupported.add(root)
                return None
