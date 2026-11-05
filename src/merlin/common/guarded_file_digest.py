"""Request-owned exact file digests under continuous kernel change watches.

No owner is discovered from environment or retained globally. Outside an
explicit host scope the ordinary full read remains mandatory. This utility
does not establish a source grant, selected pin, or compiler authority.
"""
from __future__ import annotations

import contextlib
import contextvars
import hashlib
import os
from pathlib import Path
import threading

from .bundle_snapshot import _RosterGuard

_ACTIVE = contextvars.ContextVar("guarded_file_digest_owner", default=None)


def _plain(path):
    path = Path(path)
    if not path.is_absolute() or path.resolve() != path or path.is_symlink() or not path.is_file():
        raise ValueError("guarded digest requires a canonical ordinary file")
    return path


def _read(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class GuardedDigests:
    """One bounded host request's digests, explicitly shared by joined workers.

    Watches are armed before the initial complete read. Any event (including
    overflow), watch failure, namespace/mount change or fork restores complete
    reads. An invalidated guard is never rearmed within this owner. File watches
    also see modifications through external hardlinks. Timestamps never admit
    a digest. Scope exit rechecks all previously read bytes before publication.
    """
    def __init__(self):
        self.pid, self.live = os.getpid(), True
        self._lock, self._entries = threading.RLock(), {}

    def _digest(self, path):
        if self.pid != os.getpid() or not self.live:
            return _read(path)
        with self._lock:
            record = self._entries.get(path)
            if record is not None:
                digest, guard = record
                if guard.unchanged():
                    return digest
                current = _read(path)
                if current != digest:
                    raise ValueError("guarded selected file bytes changed")
                return current
            if len(self._entries) >= 64:
                return _read(path)
            def scan(_root):
                selected = _plain(path)
                return [((selected.name,), selected, "file", selected.stat().st_size)]
            guard = _RosterGuard(path.parent, scan(path.parent), scanner=scan)
            try:
                digest = _read(path)
                if not guard.unchanged():
                    guard.close()
                    return digest
                self._entries[path] = digest, guard
                return digest
            except BaseException:
                guard.close()
                raise

    def validate(self):
        if self.pid != os.getpid() or not self.live:
            raise ValueError("guarded digest owner is no longer live")
        with self._lock:
            for path in tuple(self._entries):
                self._digest(_plain(path))

    @contextlib.contextmanager
    def scope(self):
        if self.pid != os.getpid() or not self.live:
            raise ValueError("guarded digest scope requires its original live process")
        token = _ACTIVE.set(self)
        try:
            yield self
            self.validate()
        finally:
            _ACTIVE.reset(token)

    def close(self):
        if self.pid != os.getpid():
            return
        with self._lock:
            self.live = False
            for _, guard in self._entries.values():
                guard.close()
            self._entries.clear()


def sha256(path):
    """Full read by default; guarded reuse only inside an explicit host scope."""
    path = _plain(path)
    owner = _ACTIVE.get()
    return _read(path) if owner is None else owner._digest(path)
