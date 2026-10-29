"""Process-local original emission identities and exact host-written aliases.

Neither serialized identifiers nor equal bytes can issue an identity. Alias
records retain only bounded host observations within one original live scope.
They establish path/byte custody, not persistent inode or machine semantics.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from weakref import WeakValueDictionary


class IssuedEmission(str):
    """Immutable text; genuine issuance is checked by exact object identity."""

    __slots__ = ("__weakref__",)


class EmissionProducts:
    def __init__(self, maximum_aliases, maximum_bytes=256 * 1024 * 1024):
        self.pid = os.getpid()
        self.maximum_aliases = maximum_aliases
        self.issued = WeakValueDictionary()
        self.aliases = {}
        self.writers = {}
        self.alias_count = 0
        self.maximum_bytes = maximum_bytes
        self.retained_bytes = 0

    def verify(self):
        if os.getpid() != self.pid:
            raise ValueError("original emission products belong to a different host process")

    def identity(self, source):
        self.verify()
        if type(source) is not IssuedEmission:
            return None
        if self.issued.get(id(source)) is not source:
            raise ValueError("emission text has no current original host issuance")
        return id(source)

    def issue(self, source, pin):
        self.verify()
        value = IssuedEmission(source)
        size = sys.getsizeof(value)
        if self.retained_bytes + size > self.maximum_bytes:
            raise ValueError("original emission products exceed the bounded retained memory")
        self.issued[id(value)] = value
        self.alias(value, pin)
        self.retained_bytes += size
        return value

    def alias(self, source, pin):
        if self.identity(source) is None:
            raise ValueError("an artifact alias requires original issued emission text")
        import hashlib

        if pin["sha256"] != hashlib.sha256(source.encode()).hexdigest():
            raise ValueError("artifact alias differs from its original issued emission")
        path = Path(pin["path"])
        rows = self.aliases.get(path, ())
        if any(value is source and expected == pin for value, expected in rows):
            return
        if self.alias_count >= self.maximum_aliases:
            raise ValueError("original emission aliases exceed the bounded live inventory")
        self.aliases.setdefault(path, []).append((source, dict(pin)))
        self.alias_count += 1

    def invalidate(self, path):
        """A host rewrite ends this path's ownership before any new write."""
        self.verify()
        path = Path(path)
        if path.resolve() != path or any(owner.is_symlink() for owner in (path, *path.parents)):
            raise ValueError("artifact writes require canonical plain original paths")
        if path not in self.writers and len(self.writers) >= self.maximum_aliases:
            raise ValueError("original emission writers exceed the bounded live inventory")
        writer = self.writers[path] = object()
        self.alias_count -= len(self.aliases.pop(path, ()))
        return writer

    def verify_write(self, path, writer):
        self.verify()
        if self.writers.get(Path(path)) is not writer:
            raise ValueError("another observed writer replaced the original emission path")

    def read(self, path, text, pin):
        self.verify()
        rows = self.aliases.get(path)
        if rows is None:
            return text  # An arbitrary file does not reconstruct issuance.
        matches = [value for value, expected in rows if expected == pin]
        if len(matches) != 1:
            raise ValueError("artifact path has changed or has ambiguous original emission ownership")
        value = matches[0]
        self.identity(value)
        if text != value:
            raise ValueError("artifact bytes differ from their original issued emission")
        return value

    def cache_key(self, source):
        identity = self.identity(source)
        return source if identity is None else (id(self), identity, source)

    def close(self):
        self.issued.clear()
        self.aliases.clear()
        self.writers.clear()
        self.alias_count = 0
        self.retained_bytes = 0
