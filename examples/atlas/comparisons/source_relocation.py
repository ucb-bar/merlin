"""Relocate pinned source metadata without changing its hashes or source policy.

Mapping is pure: only absolute path strings and absolute dictionary keys change.
Actual metadata reads admit every path component before touching the filesystem.
A digest authenticates the original bytes, before constructing a relocated view.
"""

from merlin.common.digest import sha256_bytes
import hashlib
import json
import os
from pathlib import Path
import stat


def absolute_path(value):
    path = Path(value)
    if any(word in part.lower() for part in path.parts for word in ('hammer', 'vlsi')):
        raise ValueError('restricted path component')
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError('require an absolute path without parent components')
    return path


def physical_path(value):
    original = Path(value)
    # Screen relative components before making an absolute path or any stat.
    if any(word in part.lower() for part in original.parts for word in ('hammer', 'vlsi')):
        raise ValueError('restricted path component')
    if '..' in original.parts:
        raise ValueError('parent path component')
    path = absolute_path(original.absolute())
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            break
        if stat.S_ISLNK(mode):
            raise ValueError('source metadata requires physical paths without symlinks')
    return path


class SourceRelocation:
    def __init__(self, rules=()):
        self.rules = []
        self._changes = set()
        self.raw_metadata = []
        seen = set()
        for rule in rules:
            if not isinstance(rule, str) or '=' not in rule:
                raise ValueError('relocation must be old_absolute_prefix=new_absolute_prefix')
            old, new = (absolute_path(part) for part in rule.split('=', 1))
            if old in seen:
                raise ValueError('duplicate relocation prefix')
            seen.add(old)
            self.rules.append((old, new))
        self.rules.sort(key=lambda pair: (-len(pair[0].parts), str(pair[0])))

    def path(self, value):
        """Map one absolute path once; never stat either prefix or the result."""
        original = absolute_path(value)
        for old, new in self.rules:
            if original.is_relative_to(old):
                mapped = absolute_path(new / original.relative_to(old))
                if mapped != original:
                    self._changes.add((str(original), str(mapped)))
                return str(mapped)
        return str(original)

    def apply(self, value):
        """Copy a JSON value, rewriting paths and checking dictionary collisions."""
        if isinstance(value, str):
            return self.path(value) if value.startswith('/') else value
        if isinstance(value, list):
            return [self.apply(item) for item in value]
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                mapped_key = self.path(key) if isinstance(key, str) and key.startswith('/') else key
                if mapped_key in result:
                    raise ValueError('relocated dictionary keys collide')
                result[mapped_key] = self.apply(item)
            return result
        return value

    def read_json(self, path, expected_sha256):
        """Admit the explicit actual file, hash its raw bytes, then map its JSON.

        The file argument is already an observed path. It is not mapped again;
        this avoids chained substitution of a path from an earlier mapped view.
        """
        if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
                or any(char not in '0123456789abcdef' for char in expected_sha256)):
            raise ValueError('explicit metadata SHA256 required')
        path = physical_path(path)
        raw = path.read_bytes()
        digest = sha256_bytes(raw)
        if digest != expected_sha256:
            raise ValueError('original metadata digest differs: ' + str(path))
        record = dict(path=str(path), sha256=digest, bytes=len(raw))
        result = self.apply(json.loads(raw))
        self.raw_metadata.append(record)
        return result, record

    def report(self):
        return dict(policy='original metadata bytes pinned; only path values/absolute keys relocated',
                    rules=[dict(original_prefix=str(old), observed_prefix=str(new)) for old, new in self.rules],
                    raw_metadata=list(self.raw_metadata),
                    substitutions=[dict(original=old, observed=new) for old, new in sorted(self._changes)])


def argument_relocator(args):
    """Share one lexical mapping context across a CLI's metadata reads."""
    relocation = getattr(args, '_source_relocation', None)
    if relocation is None:
        relocation = SourceRelocation(getattr(args, 'relocate', ()))
        args._source_relocation = relocation
    return relocation


def read_metadata(args, report, path, digest):
    """Read original pinned bytes, expose a relocated view, retain raw identity."""
    relocation = argument_relocator(args)
    document, _ = relocation.read_json(path, digest)
    report['source_relocation'] = relocation.report()
    return document
