"""Per-observation lexical and resolved path indexes for ordered bind mounts."""

from __future__ import annotations

import bisect
import os
import stat
from pathlib import Path


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
