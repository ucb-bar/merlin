"""Indexed lexical path ancestry without filesystem resolution or authority.

These helpers select copy roots and detect overlap. They neither validate paths
nor grant access: callers retain their symlink, ownership and containment checks.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


def minimal_roots(paths: Iterable[Path]) -> tuple[Path, ...]:
    """Same depth/path ordering as a parent-first scan, with indexed membership.

    Sorting costs O(n log n); ancestor queries cost O(n * path depth), rather
    than comparing every input to every previously retained unrelated root.
    Absolute lexical spellings preserve symlink aliases and parent components.
    """
    sources = sorted({path.absolute() for path in paths}, key=lambda path: (len(path.parts), str(path)))
    retained: set[Path] = set()
    ordered = []
    for source in sources:
        if source in retained or any(parent in retained for parent in source.parents):
            continue
        ordered.append(source)
        retained.add(source)
    return tuple(ordered)


@dataclass(frozen=True)
class LexicalPathIndex:
    """Exact roots and their strict ancestors; no implicit canonical aliases."""

    roots: frozenset[Path]
    ancestors: frozenset[Path]

    @classmethod
    def from_paths(cls, paths: Iterable[Path]) -> LexicalPathIndex:
        roots = frozenset(path.absolute() for path in paths)
        ancestors = set()
        for root in roots:
            path, parent = root, root.parent
            # Every previously visited parent already contributed its whole
            # ancestor chain. Explicit sibling leaves need not rebuild those
            # same deep Path objects for each input. This is lexical data only;
            # callers still perform every filesystem and ownership check.
            while parent != path and parent not in ancestors:
                ancestors.add(parent)
                path, parent = parent, parent.parent
        return cls(roots, frozenset(ancestors))

    def contains(self, path: Path) -> bool:
        """Equal to or below a declared root; never admit its parent."""
        path = path.absolute()
        return path in self.roots or any(parent in self.roots for parent in path.parents)

    def overlaps(self, path: Path) -> bool:
        """True for equal, ancestor or descendant spellings in O(path depth)."""
        path = path.absolute()
        return path in self.ancestors or self.contains(path)
