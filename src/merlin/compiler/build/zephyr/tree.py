"""Read configuration capabilities from one explicitly selected Zephyr tree.

Instances belong to a single build. No process-global cache carries facts from
one tree into another, or retains a missing-tree answer after configuration.
This is symbol discovery, not a replacement for Zephyr's Kconfig evaluator.
"""

from __future__ import annotations

from functools import cached_property
from pathlib import Path


class TreeConfiguration:
    """Conservative capabilities of a selected tree, or an unselected tree."""

    def __init__(self, base: Path | None):
        self.base = Path(base) if base is not None else None

    @cached_property
    def declarations(self) -> tuple[tuple[str, ...], ...]:
        if self.base is None:
            return ()
        roots = (self.base / "arch", self.base / "kernel", self.base / "subsys" / "debug")
        return tuple(
            tuple(path.read_text(errors="replace").splitlines())
            for root in roots
            if root.is_dir()
            for path in root.rglob("Kconfig*")
            if path.is_file()
        )

    @cached_property
    def symbols(self) -> frozenset[str]:
        found = set()
        for lines in self.declarations:
            for line in lines:
                head, _, name = line.strip().partition(" ")
                if head in ("config", "menuconfig") and name and " " not in name:
                    found.add(name)
        return frozenset(found)

    def has(self, symbol: str) -> bool:
        return symbol in self.symbols

    def default_int(self, symbol: str) -> int | None:
        """Read an unconditional literal default; conditional values stay unknown."""
        for lines in self.declarations:
            in_symbol = False
            for line in lines:
                head, _, value = line.strip().partition(" ")
                if head in ("config", "menuconfig"):
                    in_symbol = value == symbol
                    continue
                if in_symbol and head == "default":
                    value = value.strip()
                    return int(value) if value.isdigit() else None
        return None
