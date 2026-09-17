"""Read one named block out of an RTL-facts body -- the lookup every facts consumer used to hand-roll.

A facts body lists its interfaces as ``[{"name": ..., ...}, ...]``. Measured 2026-09-14: about a dozen
modules found a block by name with their own ``next((i for i in ... if i.get("name") == ...), ...)``,
each choosing its own answer for a body with no ``interfaces`` key, a ``None`` list, or a non-dict entry.
This is the one answer: absent means ``None``, and malformed entries are skipped rather than crashed on.
Dependency-free on purpose, so any module can import it at top level without an import cycle.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def interface(body: Mapping[str, Any] | None, name: str) -> dict[str, Any] | None:
    """The interface block named ``name`` in a facts ``body``, or ``None`` when the body declares none."""
    for block in (body or {}).get("interfaces") or ():
        if isinstance(block, Mapping) and block.get("name") == name:
            return dict(block) if not isinstance(block, dict) else block
    return None
