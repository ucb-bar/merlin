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


def _blocks(body: Mapping[str, Any] | None, key: str) -> tuple[dict[str, Any], ...]:
    """Every well-formed mapping in the list at ``key``; ``()`` when the body declares none.

    ABSENT and EMPTY are deliberately the same answer here. They are different facts -- "this device has
    no such thing" versus "the extractor found none" -- but nothing in the body distinguishes them, so
    reporting them apart would be inventing the distinction. A caller that needs it asks the extractor's
    own status field, never the length of this tuple.
    """
    return tuple(b for b in (body or {}).get(key) or () if isinstance(b, Mapping))


def interfaces(body: Mapping[str, Any] | None) -> tuple[dict[str, Any], ...]:
    """Every interface block, for a caller that enumerates rather than looks one up by name."""
    return _blocks(body, "interfaces")


def arrays(body: Mapping[str, Any] | None) -> tuple[dict[str, Any], ...]:
    """The compute arrays a body describes. Absent on every family that declares no array."""
    return _blocks(body, "arrays")


def memories(body: Mapping[str, Any] | None) -> tuple[dict[str, Any], ...]:
    """The on-chip stores a body describes, unfiltered and unscored.

    ``targetgen.address_space`` is the reader that turns these into stores with roles and capacities;
    this is only the shape-safe way to reach the list.
    """
    return _blocks(body, "memories")


def datapaths(body: Mapping[str, Any] | None) -> tuple[dict[str, Any], ...]:
    """The datapaths a body describes.

    Note for any caller tempted to key off ``name``: the naming AXIS differs between families -- some
    name a datapath by its ROLE (``input``, ``accumulator``) and some by its FORMAT (``int8``,
    ``float8``) -- so a name is not a role and must not be read as one.
    """
    return _blocks(body, "datapaths")


def timing(body: Mapping[str, Any] | None) -> tuple[dict[str, Any], ...]:
    """The per-module timing records a body describes.

    These are register-chain depths of a MODULE, not per-instruction latencies. They corroborate a
    declared cost; they do not supply one.
    """
    return _blocks(body, "timing")


def simt(body: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The SIMT geometry block, or ``None`` on a body that declares none."""
    block = (body or {}).get("simt")
    return block if isinstance(block, Mapping) else None


def source(body: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """What the facts were extracted FROM, normalised to a mapping.

    One family records this as a bare string rather than a block, so every caller that reached for
    ``body["source"]["kind"]`` crashed on it. A string is returned as ``{"kind": <the string>}`` so the
    shape is one shape; absent stays ``None``.
    """
    block = (body or {}).get("source")
    if isinstance(block, Mapping):
        return dict(block)
    if isinstance(block, str) and block:
        return {"kind": block}
    return None
