"""Snapshot recorded RTL sources with optional explicit, byte-identical relocation."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path


def observe_sources(facts, observe, diagnostics, *, relocations=None, facts_sha256=None):
    """Relocation changes custody paths only; facts and their recorded digests stay intact."""
    if relocations is not None and not isinstance(relocations, Mapping):
        raise ValueError("RTL source relocations must be a mapping")
    if relocations and (not isinstance(facts_sha256, str) or len(facts_sha256) != 64):
        raise ValueError("RTL source relocation requires selected raw facts identity")
    selected = {}
    for original, row in (relocations or {}).items():
        if (
            not isinstance(original, str)
            or not Path(original).is_absolute()
            or ".." in Path(original).parts
            or not isinstance(row, Mapping)
            or set(row) != {"path", "sha256"}
            or not isinstance(row["path"], str)
            or not Path(row["path"]).is_absolute()
            or ".." in Path(row["path"]).parts
            or not isinstance(row["sha256"], str)
            or len(row["sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in row["sha256"])
        ):
            raise ValueError("invalid explicit RTL source relocation")
        selected[original] = dict(row)
    inputs, body = facts.get("inputs") or {}, facts.get("facts") or {}
    consistency = facts.get("source_consistency") or {}
    records = [
        inputs,
        body.get("source") or {},
        *(inputs.get("firrtl_inputs") or []),
        *(inputs.get("reader_sources") or []),
        *(consistency.get("sources") or []),
        (consistency.get("production") or {}).get("input_preparation") or {},
    ]
    used = {}
    for record in records:
        if not isinstance(record, Mapping):
            continue
        for key, value in record.items():
            if not ((key.endswith("_path") or key == "path") and isinstance(value, str) and value):
                continue
            original = Path(value)
            if not original.is_absolute():
                continue
            expected = record.get("sha256" if key == "path" else key.removesuffix("_path") + "_sha256")
            relocated = selected.get(value)
            path = Path(relocated["path"]) if relocated else original
            if relocated:
                if expected != relocated["sha256"]:
                    raise ValueError("RTL source relocation is not bound to the recorded source digest")
                if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
                    raise ValueError("RTL source relocation must select a regular non-indirect file")
            observed = observe(path, f"rtl-source:{key}", **({"required": True} if relocated else {}))
            actual = hashlib.sha256(observed).hexdigest() if observed is not None else None
            if relocated:
                if actual != expected:
                    raise ValueError("RTL source relocation bytes differ from the recorded digest")
                used[value] = {"original_path": value, "replacement_path": str(path), "sha256": actual}
            elif observed is not None and isinstance(expected, str) and len(expected) == 64 and expected != actual:
                diagnostics.append(
                    {
                        "component": "rtl-source",
                        "status": "contradiction",
                        "reason": f"recorded source digest differs from observed bytes: {path}",
                    }
                )
    if set(used) != set(selected):
        raise ValueError("RTL source relocation has no matching digest-bound facts source")
    if not used:
        return None
    return {
        "schema": "merlin.phase0.rtl_source_relocation.v1",
        "facts_sha256": facts_sha256,
        "sources": [used[key] for key in sorted(used)],
        "scope": "byte-identical recorded source custody only; no new extraction or hardware qualification",
    }
