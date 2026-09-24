"""Generated Phase 0 member selections for the two different downstream jobs.

The selection digest identifies names and purpose. The containing corpus/run seal
binds the actual bytes; a selection digest alone is never a capsule byte seal.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

_PURPOSES = {
    "phase1": "functional_conformance",
    "phase2": "performance_optimization",
    "diagnostic": "phase0_diagnostic",
}


def _digest(document: dict[str, Any]) -> str:
    payload = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _member_path(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("phase corpus member must be a category/capsule path")
    path = Path(value)
    if (
        path.is_absolute()
        or len(path.parts) != 2
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("phase corpus member must be a safe category/capsule path")
    return value


def _selection(role: str, members: list[str]) -> dict[str, Any]:
    record: dict[str, Any] = {"purpose": _PURPOSES[role], "generated_members": sorted(members)}
    record["selection_sha256"] = _digest(record)
    return record


def generate_phase_selections(written: list[str], *, performance_category: str) -> dict[str, Any]:
    """Classify emitted public capsules by role, without inferring role from target name."""
    if not performance_category.startswith("_") or performance_category in {"_", "__"}:
        raise ValueError("performance category must be a private-discovery category")
    buckets: dict[str, list[str]] = {role: [] for role in _PURPOSES}
    for value in written:
        member = _member_path(value)
        category = member.split("/", 1)[0]
        if category == "hidden":
            continue
        role = "phase2" if category == performance_category else "diagnostic" if category.startswith("_") else "phase1"
        buckets[role].append(member)
    if any(len(set(members)) != len(members) for members in buckets.values()):
        raise ValueError("phase corpus selections contain duplicate members")
    return {"schema_version": 1, **{role: _selection(role, members) for role, members in buckets.items()}}


def validate_phase_selections(
    selection: object,
    *,
    performance_category: str,
    generated_members: set[str] | None = None,
    complete: bool = False,
) -> dict[str, tuple[str, ...]]:
    """Verify disjoint roles and exact declared generation when the full output is available."""
    if not isinstance(selection, dict) or selection.get("schema_version") != 1:
        raise ValueError("phase corpus selections lack the current schema")
    buckets: dict[str, tuple[str, ...]] = {}
    for role, purpose in _PURPOSES.items():
        row = selection.get(role)
        if not isinstance(row, dict) or row.get("purpose") != purpose:
            raise ValueError(f"phase corpus {role} has the wrong purpose")
        values = row.get("generated_members")
        if not isinstance(values, list):
            raise ValueError(f"phase corpus {role} lacks generated members")
        members = tuple(_member_path(value) for value in values)
        if list(members) != sorted(set(members)) or row.get("selection_sha256") != _digest(
            {"purpose": purpose, "generated_members": list(members)}
        ):
            raise ValueError(f"phase corpus {role} selection identity changed")
        buckets[role] = members
    joined = [member for members in buckets.values() for member in members]
    if len(joined) != len(set(joined)):
        raise ValueError("phase corpus roles overlap")
    for role, members in buckets.items():
        for member in members:
            category = member.split("/", 1)[0]
            expected = (
                "phase2" if category == performance_category else "diagnostic" if category.startswith("_") else "phase1"
            )
            if role != expected or category == "hidden":
                raise ValueError(f"phase corpus member has the wrong role: {member}")
    if generated_members is not None:
        declared = {_member_path(value) for value in generated_members if not str(value).startswith("hidden/")}
        actual = set(joined)
        if not actual <= declared or (complete and actual != declared):
            raise ValueError("phase corpus selections disagree with generated provenance")
    return buckets
