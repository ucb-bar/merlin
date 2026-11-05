"""The hidden cohort a Phase 0 run emits must not repeat a public program.

A holdout measures transfer only when its concrete point is one the agent has not already compiled.
Two capsules are the same POINT when they would build the same program: the same operation, the same
operand shapes, dtypes and roles, the same attributes and modes. The capsule name and the tensor
labels are not part of it -- renaming an operand yields a byte-identical program. The check runs on
the capsules the run actually wrote, so it holds for whatever the public and hidden derivations
produced, and it reports COUNTS only: a hidden point is never printed.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

SCHEMA = "merlin.phase0.hidden_disjointness.v1"


def capsule_point(capsule: Mapping[str, Any]) -> str:
    """The program a capsule builds, independent of its name and operand labels."""
    inputs = [row for row in capsule.get("inputs") or () if isinstance(row, Mapping)]
    labels = {str(row.get("name")) for row in inputs}
    operation = capsule.get("operation") or {}
    from merlin.targetgen.legacy_labels import operation_attributes

    attributes = operation_attributes(operation.get("attributes") or {})
    labels |= {str(value) for key, value in attributes.items() if key == "out"}
    stated = {
        key: value
        for key, value in attributes.items()
        if not (isinstance(value, str) and value in labels) and key != "name"
    }
    point = {
        "op": operation.get("op"),
        "inputs": sorted((str(row.get("role")), list(row.get("shape") or ()), str(row.get("dtype"))) for row in inputs),
        "attributes": stated,
        "modes": (capsule.get("expected") or {}).get("modes") or {},
        "numeric_policy": capsule.get("numeric_policy") or {},
    }
    return json.dumps(point, sort_keys=True, default=str)


def _points(directories: Iterable[Path]) -> dict[str, str]:
    out = {}
    for directory in directories:
        path = Path(directory) / "capsule.yaml"
        if path.is_file():
            out[str(directory)] = capsule_point(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    return out


def check(written: Iterable[Path], *, hidden_category: str = "hidden") -> dict[str, Any]:
    """Counts of hidden and public points and of hidden points that repeat a public one."""
    written = [Path(path) for path in written]
    hidden = _points(path for path in written if path.parent.name == hidden_category)
    public = _points(path for path in written if path.parent.name != hidden_category)
    shared = set(hidden.values()) & set(public.values())
    return {
        "schema": SCHEMA,
        "status": "disjoint" if not shared else "overlap",
        "hidden_capsules": len(hidden),
        "public_capsules": len(public),
        "overlapping_hidden_capsules": sum(point in shared for point in hidden.values()),
        "scope": "program points of the capsules this run wrote; names and operand labels are ignored",
    }
