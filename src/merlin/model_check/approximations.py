"""A submission's declared approximate kernels (``manifest.yaml`` key ``numerics.approximations``).

Every kernel whose interface opcode is listed is judged by the ratio rule; every other kernel must be
bit-exact (contract §5). Unknown keys are refused: a misspelled declaration must not silently turn a
kernel back into an exact one, or the reverse.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_KEYS = {"id", "opcodes", "description", "reason"}


class ApproximationError(ValueError):
    """The ``numerics`` block of a submission manifest is malformed."""


def declared(submission: Path) -> list[dict[str, Any]]:
    """The submission's approximation declarations, validated; ``[]`` when it declares none."""
    manifest = Path(submission) / "manifest.yaml"
    if not manifest.is_file():
        raise ApproximationError(f"{manifest}: no manifest")
    data = yaml.safe_load(manifest.read_text()) or {}
    numerics = data.get("numerics")
    if numerics is None:
        return []
    if not isinstance(numerics, dict) or set(numerics) - {"approximations"}:
        raise ApproximationError("numerics may only declare 'approximations'")
    rows = numerics.get("approximations") or []
    if not isinstance(rows, list):
        raise ApproximationError("numerics.approximations must be a list")
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != _KEYS:
            raise ApproximationError(f"each approximation declares exactly {sorted(_KEYS)}")
        ident = str(row["id"])
        if not ident or any(not (ch.islower() or ch.isdigit() or ch == "_") for ch in ident) or ident in seen:
            raise ApproximationError(f"approximation id {ident!r} must be unique and [a-z0-9_]+")
        seen.add(ident)
        if (
            not isinstance(row["opcodes"], list)
            or not row["opcodes"]
            or not all(isinstance(o, str) for o in row["opcodes"])
        ):
            raise ApproximationError(f"approximation {ident!r}: opcodes must be a non-empty list of names")
    return rows
