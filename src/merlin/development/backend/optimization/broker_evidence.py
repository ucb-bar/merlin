"""Shared broker receipt identities; scientific schemas live with their workflow."""

from collections.abc import Mapping, Sequence
from typing import Any


def workflow_binding(rows: Sequence[Mapping[str, Any]], expected: str) -> dict[str, Any]:
    """Check explicit new identities without inventing identities for historical bytes."""
    for row in rows:
        if "workflow_id" in row and row["workflow_id"] != expected:
            raise ValueError("broker receipt workflow policy does not match selected authority")
    bound = bool(rows) and all("workflow_id" in row for row in rows)
    return {"status": "bound" if bound else "historical_unbound", "id": expected if bound else None}


_HEX = frozenset("0123456789abcdef")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value.lower() == value
        and all(character in _HEX for character in value)
    )
