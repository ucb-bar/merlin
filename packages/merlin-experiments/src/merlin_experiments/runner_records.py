"""Atomic process-plan records and structural JSON validation."""

from __future__ import annotations

import json
from pathlib import Path

from .spec import SpecError


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> dict:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SpecError(f"invalid run record {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise SpecError(f"invalid run record {path}: expected an object")
    return document
