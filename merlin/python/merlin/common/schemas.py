"""Load and validate artifacts against the YAML schemas in ``merlin/schemas/``.

The schemas are intentionally lightweight (see ``merlin/schemas/AGENT.md``): each is a
YAML doc with a ``title``, ``purpose``, a ``required_top_level_fields`` list, and an
``example`` block. They are *not* formal JSON Schema yet. Validation here therefore means:
"the object is a mapping that contains every required top-level field." That is enough to
keep cross-workstream artifacts honest without over-engineering a type system.

This module is the single dependency every kernel-mining emitter relies on, so it stays
dependency-light (stdlib + PyYAML) and side-effect free.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def schemas_dir() -> Path:
    """Return the directory holding ``*.schema.yaml`` files.

    Honors ``MERLIN_SCHEMAS_DIR`` for installed/relocated layouts; otherwise resolves
    ``<repo>/merlin/schemas`` relative to this source file.
    """
    env = os.environ.get("MERLIN_SCHEMAS_DIR")
    if env:
        return Path(env)
    # this file: <repo>/merlin/python/merlin/common/schemas.py
    #   parents[3] == <repo>/merlin
    return Path(__file__).resolve().parents[3] / "schemas"


@lru_cache(maxsize=None)
def load_schema(name: str) -> dict[str, Any]:
    """Load a schema by short name (e.g. ``"kernel_record"``).

    Accepts either the short name or a full ``<name>.schema.yaml`` filename.
    """
    stem = name[: -len(".schema.yaml")] if name.endswith(".schema.yaml") else name
    path = schemas_dir() / f"{stem}.schema.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"schema not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"schema {path} did not parse to a mapping")
    return data


def required_fields(name: str) -> list[str]:
    """Return the ``required_top_level_fields`` list for a schema."""
    return list(load_schema(name).get("required_top_level_fields", []))


def validate(obj: Any, schema_name: str) -> list[str]:
    """Validate ``obj`` against a schema. Return a list of problems (empty == valid).

    Checks that ``obj`` is a mapping and contains every required top-level field with a
    non-None value. Does not (yet) type-check nested fields.
    """
    problems: list[str] = []
    if not isinstance(obj, dict):
        return [f"expected a mapping for schema '{schema_name}', got {type(obj).__name__}"]
    for field in required_fields(schema_name):
        if field not in obj:
            problems.append(f"missing required field '{field}'")
        elif obj[field] is None:
            problems.append(f"required field '{field}' is null")
    return problems


def validate_or_raise(obj: Any, schema_name: str) -> None:
    """Validate and raise ``ValueError`` listing all problems if invalid."""
    problems = validate(obj, schema_name)
    if problems:
        joined = "; ".join(problems)
        raise ValueError(f"{schema_name} validation failed: {joined}")
