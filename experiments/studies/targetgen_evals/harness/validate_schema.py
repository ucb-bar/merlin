"""Validate run artifacts against JSON Schema definitions."""

from __future__ import annotations

import json
from pathlib import Path


def run(run_dir: Path, manifest: dict, schemas_dir: Path) -> dict:
    metrics: dict = {
        "validator": "schema",
        "schema_valid": False,
        "files_checked": 0,
        "errors": [],
    }

    try:
        import jsonschema
    except ImportError:
        metrics["errors"].append("jsonschema not installed; schema validation skipped")
        metrics["schema_valid"] = None  # NA
        return metrics

    manifest_schema_path = schemas_dir / "run_manifest.schema.json"
    if not manifest_schema_path.exists():
        metrics["errors"].append("run_manifest.schema.json not found")
        return metrics

    with open(manifest_schema_path) as f:
        schema = json.load(f)

    try:
        jsonschema.validate(manifest, schema)
        metrics["schema_valid"] = True
        metrics["files_checked"] = 1
    except jsonschema.ValidationError as e:
        metrics["schema_valid"] = False
        metrics["errors"].append(f"run_manifest.yaml: {e.message}")

    return metrics
