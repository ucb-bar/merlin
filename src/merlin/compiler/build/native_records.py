"""Publish selected native preparation records inside unpublished staging.

These records bind source transformations and declared parameter bytes to the
artifact. They do not establish numerical quality or device execution.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from merlin.compiler.quantization.native import precision_plan_bytes, prepared_static_inputs


def _destination(staged, relative):
    """Keep preparation artifacts inside the actual unpublished directory."""
    staged = Path(staged)
    relative = Path(relative)
    path = staged / relative
    parents = (staged.joinpath(*relative.parts[:index]) for index in range(len(relative.parts) + 1))
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or any(parent.is_symlink() for parent in parents)
        or path.exists()
    ):
        raise ValueError("target preparation destination already exists or is linked outside staging")
    return path


def _write_manifest(staged, manifest):
    (staged / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")


def publish_precision_plan(staged, manifest, precision_plan, *, binding=None, staged_root=None):
    """Track a graph plan or source-bound tile and its declared packed inputs."""
    from merlin.compiler.quantization.contraction.contract import DiagnosticMatmulTile

    raw_plan = precision_plan_bytes(precision_plan)
    tile = isinstance(precision_plan, DiagnosticMatmulTile)
    tracked = manifest.get("stage_artifacts")
    if not isinstance(tracked, dict) or "precision-plan.json" in tracked:
        raise ValueError("target manifest cannot track the source precision plan")
    if tile and "constant_inputs" in manifest:
        raise ValueError("target manifest cannot track the source precision plan")
    if not tile and "constant_inputs" not in manifest:
        raise ValueError("target graph manifest has no explicit constant input record")
    plan_path = staged / "precision-plan.json"
    if plan_path.exists() or plan_path.is_symlink():
        raise ValueError("target precision plan destination already exists")
    if tile and (binding is None or staged_root is None):
        raise ValueError("source-bound constants require their selected support and staged compiler inputs")
    static = (
        prepared_static_inputs(
            binding,
            precision_plan,
            Path(staged_root),
            json.loads((staged / "execution_plan.json").read_bytes()),
        )
        if tile
        else ()
    )
    destinations = []
    for item in static:
        if item.payload is None:
            continue
        relative = item.record()["path"]
        if relative in tracked:
            raise ValueError("target manifest already tracks a source constant")
        destinations.append((_destination(staged, relative), item.payload, relative))
    # Validate every selected destination before writing any parameter payload.
    for path, payload, _ in destinations:
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(payload)
    plan_path.write_bytes(raw_plan)
    manifest = {
        **manifest,
        "stage_artifacts": {
            **tracked,
            "precision-plan.json": hashlib.sha256(raw_plan).hexdigest(),
            **{relative: hashlib.sha256(payload).hexdigest() for _, payload, relative in destinations},
        },
    }
    if tile:
        manifest["constant_inputs"] = [item.record() for item in static]
    _write_manifest(staged, manifest)
    return manifest


def publish_ordered_records(
    staged, manifest, *, source_request, k_source_request, ordered_mn_record=None, ordered_k_record=None
):
    """Retain the original outer-M/N then ordered-K request digest chain."""
    rows = (
        ("ordered-k-tiling.json", ordered_k_record, "source_request_digest", k_source_request, "ordered-K"),
        ("ordered-mn-tiling.json", ordered_mn_record, "outer_source_request_digest", source_request, "ordered-M/N"),
    )
    for name, record, field, source, description in rows:
        if record is None:
            continue
        tracked = manifest.get("stage_artifacts")
        if not isinstance(tracked, dict) or name in tracked:
            raise ValueError(f"target manifest cannot track the {description} partition")
        path = _destination(staged, name)
        path.write_bytes(record)
        manifest = {
            **manifest,
            field: source.digest(),
            "stage_artifacts": {
                **tracked,
                name: hashlib.sha256(record).hexdigest(),
            },
        }
        _write_manifest(staged, manifest)
    return manifest
