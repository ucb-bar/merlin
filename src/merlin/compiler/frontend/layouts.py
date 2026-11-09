"""Bind an explicit physical boundary representation to an admitted source.

This checks source/request correspondence. Layout legality and encodings remain
obligations of the selected backend and approved hardware contract.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path


def bind_physical_layouts(args, request, precision_plan):
    """Restore the versioned native layout document without guessing a layout."""
    path = getattr(args, "physical_layouts", None)
    if path is None:
        return request, None
    if getattr(args, "request", None) or any(
        getattr(args, name, None) is not None
        for name in ("ordered_k_tile", "ordered_m_tile", "ordered_n_tile", "outer_tile_index")
    ):
        raise ValueError("physical layouts require parsed Linalg without ordered tile transformations")
    row = json.loads(Path(path).read_bytes())
    fields = {"schema", "target_identity", "source_identity", "physical_extents",
              "boundary_layouts", "output_layouts"}
    schema = row.get("schema") if isinstance(row, dict) else None
    if schema not in {"merlin.physical_boundary_layouts.v1", "merlin.physical_boundary_layouts.v2"} or (
        set(row) != (fields | {"request_digest"} if schema.endswith("v2") else fields)
    ):
        raise ValueError("physical layouts need the complete versioned boundary schema")
    if precision_plan is not None and schema.endswith("v1"):
        raise ValueError("precision-transformed layouts need the exact-request v2 schema")
    if schema.endswith("v2") and row["request_digest"] != request.digest():
        raise ValueError("physical layouts differ from the precision-transformed request")
    if row["target_identity"] != request.target_identity or row["source_identity"] != request.source_identity:
        raise ValueError("physical layouts differ from the parsed source or selected target")
    if (not isinstance(row["physical_extents"], dict) or not row["physical_extents"] or
            not isinstance(row["boundary_layouts"], dict) or not row["boundary_layouts"] or
            not isinstance(row["output_layouts"], list) or not row["output_layouts"]):
        raise ValueError("physical layouts need typed extents, input layouts, and ordered output layouts")
    bound = replace(request, physical_extents=tuple(sorted(row["physical_extents"].items())),
                    boundary_layouts=tuple(sorted(row["boundary_layouts"].items())),
                    output_layouts=tuple(row["output_layouts"]))
    canonical = {**row, "physical_extents": dict(bound.physical_extents),
                 "boundary_layouts": dict(bound.boundary_layouts), "output_layouts": list(bound.output_layouts)}
    return bound, (json.dumps(canonical, sort_keys=True, indent=2) + "\n").encode()


def check_layout_artifact(artifact, manifest, document):
    """Require the selected document, or refuse an unrequested representation."""
    name = "physical-boundary-layouts.json"
    path = artifact / name
    tracked = manifest.get("stage_artifacts")
    if document is None:
        if path.exists() or path.is_symlink() or isinstance(tracked, dict) and name in tracked:
            raise ValueError("native artifact has an unrequested physical boundary layout")
    elif (path.is_symlink() or not path.is_file() or path.read_bytes() != document or
          not isinstance(tracked, dict) or tracked.get(name) != hashlib.sha256(document).hexdigest()):
        raise ValueError("native physical boundary layout differs from the selected source representation")
