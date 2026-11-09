"""Replay source preparation records and declared parameter payloads.

Successful checks establish source/artifact identity only. Numerical semantics,
application quality, actual execution and target qualification remain separate.
"""

from __future__ import annotations

import json
from pathlib import Path

from merlin.compiler.quantization.native import precision_plan_bytes, prepared_static_inputs


def check_ordered_records(
    artifact, manifest, *, source_request, k_source_request, ordered_mn_record=None, ordered_k_record=None
):
    """Require exact requested records and refuse undeclared decompositions."""
    rows = (
        ("ordered-k-tiling.json", ordered_k_record, "source_request_digest", k_source_request, "ordered-K"),
        ("ordered-mn-tiling.json", ordered_mn_record, "outer_source_request_digest", source_request, "ordered-M/N"),
    )
    for name, record, field, source, description in rows:
        path = artifact / name
        if record is None:
            if path.exists() or path.is_symlink() or field in manifest:
                raise ValueError(f"native artifact has an unrequested {description} partition")
        elif (
            manifest.get(field) != source.digest()
            or path.is_symlink()
            or not path.is_file()
            or path.read_bytes() != record
        ):
            raise ValueError(f"{description} partition differs from the original source")


def check_precision_plan(artifact, manifest, precision_plan, *, binding=None, staged_root=None):
    """Rebuild selected tile constants from the original staged parameter source."""
    from merlin.semantic_compiler.precision_contract import DiagnosticMatmulTile

    path = artifact / "precision-plan.json"
    if precision_plan is None:
        if path.exists() or path.is_symlink():
            raise ValueError("native artifact has an undeclared source precision plan")
        return
    if path.is_symlink() or not path.is_file() or path.read_bytes() != precision_plan_bytes(precision_plan):
        raise ValueError("source precision plan differs from staged compiler inputs")
    if not isinstance(precision_plan, DiagnosticMatmulTile):
        return
    if binding is None or staged_root is None:
        raise ValueError("source-bound constants require their selected support and staged compiler inputs")
    prepared = prepared_static_inputs(
        binding,
        precision_plan,
        Path(staged_root),
        json.loads((artifact / "execution_plan.json").read_bytes()),
    )
    if manifest.get("constant_inputs") != [item.record() for item in prepared]:
        raise ValueError("source-bound constant inputs differ from staged compiler inputs")
    for item in prepared:
        if item.payload is not None:
            destination = artifact / item.record()["path"]
            if (
                destination.is_symlink()
                or not destination.is_file()
                or destination.parent.is_symlink()
                or destination.read_bytes() != item.payload
            ):
                raise ValueError("source-bound constant payload differs from staged parameter")
