"""Read-only conformance coverage for one verified Phase 0 source run."""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from merlin.targetgen import conformance

from ..spec import SpecError
from .preparation import source_run


def inspect_run(run_dir: Path, spec_path: Path) -> dict:
    """Measure public source-pool coverage without grading or admitting a cohort.

    The completed run and its frozen inputs are verified before reading capsules.
    The conformance spec is an explicit, separately hashed diagnostic input; this
    command neither changes that reference nor turns a source-pool match into a
    numerical or hardware verdict.
    """
    source = run_dir.expanduser().resolve(strict=True)
    plan, _, corpus = source_run(source)
    selected = spec_path.expanduser().resolve(strict=True)
    if not selected.is_file():
        raise SpecError("conformance spec must be a regular file")
    raw = selected.read_bytes()
    try:
        spec = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise SpecError(f"invalid conformance YAML: {selected}") from exc
    if not isinstance(spec, dict) or not isinstance(spec.get("cells"), list):
        raise SpecError("conformance spec must contain a cells list")
    target = plan["target"]
    if spec.get("target") != target:
        raise SpecError(f"conformance spec target {spec.get('target')!r} differs from run target {target!r}")
    edge = (spec.get("boundaries") or {}).get("tile_edge")
    if edge is not None and (type(edge) is not int or edge < 1):
        raise SpecError("conformance spec tile_edge must be a positive integer or absent")
    result = conformance.uncovered(spec, [corpus], labels={"public"}, tile_dim=edge)
    return {
        "schema_version": 1,
        "target": target,
        "phase0_run": str(source),
        "corpus": str(corpus),
        "spec": {"path": str(selected), "sha256": hashlib.sha256(raw).hexdigest()},
        "scope": "generated public source pool; not admitted, graded, or certified",
        "coverage": result,
    }
