"""Versioned fork minting + lineage — forks ACCUMULATE under artifacts/targets/<target>/, never
replace a parent. Each tuning step mints a new package dir named
``<target>_tuned_v{version}_d{depth}_{timestamp}`` with a manifest recording its parent and the
evidence that justified it, so the whole beam-search tree is inspectable and reproducible.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..common.yaml import write_yaml


def mint_run_id(target: str, version: int, depth: int, timestamp: str) -> str:
    """`<target>_tuned_v{version}_d{depth}_{timestamp}` — v=generation, d=beam depth."""
    return f"{target}_tuned_v{version}_d{depth}_{timestamp}"


def write_fork(out_root: str | Path, target: str, run_id: str, *, schedule_text: str,
               knobs: dict[str, Any], lineage: dict[str, Any], status: str = "proposed") -> Path:
    """Write a fork package dir (schedule.mlir + knobs.yaml + manifest.yaml with lineage).

    ``lineage`` carries parent_run_id, version, depth, source_evidence, lever. Never overwrites a
    parent: callers pass a fresh ``run_id`` (see :func:`mint_run_id`). Returns the package dir.
    """
    d = Path(out_root) / target / run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "schedule.mlir").write_text(schedule_text, encoding="utf-8")
    write_yaml(d / "knobs.yaml", knobs, header="RVV fork knobs (mining.from_strategy)")
    manifest = {
        "target": target,
        "run_id": run_id,
        "family": "vector_schedule",
        "schedule_format": "transform_dialect_mlir",
        "status": status,                     # proposed -> spike_verified -> cycle_confirmed
        "authoring": {"mode": "deterministic_generated_from_spec",
                      "generated_by_agent": bool(lineage.get("generated_by_agent", False)),
                      "author": lineage.get("author", "mining.from_strategy")},
        "lineage": lineage,
        "outputs": {"schedule": "schedule.mlir", "knobs": "knobs.yaml"},
    }
    write_yaml(d / "manifest.yaml", manifest, header="Isolated RVV fork package (mining.fork)")
    return d
