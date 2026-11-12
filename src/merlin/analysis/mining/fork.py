"""Versioned fork minting + lineage — forks ACCUMULATE under artifacts/targets/<target>/, never
replace a parent. Each tuning step mints a new package dir named
``<target>_tuned_v{version}_d{depth}_{timestamp}`` with a manifest recording its parent and the
evidence that justified it, so the whole beam-search tree is inspectable and reproducible.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from merlin.common.yaml import write_yaml
from merlin.compiler.backend.host import HostPackage as RvvPackage
from merlin.compiler.backend.host import load_host_package as load_rvv_package


def mint_run_id(target: str, version: int, depth: int, timestamp: str) -> str:
    """`<target>_tuned_v{version}_d{depth}_{timestamp}` — v=generation, d=beam depth."""
    return f"{target}_tuned_v{version}_d{depth}_{timestamp}"


def write_fork(
    out_root: str | Path,
    target: str,
    run_id: str,
    *,
    schedule_text: str,
    knobs: dict[str, Any],
    lineage: dict[str, Any],
    status: str = "proposed",
    renderer_sources: dict | None = None,
) -> Path:
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
        "status": status,  # proposed -> spike_verified -> cycle_confirmed
        "authoring": {
            "mode": "deterministic_generated_from_spec",
            "generated_by_agent": bool(lineage.get("generated_by_agent", False)),
            "author": lineage.get("author", "mining.from_strategy"),
        },
        "lineage": lineage,
        "outputs": {"schedule": "schedule.mlir", "knobs": "knobs.yaml"},
    }
    if renderer_sources is not None:
        manifest["authoring"]["renderer_sources"] = renderer_sources
    write_yaml(d / "manifest.yaml", manifest, header="Isolated RVV fork package (mining.fork)")
    return d


def mint_fork(
    parent: RvvPackage | str | Path,
    overrides: dict[str, Any],
    *,
    version: int,
    depth: int,
    timestamp: str,
    source_evidence: list[str],
    lever: str,
    target: str = "rvv",
    out_root: str | Path = "out/artifacts/targets",
    generated_by_agent: bool = False,
    codegen_provider=None,
    support_target: str | None = None,
) -> Path:
    """Mint a versioned fork by rendering merged knobs and preserving the parent lineage."""
    from merlin.analysis.mining.from_strategy import _codegen, policy_source_record, render_schedule

    if not isinstance(parent, RvvPackage):
        parent = load_rvv_package(parent)
    if codegen_provider is None and support_target is None:
        codegen_provider = parent.codegen_provider
    provider = _codegen(codegen_provider, support_target=support_target)
    sources = policy_source_record(provider)
    knobs = copy.deepcopy(parent.knobs)
    knobs.update(overrides)
    schedule_text = render_schedule(knobs, codegen_provider=provider)
    if policy_source_record(provider) != sources:
        raise ValueError("selected vector policy source changed during schedule rendering")
    run_id = mint_run_id(target, version, depth, timestamp)
    lineage = {
        "parent_run_id": parent.run_id,
        "version": version,
        "depth": depth,
        "source_evidence": source_evidence,
        "lever": lever,
        "generated_by_agent": generated_by_agent,
    }
    return write_fork(
        out_root, target, run_id, schedule_text=schedule_text, knobs=knobs, lineage=lineage, renderer_sources=sources
    )
