"""Load a ``workload_region`` for DSE guidance.

Mirrors the resolution order of ``merlin.design_pressure.cli._load_region`` (explicit YAML
path, the synthetic ``vla_action_chunk_decode`` builder, or a ``semantic_memory`` benchmark
name) so the guidance CLI accepts the same workload identifiers as ``merlin-design-pressure``.
Region interpretation (reuse, epilogue, M/K/N) is reused from
:mod:`merlin.design_pressure.region` — this module only resolves *which* region to load.
"""
from __future__ import annotations

from pathlib import Path

from merlin.common import paths
from merlin.common.yaml import load_yaml


def load_region(workload: str | None = None, region_yaml: str | None = None,
                H: int = 16) -> dict:
    """Resolve a workload region dict.

    Precedence: ``region_yaml`` path > synthetic ``vla_action_chunk_decode`` > benchmark name
    under ``merlin/benchmarks/semantic_memory/<workload>.yaml``.
    """
    if region_yaml:
        return load_yaml(region_yaml)
    if workload == "vla_action_chunk_decode":
        from merlin.design_pressure.workloads.vla_action_chunk_decode import build_region
        return build_region(H=H, reuse_count=H)
    if workload:
        bench = paths.merlin_dir() / "benchmarks" / "semantic_memory" / f"{workload}.yaml"
        if bench.is_file():
            return load_yaml(bench)
    raise SystemExit(
        f"unknown workload '{workload}' (pass --region-yaml or a semantic_memory benchmark name)"
    )
