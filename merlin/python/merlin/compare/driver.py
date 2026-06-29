"""The merlin-compare DRIVER — one repeatable command stitching the five layers into a VERSIONED
``mined_knowledge/rvv/compare_<ts>/`` artifact.

Orchestration (all five layers REUSE existing tools; this module is glue only):
  1. EMPIRICAL  — empirical.measure_all (ingest cached harness JSONs)            -> measured table
  2. STRUCTURAL — structural.cca_table / cca_for (lift cached decode -> cca.CCA) -> per-config CCA
  3. ATTRIBUTION— attribution.attribute (cca_compare + action_catalog)           -> divergences+actions
  4. FIGURES    — figures.render (reuse plot_paper_style palette/helpers)        -> PNGs
  5. REPORT     — report.write_report + write_manifest                           -> compare.md + manifest.yaml
"""
from __future__ import annotations

import time
from pathlib import Path

from . import attribution, empirical, figures, report, structural
from .spec import Spec


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def run(spec: Spec, *, out_root: Path | None = None, run_board: bool = False,
        root: Path | None = None, ts: str | None = None) -> Path:
    """Execute a full compare run; return the artifact directory."""
    root = root or _repo_root()
    out_root = Path(out_root) if out_root else (root / "artifacts" / "compare")
    ts = ts or time.strftime("%Y%m%d_%H%M%S")
    out_dir = out_root / f"compare_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. EMPIRICAL
    measurements = empirical.measure_all(spec, run=run_board, root=root)

    # 2. STRUCTURAL — representative-per-config + per-(config, model-workload)
    ccas = structural.cca_table(spec, root=root)
    workload_ccas = {}
    for cfg in spec.configs:
        for wl in spec.workloads:
            if wl.kind == "model":
                workload_ccas[(cfg.name, wl.name)] = structural.cca_for(cfg, wl, root=root)

    # 3. ATTRIBUTION
    attrs = attribution.attribute(spec, measurements, ccas, workload_ccas=workload_ccas)
    gap_axes = attribution.gap_driver_axes(attrs)

    # 4. FIGURES
    figs = figures.render(spec, measurements, ccas, out_dir)

    # 5. REPORT + MANIFEST
    report.write_report(out_dir, spec=spec, measurements=measurements, ccas=ccas,
                        attrs=attrs, figures=figs, root=root, gap_axes=gap_axes)
    report.write_manifest(out_dir, spec=spec, measurements=measurements, ccas=ccas,
                         figures=figs, root=root)
    return out_dir
