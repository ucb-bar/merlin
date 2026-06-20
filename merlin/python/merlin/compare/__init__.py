"""merlin.compare — unified, spec-driven, versioned comparison driver.

Makes "compare baseline vs ours vs XNNPACK vs OpenBLAS (and across iterations), empirically AND
structurally, with gap-attribution and figures" ONE repeatable command. Five layers, each reusing
an existing tool:
  empirical   (measured table; ingest cached harness JSONs, board-run seam)
  structural  (per-config CCA; reuse kernels.cca + kernels.decode)
  attribution (measured gap × CCA divergence × routed action; reuse cca_compare + action_catalog)
  figures     (paper-styled PNGs; reuse scripts/plot_paper_style palette/helpers)
  report      (compare.md dashboard + deterministic manifest.yaml)
"""
from __future__ import annotations

from .driver import run
from .spec import Config, Spec, Workload

__all__ = ["run", "Spec", "Config", "Workload"]
