# AGENT.md — merlin/python/merlin/compare

## Purpose

merlin.compare — unified, spec-driven, versioned comparison driver.

## Modules

- `attribution.py` — LAYER 3 — ATTRIBUTION: the new glue that automates the manual ``kernel_breakdown.md``.
- `cli.py` — ``merlin-compare`` CLI entry — spec-driven, versioned comparison driver.
- `driver.py` — The merlin-compare DRIVER — one repeatable command stitching the five layers into a VERSIONED
- `empirical.py` — LAYER 1 — EMPIRICAL: the measured table, behind a ``measure(config, workload, target)`` seam.
- `figures.py` — LAYER 4 — FIGURES: paper-styled PNGs driven by the artifact's ingested data.
- `report.py` — LAYER 5 — REPORT + MANIFEST: the dashboard ``compare.md`` and the deterministic ``manifest.yaml``.
- `spec.py` — Target-agnostic comparison SPEC — the single source of truth a ``merlin-compare`` run is driven by.
- `structural.py` — LAYER 2 — STRUCTURAL: per-config CCA (the Common Compute Abstraction of each config's matmul).

<!-- Purpose/Modules derived from docstrings via build_tools/scripts/gen_package_docs.py.
     Add hand-written notes (invariants, gotchas) below. -->
