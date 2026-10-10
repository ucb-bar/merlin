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

Optional `contraction_selections` in the ordinary freezing path is a separate
complete-original integer-format requirement. It reopens protected selected
source/trace/MLIR bytes and the complete session program roster at registration
and freezing; saved census metadata cannot replace that check. Default mixed
capture and numerical/precision policy meanings remain unchanged. The first
supported positive domain is narrow rank-two signed-i8/i32 mm; unknown original
calls, geometry, multiplicity and fragments refuse explicit complete-integer
registration. This source-format gate is neither measurement nor offload,
runtime, physical timing or full Phase 0 admission.

Study v3 explicitly declares per-precision `contraction_format_requirements`.
Normal capture dispatch requires a complete independently selected original
program pin roster before launch, then joins emitted trace/MLIR/session bytes to
that roster. The normal CLI and freezer cannot omit the declared requirement;
frozen preflight reopens exact selection bytes. Selection descriptors and saved
records transport data, not producer or eligibility authority. No original graph
is synthesized from post-capture provenance. Independent original-source producer
qualification remains a separate open prerequisite. Version 2 absent selection
retains legitimate mixed captures and numerical/precision policies.
