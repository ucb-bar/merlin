# AGENT.md — src/merlin/kernels/ceiling_drivers

## Purpose

Shared native driver resources consumed by runtime backends and ceiling measurements.
Core owns the C/header bytes and namespace initializer. Optional Python measurement controllers
live in `packages/merlin-analysis/src/merlin/kernels/ceiling_drivers/` under unchanged imports.

## Modules

- `merlin.kernels.ceiling_drivers.multishape_compare` — analysis-owned multi-shape comparison.
- `merlin.kernels.ceiling_drivers.run_expert_gemm` — analysis-owned single-shape comparison.

Runtime consumers must resolve driver resources without requiring analysis or mining. Measurement
controllers resolve this core package rather than assume resources sit beside their Python files.

<!-- Purpose/Modules derived from docstrings via build_tools/scripts/gen_package_docs.py.
     Add hand-written notes (invariants, gotchas) below. -->
