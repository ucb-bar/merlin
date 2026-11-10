# src/merlin/kernels/ceiling_drivers

## Purpose

Shared native driver resources consumed by runtime backends and ceiling measurements.
Core owns the C/header bytes and namespace initializer. Optional Python measurement controllers
live in `src/merlin/kernels/ceiling_drivers/` under unchanged imports.

## Modules

- `merlin.kernels.ceiling_drivers.multishape_compare` — analysis-owned multi-shape comparison.
- `merlin.kernels.ceiling_drivers.run_expert_gemm` — analysis-owned single-shape comparison.

Runtime consumers must resolve driver resources without requiring analysis or mining. Measurement
controllers resolve this core package rather than assume resources sit beside their Python files.

<!-- Purpose/Modules derived from docstrings via build_tools/scripts/gen_package_docs.py.
     Add hand-written notes (invariants, gotchas) below. -->

## Research workflow responsibilities

Owns `run_expert_gemm` and `multishape_compare`, retaining their historical imports.
Driver C/header resources retain their compiler owner and resolve through the installed package.
Select `merlin[analysis,mining]` for generated workload comparisons. Keep source identities,
measurement scopes, failures and numeric checks unchanged. Importing a controller must not run
a compiler, simulator or board job; validation here uses fixture/subprocess stubs.
