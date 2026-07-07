# AGENT.md — merlin/python/merlin/kernels/ceiling_drivers

## Purpose

Expert-kernel ceiling drivers: measure the performance bar our RVV codegen is ranked against.

## Modules

- `multishape_compare.py` — Cross-framework GEMM ceiling matrix on ONE substrate (spike), multiple shapes.
- `run_expert_gemm.py` — Measure the EXPERT RVV GEMM ceiling (XNNPACK + OpenBLAS) on spike, standalone.

<!-- Purpose/Modules derived from docstrings via build_tools/scripts/gen_package_docs.py.
     Add hand-written notes (invariants, gotchas) below. -->
