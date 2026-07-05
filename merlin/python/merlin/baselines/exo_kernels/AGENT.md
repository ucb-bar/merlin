# AGENT.md — merlin/python/merlin/baselines/exo_kernels

## Purpose

EXO kernel sources for the K1-RVV whole-model baseline arm.

## Modules

- `gemm.py` — EXO RVV GEMM kernel for the K1 whole-model glue runtime.
- `rvv256.py` — 8-wide (VLEN=256) f32 RVV register class + intrinsics for the SpacemiT K1 X60.

<!-- Purpose/Modules derived from docstrings via build_tools/scripts/gen_package_docs.py.
     Add hand-written notes (invariants, gotchas) below. -->
