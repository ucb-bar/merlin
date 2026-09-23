# AGENT.md — merlin/python/merlin/runtime/backends/openblas_board

## Purpose

BOARD (RVV) OpenBLAS kernel backend: route the f32 ``linalg.matmul`` dispatches of a whole-model lowering to OpenBLAS's RVV 8x8 GEMM microkernel — the OpenBLAS analogue of the…

<!-- Purpose/Modules derived from docstrings via build_tools/scripts/gen_package_docs.py.
     Add hand-written notes (invariants, gotchas) below. -->
