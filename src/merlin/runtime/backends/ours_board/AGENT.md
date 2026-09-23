# AGENT.md — merlin/python/merlin/runtime/backends/ours_board

## Purpose

BOARD (RVV) OURS GEMM kernel backend: route the f32 ``linalg.matmul`` dispatches of a whole-model lowering to OUR OWN compiler-emitted MR=4 accumulator-resident RVV micro-kernel (the "v3" kernel) —…

<!-- Purpose/Modules derived from docstrings via build_tools/scripts/gen_package_docs.py.
     Add hand-written notes (invariants, gotchas) below. -->
