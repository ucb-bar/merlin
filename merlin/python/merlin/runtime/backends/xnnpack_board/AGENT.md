# AGENT.md — merlin/python/merlin/runtime/backends/xnnpack_board

## Purpose

BOARD (RVV) XNNPACK kernel backend: route the f32 ``linalg.matmul`` dispatches of a whole-model lowering to XNNPACK's RVV GEMM microkernel, the K1/RVV analogue of the host ``xnnpack_host`` backend…

<!-- Purpose/Modules derived from docstrings via build_tools/scripts/gen_package_docs.py.
     Add hand-written notes (invariants, gotchas) below. -->
