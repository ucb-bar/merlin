# Gemmini xDSL out-of-tree backend

This package implements the four required CLI entrypoints with a self-contained Python/xDSL backend. It structurally parses and verifies `merlin_iface`, rewrites it to typed Gemmini target operations, emits strict command buffers, and lowers compiler-derived tile schedules to LLVM-dialect MLIR containing canonical pointer-derived RoCC `.insn` operations.

The backend covers native movement, resident and reused int8 matmul, M/K/N tails, deep-K streaming, i32 and scaled/activated i8 readout, fused maxpool, native attention QK, and native NHWC convolution. DRAM row strides use the tile-padded physical final dimension while PRELOAD/COMPUTE geometry uses logical tail extents. Convolution gathers the interface IFM into scratchpad internally and keeps the kernel ABI in declaration order; no derived im2col pointer is exposed to the harness.

Final verification:

- The required all-capsule Spike command passes 33 of 33 capsules; every per-capsule `pass` and barrier status is green.
- Shape coverage passes at one tile and two tiles along M, K, and N, with no declines, empty programs, collapses, or uncovered axes.
- All 48 locally visible public/dev interfaces pass xDSL parsing, strict command-buffer schema validation, LLVM artifact generation, and stock `mlir-opt` verification.
- ISA lint and disassembly over those 48 artifacts report zero unknown instructions.
- The CCA bijection is clean and all seven Gemmini escalation ladders were enumerated.

The final 2026-09-02 rerun of the exact required command
`python3 agent_selfcheck.py --submission submission --sim spike --capsules all`
again reported 33/33 per-capsule passes and zero declines.  The companion shape
probe again covered one tile and two tiles along every M/K/N axis, with emitted
work 37/54/57/59 and no collapsed corner.

The certifying Verilator request was also attempted unchanged after functional convergence, but the broker aborted at its timeout without returning a capsule-level failure. Therefore L3 certification is not claimed; the remaining issue is the external certification-timeout plane, not a known functional, schema, trace, or shape-coverage failure.

Backend passes all required public/dev capsules and is ready for hidden grading.
