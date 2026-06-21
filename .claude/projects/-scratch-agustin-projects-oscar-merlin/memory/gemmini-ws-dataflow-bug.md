---
name: gemmini-ws-dataflow-bug
description: capsule_bench_v0 — matmul spike failures caused by OS instruction pattern under WS config
metadata:
  type: project
---

In the gemmini_capsule_bench_v0 merlin_assisted backend (`_qa_ws/merlin_abc1/workspace/submission`),
16/20 capsules failed L2 spike with "spike oracle != golden==reference==simulate". Root cause: the
RoCC matmul lowering used the **output-stationary** instruction pattern (`preload(GARBAGE, C)` +
`compute_preloaded(A, B)`, K-accumulate spatially, write C only on last k) while CONFIG_EX declared
**WEIGHT_STATIONARY**. reference/simulate only see the abstract command buffer so they pass; spike
runs the real RoCC and diverges.

Fix: mirror `sp_tiled_matmul_ws` (gemmini.h): PRELOAD loads weights B (`pre_sp=B_sp`), COMPUTE feeds
A with GARBAGE second operand; K-accumulation via accumulator bit30 (overwrite k==0 by clearing
bit30, accumulate k>0). Also: emit CONFIG_EX **once** per program (A6 resident_reuse trace requires a
single weight-stationary config) and mvin each resident weight once. Inputs+outputs are zero-padded
to DIM=16 multiples (kernel_abi pointee_layout) → use padded strides (A=Kp, B=Np, C=Np).

A1 movement (VECTOR_MAP identity) fails at L0 (golden != reference cb) — VECTOR_MAP semantics not
documented in command_buffer_abi.yaml. B3/B4 conv need 2D im2col matmul in the command buffer.
