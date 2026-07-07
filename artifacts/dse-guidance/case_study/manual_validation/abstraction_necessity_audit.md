# Abstraction-necessity audit (P19 Phase 4)

Validates the strict necessity predicates against source. Structural only.

## Predicate correctness (per workload, cited in the per-workload audits)
- **resident_weight_object — necessary 11/11, but rests on configured K** (see residency_audit.md). The
  per-workload K is correct in `predicate_audit_table.csv` (the P17 K-rollup fix); the necessity itself is
  config-driven and is already flagged `suspicious` (necessity rests on configured/reference K). Honest, but
  must be presented as "necessary *if the configured K-loop is real*".
- **skinny_gemm_or_gemv_engine — necessary/useful 9–10/11.** The P17 rename + true_gemv/skinny split is
  confirmed warranted (residual "gemv" is mostly skinny projection at small captured M, not single-vector
  GEMV; primitive_frontier_audit.md). Predicate is discriminating.
- **matrix_engine — necessary 2/11 (corpus-narrow).** Correct: dense square GEMM dominates only rdt
  (depth-2 giant op) — and that op does not generalize (rdt2). So "matrix_engine necessary" is essentially
  an rdt-at-depth-2 artifact; honestly corpus-narrow.
- **low-bit abstractions — blocked (correct, with the precise reason).** bitvla source HAS real int2
  BitLinear packing (modeling_bitnet.py quantize_to_int2 + absmean scale + W8 act), but the **captured
  branch is fake-quant→f32** (enable_qlora=False → STE round/clamp → f32; only lm_head is torchao-int8).
  So blocked = **present_in_source_erased_by_export**, NOT absent and NOT a Merlin failure. The qdq-level
  recapture restores `quant_ext.dequantize` (capture-level ablation) — i.e. the path to unblock exists.
- **KV abstractions — blocked (correct, reason refined).** KV state is erased because the capture is a
  single forward with `use_cache=False` (openvla/molmoact) or a non-AR diffusion step — i.e. the K/decode
  loop is unrolled. molmoact's blocked-reason text says "attention lowered" but the real cause is
  `use_cache=False` (attention itself IS recovered) — a minor wording fix for that row.

## Semantic weaknesses found
- **Region attribution is degenerate in single-step diffusion captures** (rdt/pi05/xr0): nearly all ops
  labeled `repeated_head` (or fqn-pattern `prefix_builder` that are really per-layer qkv). The backbone/
  prefix/repeated split carries little discriminating information when the capture is one uniform stack with
  the VLM backbone host-side (entering as input features). This is a **capture-fidelity limit**, not a
  predicate bug — but the "region role" facts should be marked low-confidence for these workloads.
- **Many "possible" abstractions are boilerplate** (available, not gated by a discriminating signal). P16
  already separates `possible` from `necessary/useful`; the audit confirms `possible` ≈ "not demanded by
  this corpus", which is the honest reading.

## Verdict
- `boundary_necessity_matrix` (categorical): **main-slide** — discriminating, low-bit/KV correctly blocked,
  with the caveat that `resident_weight_object`/loop abstractions rest on configured K and region roles are
  low-confidence in single-step captures.
