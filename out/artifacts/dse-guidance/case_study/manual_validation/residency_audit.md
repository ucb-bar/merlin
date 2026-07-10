# Residency audit (P19 Phase 4)

Validates the residency / resident_weight_object conclusions against source. Structural only.

## What is faithful
- **Weights are summed per-op, NOT deduplicated** by identity/symbol/value (attribution.py add()). This is
  intentional and documented — the flat capture has no SSA weight identity. Consequence: `weight_bytes` is
  a per-op sum; **tied/shared weights are double-counted by design** (e.g. a real TinyLlama ties
  embed↔lm_head, but the *capture* unties them — small_llama/tiny_llama emit two distinct tensors, so no
  double-count *here*, but a deploy capture with tying would over-count). State this as a known over-count
  bound, not a unique-resident-set size.
- **rdt2 reframes the residency story**: it is NOT "one big resident matmul" (that was rdt-at-depth-2). Its
  residency target is the **loop-carried latent + the 4 KV-cache inputs** carried across the K denoise
  steps (KV enters the graph as inputs, host-computed). ⇒ residency-pressure is workload-shape-specific.

## What is config-driven (the load-bearing caveat)
- **K is sidecar/assumed for every workload** (MODEL_ARCH `loop_count_source="assumed"`); the captured MLIR
  is a single step (loop unrolled by torch.export). So `avoidable_weight_reload = weight_bytes·(K−1)`,
  `resident_*` residency pressure, and "weights reused across K" are all **extrapolations from an assumed
  K**, not IR facts. `resident_weight_object` is "necessary 11/11" only **under the configured-K residency
  predicate** (K>1 ∧ weight>1MB ∧ avoidable>weight) — already flagged suspicious in
  `predicate_audit_table.csv` (rests on configured K). **If K=1 the abstraction would not be necessary.**
- **xr0 K corrected 10→5** (P19): source `num_steps=5` (Xiaomi-Robotics-0/XR0.py); MODEL_ARCH + RECAP_MODELS
  said 10 (config drift). xr0 residency/avoidable-reload numbers now reflect K=5.
- **Several "decode" workloads are captured as prefill** (tiny_llama/molmoact/small_llama, `use_cache=False`,
  single forward) — so there is no KV-cache growth or decode-loop weight reuse in the capture; the K=32
  "decode" reuse is config-only.

## Verdict
- `residency_vs_K` / `decision_weight_residency`: **main-slide, with the configured/reference-K caveat**
  (the K axis is assumed; the *structural* shape — reload grows with K, resident is flat — is valid, the
  magnitudes are not). Already labeled "configured/reference K" in P16.
- `capacity_x_dtype`: **main-slide** — the int4<int8<bf16 capacity ordering is a pure byte-scaling fact
  (resident_int8_B = f32/4), structurally sound; magnitudes structural-only.
- Resident-pressure *ranking* across workloads is the robust signal; absolute resident bytes are random-init
  and must not be presented as deployment capacities.
