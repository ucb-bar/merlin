# Signal findings report (all)

> Given only workload artifacts: what changes the future DSE search space. Every finding is recovered/derived from the captures (or a host measurement); **no quantity is claimed for unbuilt hardware** (cycles / area / energy / throughput are refused, not estimated). Organized by the DSE question each metric answers.

## Q_primitives: what compute primitives should DSE include?

_Signal metrics: coverage_under_10pct, dominant_shape_class, max_regret, n_matmuls, total_macs_

- **total macs** [tier A] — per-replan compute volume  _(workloads: bitvla, groot_n1d7, molmoact, openvla, pi05, rdt, rdt2, smolvla, tiny_llama, xr0)_
- **n matmuls** [tier A] — operator count to cover  _(workloads: bitvla, groot_n1d7, molmoact, openvla, pi05, rdt, rdt2, smolvla, tiny_llama, xr0)_
- **coverage under 10pct** [tier B] — best-covering primitive for this workload  _(workloads: ALL, bitvla, groot_n1d7, molmoact, openvla, pi05, rdt, rdt2, smolvla, tiny_llama, xr0)_
- **max regret** [tier B] — cross-workload coverage spread (overfit risk if high)  _(workloads: ALL)_

## Q_heterogeneity: should DSE explore heterogeneous / replicated units?

_Signal metrics: available_parallelism, clean_8way_mn_shards, mac_fraction_dense_gemm, mac_fraction_skinny_gemm_or_gemv, serialization_

- **available parallelism** [tier B] — low inter-op parallelism favors intra-op sharding (not many identical units kept busy by concurrency)  _(workloads: bitvla, groot_n1d7, molmoact, openvla, pi05, rdt, rdt2, smolvla, tiny_llama, xr0)_

## Q_residency: should DSE explore weight residency / packed stores?

_Signal metrics: avoidable_weight_reload, head_weight_bytes, resident_int8_B_

- **head weight bytes** [tier A] — resident-weight capacity requirement  _(workloads: bitvla, groot_n1d7, molmoact, openvla, pi05, rdt, rdt2, smolvla, tiny_llama, xr0)_
- **avoidable weight reload** [tier B] — resident_weight_object residency benefit (bytes), no bandwidth claim  _(workloads: bitvla, groot_n1d7, molmoact, openvla, pi05, rdt, rdt2, smolvla, tiny_llama, xr0)_
- **resident int8 B** [tier B] — int8 resident-capacity requirement  _(workloads: bitvla, groot_n1d7, molmoact, openvla, pi05, rdt, rdt2, smolvla, tiny_llama, xr0)_

## Q_command: should DSE explore command/loop/dispatch abstractions?

_Signal metrics: head_cadence, measured_dispatch_ratio, overlap_candidates_yes_

- **head cadence** [tier A] — repeated-head cadence (rate class)  _(workloads: bitvla, groot_n1d7, molmoact, openvla, pi05, rdt, rdt2, smolvla, tiny_llama, xr0)_
- **measured dispatch ratio** [tier A] — MEASURED host dispatch coupling (real runtime measurement)  _(workloads: ALL)_

## Q_lowbit: should DSE explore low-bit formats / numerical placement?

_Signal metrics: accumulator_dtype, accuracy_gate_report_present, accuracy_int8_w8a8, compute_dtype, lowbit_storage_dequantized_finding, matmul_bias_epilogues_

- **matmul bias epilogues** [tier A] — fused epilogue slot present (bias) -> fused_requant_epilogue candidate  _(workloads: bitvla, groot_n1d7, molmoact, openvla, pi05, rdt, rdt2, smolvla, tiny_llama, xr0)_
- **accuracy int8 w8a8** [tier A] — gates int8 as an accuracy-legal dtype candidate  _(workloads: bitvla, openvla, rdt2, small_llama, tiny_llama)_

## Q_boundary: where should the HW/SW boundary sit?

_Signal metrics: boundary_pressure_score, compiler_proofs_assumed, compiler_proofs_proven_for_workload, compiler_proofs_unknown_

- **boundary pressure score** [tier B] — strong candidate boundary placement(s)  _(workloads: ALL)_

## Q_readiness: what blocks quantitative ranking?

_No signal recovered for this question from the current corpus._

## What remains unclaimed (and the exact input needed)

- real deployment K + control rate: a deployment/runtime trace giving actual loop counts + control frequency
- per-unit throughput / latency / area / energy: a candidate design YAML (unit shapes + a cost model); then the future DSE tool computes them
- KV / attention structure + true data deps at loop level: a Level-2 loop-preserving, attention-not-lowered capture
- packed low-bit layout + scales for the recaptured models: a low-bit (packed weights + scale metadata) capture of the recaptured models
- fp8 / int4 accuracy gates: per-format accuracy runs (W8A8 already done) for fp8 / int4
- real-magnitude weights: full-size (non-random-init) captures of the same architectures

## Devil's advocate — robust vs corpus-limited

**Robust (structural, independent of magnitudes):** shape-class distribution, the recovered SSA data-dependency graph, the backbone/head role split, the dtype/epilogue numerical contract, and per-op MAC *fractions* (relative, not absolute). These hold regardless of weight magnitudes.

**Corpus-limited (treat as directional, not settled):**
- The 4 recaptures are small, random-init f32 instances — structure and provenance are real, but absolute byte/MAC magnitudes are a small instance, so any finding that leans on absolute size is directional only.
- low-bit / KV / attention structure is erased or lowered in the capture, so those candidates are blocked, not measured (see What remains unclaimed).
- 0 family-specific findings recovered: the corpus is small and structurally homogeneous across families, so cross-family separation is weak — expanding the corpus (see corpus_expansion_plan.md) is the prerequisite before any cross-family DSE claim.
