# Presentation candidate findings (all)

> A finding may be **main** only if tier A/B, not purely assumed, with a clear DSE implication, corroborated by a verification check, and needing no performance claim.

## Main (13)

- **head weight bytes** [tier A] — head_weight_bytes: resident-weight capacity requirement  _(workloads: bitvla, groot_n1d7, molmoact, openvla, pi05, rdt, rdt2, smolvla, tiny_llama, xr0; plot: —)_
- **total macs** [tier A] — total_macs: per-replan compute volume  _(workloads: bitvla, groot_n1d7, molmoact, openvla, pi05, rdt, rdt2, smolvla, tiny_llama, xr0; plot: —)_
- **n matmuls** [tier A] — n_matmuls: operator count to cover  _(workloads: bitvla, groot_n1d7, molmoact, openvla, pi05, rdt, rdt2, smolvla, tiny_llama, xr0; plot: —)_
- **matmul bias epilogues** [tier A] — matmul_bias_epilogues: fused epilogue slot present (bias) -> fused_requant_epilogue candidate  _(workloads: bitvla, groot_n1d7, molmoact, openvla, pi05, rdt, rdt2, smolvla, tiny_llama, xr0; plot: epilogue_pattern_counts)_
- **head cadence** [tier A] — head_cadence: repeated-head cadence (rate class)  _(workloads: bitvla, groot_n1d7, molmoact, openvla, pi05, rdt, rdt2, smolvla, tiny_llama, xr0; plot: —)_
- **accuracy int8 w8a8** [tier A] — accuracy_int8_w8a8: gates int8 as an accuracy-legal dtype candidate  _(workloads: bitvla, openvla, rdt2, small_llama, tiny_llama; plot: accuracy_gate_status)_
- **measured dispatch ratio** [tier A] — measured_dispatch_ratio: MEASURED host dispatch coupling (real runtime measurement)  _(workloads: ALL; plot: —)_
- **coverage under 10pct** [tier B] — coverage_under_10pct: best-covering primitive for this workload  _(workloads: ALL, bitvla, groot_n1d7, molmoact, openvla, pi05, rdt, rdt2, smolvla, tiny_llama, xr0; plot: primitive_coverage_heatmap)_
- **available parallelism** [tier B] — available_parallelism: low inter-op parallelism favors intra-op sharding (not many identical units kept busy by concurrency)  _(workloads: bitvla, groot_n1d7, molmoact, openvla, pi05, rdt, rdt2, smolvla, tiny_llama, xr0; plot: inter_op_parallelism_by_workload)_
- **avoidable weight reload** [tier B] — avoidable_weight_reload: resident_weight_object residency benefit (bytes), no bandwidth claim  _(workloads: bitvla, groot_n1d7, molmoact, openvla, pi05, rdt, rdt2, smolvla, tiny_llama, xr0; plot: avoidable_reload_by_region)_
- **resident int8 B** [tier B] — resident_int8_B: int8 resident-capacity requirement  _(workloads: bitvla, groot_n1d7, molmoact, openvla, pi05, rdt, rdt2, smolvla, tiny_llama, xr0; plot: resident_capacity_by_dtype)_
- **boundary pressure score** [tier B] — boundary_pressure_score: strong candidate boundary placement(s)  _(workloads: ALL; plot: boundary_placement_heatmap)_
- **max regret** [tier B] — max_regret: cross-workload coverage spread (overfit risk if high)  _(workloads: ALL; plot: primitive_regret_bar)_

## Backup (7)

- **dominant shape class** [tier B] — dominant_shape_class: dominant geometry class -> the primitive shape the DSE must cover  _(workloads: bitvla, groot_n1d7, molmoact, openvla, pi05, rdt, rdt2, smolvla, tiny_llama, xr0; plot: shape_class_mac_share)_
- **accumulator dtype** [tier B] — accumulator_dtype: accumulator width for the datapath  _(workloads: bitvla, groot_n1d7, molmoact, openvla, pi05, rdt, rdt2, smolvla, tiny_llama, xr0; plot: —)_
- **compute dtype** [tier B] — compute_dtype: storage/compute dtype contract  _(workloads: bitvla, groot_n1d7, molmoact, openvla, pi05, rdt, rdt2, smolvla, tiny_llama, xr0; plot: —)_
- **mac fraction dense gemm** [tier B] — mac_fraction_dense_gemm: distinct compute family -> specialized vs monolithic units  _(workloads: ALL; plot: —)_
- **mac fraction skinny gemm or gemv** [tier B] — mac_fraction_skinny_gemm_or_gemv: distinct compute family -> specialized vs monolithic units  _(workloads: ALL; plot: —)_
- **lowbit storage dequantized finding** [tier B] — lowbit_storage_dequantized_finding: quantized zoo stores weights low-bit but runs f32 matmuls (native low-bit compute + packed layout absent) -- real low-bit storage evidence  _(workloads: ZOO; plot: —)_
- **erased low-bit / KV structure** [tier D] — packed low-bit layout, scales, and KV/attention structure are erased/lowered in the capture; native low-bit & KV boundary placements are blocked/unavailable  _(workloads: ALL; plot: —)_

