# DSE findings digest — scope: all

Self-contained summary of the workload-contract analysis. Every number is recovered from the captures or a host measurement; **no quantity is claimed for unbuilt hardware**. Each metric carries an evidence tier (A/measured = IR or real measurement; B = recovered/derived + recompute check; C = config/assumed; D = unavailable) so you can weight it yourself. Source CSV/YAML are in this same folder; the full per-fact trace (metric -> source artifact -> check) is in `unified_fact_table.csv`.

## 1. Headline metrics (canonical_signal_table.csv)

Grouped by the DSE question each answers. `entity` is the thing the metric is about (workload / abstraction / region).

### Q_primitives: what compute primitives should DSE include?

| metric | entity | value | unit | tier | strength | implication |
|---|---|---|---|---|---|---|
| coverage_under_10pct | tile_8x16 | 0.982847 | MAC-fraction | B | verified+corroborated | broadly-covering primitive for the DSE search space |
| coverage_under_10pct | tile_8x8 | 0.982847 | MAC-fraction | B | verified+corroborated | broadly-covering primitive for the DSE search space |
| coverage_under_10pct | gemv_lane_128 | 1.0 | MAC-fraction | B | verified | best-covering primitive for this workload |
| coverage_under_10pct | gemv_lane_128 | 1.0 | MAC-fraction | B | verified | best-covering primitive for this workload |
| coverage_under_10pct | gemv_lane_128 | 1.0 | MAC-fraction | B | verified | best-covering primitive for this workload |
| coverage_under_10pct | gemv_lane_128 | 1.0 | MAC-fraction | B | verified | best-covering primitive for this workload |
| coverage_under_10pct | tile_16x16 | 0.992687 | MAC-fraction | B | verified | best-covering primitive for this workload |
| coverage_under_10pct | tile_8x16 | 0.999761 | MAC-fraction | B | verified | best-covering primitive for this workload |
| coverage_under_10pct | gemv_lane_128 | 0.997687 | MAC-fraction | B | verified | best-covering primitive for this workload |
| coverage_under_10pct | tile_8x16 | 0.955611 | MAC-fraction | B | verified | best-covering primitive for this workload |
| coverage_under_10pct | gemv_lane_128 | 1.0 | MAC-fraction | B | verified | best-covering primitive for this workload |
| coverage_under_10pct | gemv_lane_128 | 0.999092 | MAC-fraction | B | verified | best-covering primitive for this workload |
| dominant_shape_class | bitvla | wide_skinny |  | B | single-source | dominant geometry class -> the primitive shape the DSE must cover |
| dominant_shape_class | groot_n1d7 | wide_skinny |  | B | single-source | dominant geometry class -> the primitive shape the DSE must cover |
| dominant_shape_class | molmoact | wide_skinny |  | B | single-source | dominant geometry class -> the primitive shape the DSE must cover |
| dominant_shape_class | openvla | wide_skinny |  | B | single-source | dominant geometry class -> the primitive shape the DSE must cover |
| dominant_shape_class | pi05 | wide_skinny |  | B | single-source | dominant geometry class -> the primitive shape the DSE must cover |
| dominant_shape_class | rdt | squareish_gemm |  | B | single-source | dominant geometry class -> the primitive shape the DSE must cover |
| dominant_shape_class | rdt2 | wide_skinny |  | B | single-source | dominant geometry class -> the primitive shape the DSE must cover |
| dominant_shape_class | smolvla | squareish_gemm |  | B | single-source | dominant geometry class -> the primitive shape the DSE must cover |
| dominant_shape_class | tiny_llama | wide_skinny |  | B | single-source | dominant geometry class -> the primitive shape the DSE must cover |
| dominant_shape_class | xr0 | wide_skinny |  | B | single-source | dominant geometry class -> the primitive shape the DSE must cover |
| max_regret | tile_8x16 | 0.999761 | MAC-fraction | B | verified | cross-workload coverage spread (overfit risk if high) |
| max_regret | tile_8x8 | 0.999761 | MAC-fraction | B | verified | cross-workload coverage spread (overfit risk if high) |
| n_matmuls | bitvla | 30 | ops | A | verified+corroborated | operator count to cover |
| n_matmuls | groot_n1d7 | 116 | ops | A | verified+corroborated | operator count to cover |
| n_matmuls | molmoact | 34 | ops | A | verified+corroborated | operator count to cover |
| n_matmuls | openvla | 30 | ops | A | verified+corroborated | operator count to cover |
| n_matmuls | pi05 | 777 | ops | A | verified+corroborated | operator count to cover |
| n_matmuls | rdt | 21 | ops | A | verified+corroborated | operator count to cover |
| n_matmuls | rdt2 | 26 | ops | A | verified+corroborated | operator count to cover |
| n_matmuls | smolvla | 302 | ops | A | verified+corroborated | operator count to cover |
| n_matmuls | tiny_llama | 30 | ops | A | verified+corroborated | operator count to cover |
| n_matmuls | xr0 | 19 | ops | A | verified+corroborated | operator count to cover |
| total_macs | bitvla | 39452672 | MACs | A | verified+corroborated | per-replan compute volume |
| total_macs | groot_n1d7 | 20393361408 | MACs | A | verified+corroborated | per-replan compute volume |
| total_macs | molmoact | 8419016704 | MACs | A | verified+corroborated | per-replan compute volume |
| total_macs | openvla | 15269888 | MACs | A | verified+corroborated | per-replan compute volume |
| total_macs | pi05 | 2146035695616 | MACs | A | verified+corroborated | per-replan compute volume |
| total_macs | rdt | 39466041344 | MACs | A | verified+corroborated | per-replan compute volume |
| total_macs | rdt2 | 991854592 | MACs | A | verified+corroborated | per-replan compute volume |
| total_macs | smolvla | 110595843584 | MACs | A | verified+corroborated | per-replan compute volume |
| total_macs | tiny_llama | 923795456 | MACs | A | verified+corroborated | per-replan compute volume |
| total_macs | xr0 | 1115879424 | MACs | A | verified+corroborated | per-replan compute volume |

### Q_heterogeneity: should DSE explore heterogeneous / replicated units?

| metric | entity | value | unit | tier | strength | implication |
|---|---|---|---|---|---|---|
| available_parallelism | bitvla | 1.5515 | work/span | B | verified | low inter-op parallelism favors intra-op sharding (not many identical units kept busy by concurrency) |
| available_parallelism | groot_n1d7 | 1.3034 | work/span | B | verified | low inter-op parallelism favors intra-op sharding (not many identical units kept busy by concurrency) |
| available_parallelism | molmoact | 1.1267 | work/span | B | verified | low inter-op parallelism favors intra-op sharding (not many identical units kept busy by concurrency) |
| available_parallelism | openvla | 1.9256 | work/span | B | verified | low inter-op parallelism favors intra-op sharding (not many identical units kept busy by concurrency) |
| available_parallelism | pi05 | 1.6125 | work/span | B | verified | low inter-op parallelism favors intra-op sharding (not many identical units kept busy by concurrency) |
| available_parallelism | rdt | 1.1117 | work/span | B | verified | low inter-op parallelism favors intra-op sharding (not many identical units kept busy by concurrency) |
| available_parallelism | rdt2 | 1.3617 | work/span | B | verified | low inter-op parallelism favors intra-op sharding (not many identical units kept busy by concurrency) |
| available_parallelism | smolvla | 1.3113 | work/span | B | verified | low inter-op parallelism favors intra-op sharding (not many identical units kept busy by concurrency) |
| available_parallelism | tiny_llama | 1.624 | work/span | B | verified | low inter-op parallelism favors intra-op sharding (not many identical units kept busy by concurrency) |
| available_parallelism | xr0 | 1.3304 | work/span | B | verified | low inter-op parallelism favors intra-op sharding (not many identical units kept busy by concurrency) |
| clean_8way_mn_shards | bitvla | 44 | op*axis | C | single-source | reduction-free M/N shards available |
| clean_8way_mn_shards | groot_n1d7 | 132 | op*axis | C | single-source | reduction-free M/N shards available |
| clean_8way_mn_shards | molmoact | 50 | op*axis | C | single-source | reduction-free M/N shards available |
| clean_8way_mn_shards | openvla | 30 | op*axis | C | single-source | reduction-free M/N shards available |
| clean_8way_mn_shards | pi05 | 1387 | op*axis | C | single-source | reduction-free M/N shards available |
| clean_8way_mn_shards | rdt | 24 | op*axis | C | single-source | reduction-free M/N shards available |
| clean_8way_mn_shards | rdt2 | 28 | op*axis | C | single-source | reduction-free M/N shards available |
| clean_8way_mn_shards | smolvla | 375 | op*axis | C | single-source | reduction-free M/N shards available |
| clean_8way_mn_shards | tiny_llama | 44 | op*axis | C | single-source | reduction-free M/N shards available |
| clean_8way_mn_shards | xr0 | 29 | op*axis | C | single-source | reduction-free M/N shards available |
| mac_fraction_dense_gemm | ALL | 0.3446 | MAC-fraction | B | single-source | distinct compute family -> specialized vs monolithic units |
| mac_fraction_skinny_gemm_or_gemv | ALL | 0.643 | MAC-fraction | B | single-source | distinct compute family -> specialized vs monolithic units |
| serialization | bitvla | some_parallelism | class | C | single-source | near-sequential DAG shape |
| serialization | groot_n1d7 | mostly_sequential | class | C | single-source | near-sequential DAG shape |
| serialization | molmoact | mostly_sequential | class | C | single-source | near-sequential DAG shape |
| serialization | openvla | some_parallelism | class | C | single-source | near-sequential DAG shape |
| serialization | pi05 | some_parallelism | class | C | single-source | near-sequential DAG shape |
| serialization | rdt | mostly_sequential | class | C | single-source | near-sequential DAG shape |
| serialization | rdt2 | mostly_sequential | class | C | single-source | near-sequential DAG shape |
| serialization | smolvla | mostly_sequential | class | C | single-source | near-sequential DAG shape |
| serialization | tiny_llama | some_parallelism | class | C | single-source | near-sequential DAG shape |
| serialization | xr0 | mostly_sequential | class | C | single-source | near-sequential DAG shape |

### Q_residency: should DSE explore weight residency / packed stores?

| metric | entity | value | unit | tier | strength | implication |
|---|---|---|---|---|---|---|
| avoidable_weight_reload | repeated_head | 34603008 | bytes | B | verified+corroborated | resident_weight_object residency benefit (bytes), no bandwidth claim |
| avoidable_weight_reload | repeated_head | 6601310208 | bytes | B | verified+corroborated | resident_weight_object residency benefit (bytes), no bandwidth claim |
| avoidable_weight_reload | repeated_head | 26512195584 | bytes | B | verified+corroborated | resident_weight_object residency benefit (bytes), no bandwidth claim |
| avoidable_weight_reload | repeated_head | 18874368 | bytes | B | verified+corroborated | resident_weight_object residency benefit (bytes), no bandwidth claim |
| avoidable_weight_reload | repeated_head | 15479341056 | bytes | B | verified+corroborated | resident_weight_object residency benefit (bytes), no bandwidth claim |
| avoidable_weight_reload | repeated_head | 1572864000 | bytes | B | verified+corroborated | resident_weight_object residency benefit (bytes), no bandwidth claim |
| avoidable_weight_reload | repeated_head | 1238958080 | bytes | B | verified+corroborated | resident_weight_object residency benefit (bytes), no bandwidth claim |
| avoidable_weight_reload | repeated_head | 1855134720 | bytes | B | verified+corroborated | resident_weight_object residency benefit (bytes), no bandwidth claim |
| avoidable_weight_reload | repeated_head | 3686793216 | bytes | B | verified+corroborated | resident_weight_object residency benefit (bytes), no bandwidth claim |
| avoidable_weight_reload | repeated_head | 676347904 | bytes | B | verified+corroborated | resident_weight_object residency benefit (bytes), no bandwidth claim |
| head_weight_bytes | bitvla | 5767168 | bytes | A | verified+corroborated | resident-weight capacity requirement |
| head_weight_bytes | groot_n1d7 | 2200436736 | bytes | A | verified+corroborated | resident-weight capacity requirement |
| head_weight_bytes | molmoact | 3787456512 | bytes | A | verified+corroborated | resident-weight capacity requirement |
| head_weight_bytes | openvla | 3145728 | bytes | A | verified+corroborated | resident-weight capacity requirement |
| head_weight_bytes | pi05 | 1719926784 | bytes | A | verified+corroborated | resident-weight capacity requirement |
| head_weight_bytes | rdt | 393216000 | bytes | A | verified+corroborated | resident-weight capacity requirement |
| head_weight_bytes | rdt2 | 309739520 | bytes | A | verified+corroborated | resident-weight capacity requirement |
| head_weight_bytes | smolvla | 206126080 | bytes | A | verified+corroborated | resident-weight capacity requirement |
| head_weight_bytes | tiny_llama | 614465536 | bytes | A | verified+corroborated | resident-weight capacity requirement |
| head_weight_bytes | xr0 | 169086976 | bytes | A | verified+corroborated | resident-weight capacity requirement |
| resident_int8_B | repeated_head | 1441792 | bytes | B | verified+corroborated | int8 resident-capacity requirement |
| resident_int8_B | repeated_head | 550109184 | bytes | B | verified+corroborated | int8 resident-capacity requirement |
| resident_int8_B | repeated_head | 946864128 | bytes | B | verified+corroborated | int8 resident-capacity requirement |
| resident_int8_B | repeated_head | 786432 | bytes | B | verified+corroborated | int8 resident-capacity requirement |
| resident_int8_B | repeated_head | 429981696 | bytes | B | verified+corroborated | int8 resident-capacity requirement |
| resident_int8_B | repeated_head | 98304000 | bytes | B | verified+corroborated | int8 resident-capacity requirement |
| resident_int8_B | repeated_head | 77434880 | bytes | B | verified+corroborated | int8 resident-capacity requirement |
| resident_int8_B | repeated_head | 51531520 | bytes | B | verified | int8 resident-capacity requirement |
| resident_int8_B | repeated_head | 153616384 | bytes | B | verified+corroborated | int8 resident-capacity requirement |
| resident_int8_B | repeated_head | 42271744 | bytes | B | verified+corroborated | int8 resident-capacity requirement |

### Q_command: should DSE explore command/loop/dispatch abstractions?

| metric | entity | value | unit | tier | strength | implication |
|---|---|---|---|---|---|---|
| head_cadence | bitvla | token_loop |  | A | corroborated x2 | repeated-head cadence (rate class) |
| head_cadence | groot_n1d7 | K_times_per_replan |  | A | corroborated x2 | repeated-head cadence (rate class) |
| head_cadence | molmoact | token_loop |  | A | corroborated x2 | repeated-head cadence (rate class) |
| head_cadence | openvla | token_loop |  | A | corroborated x2 | repeated-head cadence (rate class) |
| head_cadence | pi05 | K_times_per_replan |  | A | corroborated x2 | repeated-head cadence (rate class) |
| head_cadence | rdt | K_times_per_replan |  | A | corroborated x2 | repeated-head cadence (rate class) |
| head_cadence | rdt2 | K_times_per_replan |  | A | corroborated x2 | repeated-head cadence (rate class) |
| head_cadence | smolvla | K_times_per_replan |  | A | corroborated x2 | repeated-head cadence (rate class) |
| head_cadence | tiny_llama | token_loop |  | A | corroborated x2 | repeated-head cadence (rate class) |
| head_cadence | xr0 | K_times_per_replan |  | A | corroborated x2 | repeated-head cadence (rate class) |
| measured_dispatch_ratio | ALL | measured | ratio | A | measured | MEASURED host dispatch coupling (real runtime measurement) |
| overlap_candidates_yes | bitvla | 4 | candidates | C | single-source | phase overlaps structurally permitted |
| overlap_candidates_yes | groot_n1d7 | 3 | candidates | C | single-source | phase overlaps structurally permitted |
| overlap_candidates_yes | molmoact | 4 | candidates | C | single-source | phase overlaps structurally permitted |
| overlap_candidates_yes | openvla | 4 | candidates | C | single-source | phase overlaps structurally permitted |
| overlap_candidates_yes | pi05 | 4 | candidates | C | single-source | phase overlaps structurally permitted |
| overlap_candidates_yes | rdt | 3 | candidates | C | single-source | phase overlaps structurally permitted |
| overlap_candidates_yes | rdt2 | 3 | candidates | C | single-source | phase overlaps structurally permitted |
| overlap_candidates_yes | smolvla | 4 | candidates | C | single-source | phase overlaps structurally permitted |
| overlap_candidates_yes | tiny_llama | 3 | candidates | C | single-source | phase overlaps structurally permitted |
| overlap_candidates_yes | xr0 | 3 | candidates | C | single-source | phase overlaps structurally permitted |

### Q_lowbit: should DSE explore low-bit formats / numerical placement?

| metric | entity | value | unit | tier | strength | implication |
|---|---|---|---|---|---|---|
| accumulator_dtype | backbone_once | f32 | dtype | B | single-source | accumulator width for the datapath |
| accumulator_dtype | repeated_head | f32 | dtype | B | single-source | accumulator width for the datapath |
| accumulator_dtype | backbone_once | f32 | dtype | B | single-source | accumulator width for the datapath |
| accumulator_dtype | backbone_once | f32 | dtype | B | single-source | accumulator width for the datapath |
| accumulator_dtype | backbone_once | f32 | dtype | B | single-source | accumulator width for the datapath |
| accumulator_dtype | repeated_head | f32 | dtype | B | single-source | accumulator width for the datapath |
| accumulator_dtype | repeated_head | f32 | dtype | B | single-source | accumulator width for the datapath |
| accumulator_dtype | backbone_once | f32 | dtype | B | single-source | accumulator width for the datapath |
| accumulator_dtype | backbone_once | f32 | dtype | B | single-source | accumulator width for the datapath |
| accumulator_dtype | backbone_once | f32 | dtype | B | single-source | accumulator width for the datapath |
| accuracy_gate_report_present | ALL | yes |  | A | measured | measured int8 accuracy summary (real measurement) |
| accuracy_int8_w8a8 | bitvla | pass | band | A | measured | gates int8 as an accuracy-legal dtype candidate |
| accuracy_int8_w8a8 | groot_n1d7 | unavailable | band | D | verified | gates int8 as an accuracy-legal dtype candidate |
| accuracy_int8_w8a8 | molmoact | unavailable | band | D | verified | gates int8 as an accuracy-legal dtype candidate |
| accuracy_int8_w8a8 | openvla | pass | band | A | measured | gates int8 as an accuracy-legal dtype candidate |
| accuracy_int8_w8a8 | pi05 | unavailable | band | D | verified | gates int8 as an accuracy-legal dtype candidate |
| accuracy_int8_w8a8 | rdt | unavailable | band | D | verified | gates int8 as an accuracy-legal dtype candidate |
| accuracy_int8_w8a8 | rdt2 | pass | band | A | measured | gates int8 as an accuracy-legal dtype candidate |
| accuracy_int8_w8a8 | small_llama | pass | band | A | measured | MEASURED int8 W8A8 accuracy (real measurement) |
| accuracy_int8_w8a8 | smolvla | unavailable | band | D | verified | gates int8 as an accuracy-legal dtype candidate |
| accuracy_int8_w8a8 | tiny_llama | pass | band | A | measured | gates int8 as an accuracy-legal dtype candidate |
| accuracy_int8_w8a8 | xr0 | unavailable | band | D | verified | gates int8 as an accuracy-legal dtype candidate |
| compute_dtype | bitvla | f32 | dtype | B | single-source | storage/compute dtype contract |
| compute_dtype | groot_n1d7 | f32 | dtype | B | single-source | storage/compute dtype contract |
| compute_dtype | molmoact | f32 | dtype | B | single-source | storage/compute dtype contract |
| compute_dtype | openvla | f32 | dtype | B | single-source | storage/compute dtype contract |
| compute_dtype | pi05 | f32 | dtype | B | single-source | storage/compute dtype contract |
| compute_dtype | rdt | f32 | dtype | B | single-source | storage/compute dtype contract |
| compute_dtype | rdt2 | f32 | dtype | B | single-source | storage/compute dtype contract |
| compute_dtype | smolvla | f32 | dtype | B | single-source | storage/compute dtype contract |
| compute_dtype | tiny_llama | f32 | dtype | B | single-source | storage/compute dtype contract |
| compute_dtype | xr0 | f32 | dtype | B | single-source | storage/compute dtype contract |
| lowbit_storage_dequantized_finding | ZOO | present |  | B | single-source | quantized zoo stores weights low-bit but runs f32 matmuls (native low-bit compute + packed layout absent) -- real low-bit storage evidence |
| matmul_bias_epilogues | bitvla | 8 | ops | A | verified | fused epilogue slot present (bias) -> fused_requant_epilogue candidate |
| matmul_bias_epilogues | groot_n1d7 | 116 | ops | A | verified | fused epilogue slot present (bias) -> fused_requant_epilogue candidate |
| matmul_bias_epilogues | molmoact | 8 | ops | A | verified | fused epilogue slot present (bias) -> fused_requant_epilogue candidate |
| matmul_bias_epilogues | openvla | 12 | ops | A | verified | fused epilogue slot present (bias) -> fused_requant_epilogue candidate |
| matmul_bias_epilogues | pi05 | 530 | ops | A | verified | fused epilogue slot present (bias) -> fused_requant_epilogue candidate |
| matmul_bias_epilogues | rdt | 21 | ops | A | verified | fused epilogue slot present (bias) -> fused_requant_epilogue candidate |
| matmul_bias_epilogues | rdt2 | 18 | ops | A | verified | fused epilogue slot present (bias) -> fused_requant_epilogue candidate |
| matmul_bias_epilogues | smolvla | 173 | ops | A | verified | fused epilogue slot present (bias) -> fused_requant_epilogue candidate |
| matmul_bias_epilogues | tiny_llama | 0 | ops | A | verified | fused epilogue slot present (bias) -> fused_requant_epilogue candidate |
| matmul_bias_epilogues | xr0 | 14 | ops | A | verified | fused epilogue slot present (bias) -> fused_requant_epilogue candidate |

### Q_boundary: where should the HW/SW boundary sit?

| metric | entity | value | unit | tier | strength | implication |
|---|---|---|---|---|---|---|
| boundary_pressure_score | async_queue | 14 | evidence | B | verified+corroborated | strong candidate boundary placement(s) |
| boundary_pressure_score | dma_engine | 14 | evidence | B | verified+corroborated | strong candidate boundary placement(s) |
| boundary_pressure_score | event_token | 14 | evidence | B | verified+corroborated | strong candidate boundary placement(s) |
| boundary_pressure_score | fused_requant_epilogue | 13 | evidence | B | verified+corroborated | strong candidate boundary placement(s) |
| boundary_pressure_score | loop_carried_state_handle | 12 | evidence | B | verified+corroborated | strong candidate boundary placement(s) |
| boundary_pressure_score | partial_sum_object | 13 | evidence | B | verified+corroborated | strong candidate boundary placement(s) |
| boundary_pressure_score | region_level_dispatch | 14 | evidence | B | verified+corroborated | strong candidate boundary placement(s) |
| boundary_pressure_score | resident_weight_object | 15 | evidence | B | verified+corroborated | strong candidate boundary placement(s) |
| compiler_proofs_assumed | ALL | 5 | axes | C | single-source | abstraction axes with compiler proof status = assumed |
| compiler_proofs_proven_for_workload | ALL | 1 | axes | C | single-source | abstraction axes with compiler proof status = proven_for_workload |
| compiler_proofs_unknown | ALL | 4 | axes | C | single-source | abstraction axes with compiler proof status = unknown |

## 2. Per-operator hotspots


> Which few operators dominate the constraints DSE must size for. Structural quantities (MACs / weight bytes / tile padding waste / avoidable reload) recovered from the capture — no latency, throughput, or performance claim.

Total operators analyzed: **1385**.

**Dominant op (by MACs):** `` in rdt — 4096x4096x2048 = 34,359,738,368 MACs (87% of its workload), class squareish_gemm.

## Top ops by MACs

| workload | op | shape M×N×K | MACs | % of workload | class |
|---|---|---|---|---|---|
| rdt |  | 4096×4096×2048 | 34,359,738,368 | 87% | squareish_gemm |
| pi05 | model.paligemma_with_expert.paligemma.model.language_model.layers.slice(None, 18, None).0.mlp.gate_proj | 968×16384×2048 | 32,480,690,176 | 2% | wide_skinny |
| pi05 | model.paligemma_with_expert.paligemma.model.language_model.layers.slice(None, 18, None).0.mlp.up_proj | 968×16384×2048 | 32,480,690,176 | 2% | wide_skinny |
| pi05 | model.paligemma_with_expert.paligemma.model.language_model.layers.slice(None, 18, None).0.mlp.down_proj | 968×2048×16384 | 32,480,690,176 | 2% | projection_like |
| pi05 | model.paligemma_with_expert.paligemma.model.language_model.layers.slice(None, 18, None).1.mlp.gate_proj | 968×16384×2048 | 32,480,690,176 | 2% | wide_skinny |
| pi05 | model.paligemma_with_expert.paligemma.model.language_model.layers.slice(None, 18, None).1.mlp.up_proj | 968×16384×2048 | 32,480,690,176 | 2% | wide_skinny |
| pi05 | model.paligemma_with_expert.paligemma.model.language_model.layers.slice(None, 18, None).1.mlp.down_proj | 968×2048×16384 | 32,480,690,176 | 2% | projection_like |
| pi05 | model.paligemma_with_expert.paligemma.model.language_model.layers.slice(None, 18, None).2.mlp.gate_proj | 968×16384×2048 | 32,480,690,176 | 2% | wide_skinny |
| pi05 | model.paligemma_with_expert.paligemma.model.language_model.layers.slice(None, 18, None).2.mlp.up_proj | 968×16384×2048 | 32,480,690,176 | 2% | wide_skinny |
| pi05 | model.paligemma_with_expert.paligemma.model.language_model.layers.slice(None, 18, None).2.mlp.down_proj | 968×2048×16384 | 32,480,690,176 | 2% | projection_like |

## Top ops by tile padding waste (tile-hostility)

| workload | op | shape M×N×K | best tile waste | class |
|---|---|---|---|---|
| rdt |  | 1×2048×256 | 7.000 | gemv_like |
| rdt |  | 1×2048×2048 | 7.000 | gemv_like |
| rdt |  | 1×2048×256 | 7.000 | gemv_like |
| rdt |  | 1×2048×2048 | 7.000 | gemv_like |
| openvla | lm_head | 1×512×128 | 7.000 | gemv_like |
| openvla |  | 1×512×128 | 7.000 | gemv_like |
| openvla |  | 1×512×128 | 7.000 | gemv_like |
| openvla |  | 1×512×128 | 7.000 | gemv_like |
| openvla |  | 1×128×512 | 7.000 | gemv_like |
| openvla |  | 1×256×128 | 7.000 | gemv_like |

## Regions by avoidable weight reload (residency target)

| workload | region | avoidable reload (B) | weight bytes (B) |
|---|---|---|---|
| molmoact | repeated_head | 26,512,195,584 | 3,787,456,512 |
| pi05 | repeated_head | 15,479,341,056 | 1,719,926,784 |
| groot_n1d7 | repeated_head | 6,601,310,208 | 2,200,436,736 |
| tiny_llama | repeated_head | 3,686,793,216 | 614,465,536 |
| smolvla | repeated_head | 1,855,134,720 | 206,126,080 |
| rdt | repeated_head | 1,572,864,000 | 393,216,000 |
| rdt2 | repeated_head | 1,238,958,080 | 309,739,520 |
| xr0 | repeated_head | 676,347,904 | 169,086,976 |
| bitvla | repeated_head | 34,603,008 | 5,767,168 |
| openvla | repeated_head | 18,874,368 | 3,145,728 |

## 3. Abstraction necessity (strict — what DSE should commit to)


> Strict replacement for the permissive support table: each abstraction is classified per workload as **necessary / useful / possible / blocked / not_applicable** by a threshold predicate over the recovered signals (not `any-X` presence). 'possible' = available but not gated by a discriminating signal; 'blocked' = the capture erased the needed structure.

**Corpus rollup:** 4 necessary · 5 useful · 11 possible · 7 blocked · 0 not-applicable (of 27 abstractions).

| abstraction | macro | bitvla | groot_n1d7 | molmoact | openvla | pi05 | rdt | rdt2 | smolvla | tiny_llama | xr0 | predicate | needs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| resident_weight_object | **necessary** | necessary | necessary | necessary | necessary | necessary | necessary | necessary | necessary | necessary | necessary | repeated weight reuse (K>1), resident weights >1MB, avoidable_reload>weight_bytes (per-workload K/MB in predicate_audit) | loop-preserving capture (K/cadence are configured/reference) |
| skinny_gemm_or_gemv_engine | **necessary** | necessary | necessary | necessary | necessary | necessary | useful | necessary | useful | necessary | necessary | gemv/skinny MAC fraction >0.5 (true_gemv+skinny; split in predicate_audit) | — |
| epilogue_requant_unit | **necessary** | useful | necessary | useful | necessary | necessary | necessary | necessary | necessary | possible | necessary | epilogue has scale+activation | — |
| fused_requant_epilogue | **necessary** | useful | necessary | useful | necessary | necessary | necessary | necessary | necessary | possible | necessary | epilogue has scale+activation | — |
| matrix_engine | **useful** | possible | possible | possible | possible | possible | necessary | possible | necessary | possible | possible | dense (squareish_gemm) MAC fraction >0.5 | — |
| accumulator_merge | **useful** | useful | useful | useful | useful | useful | useful | useful | useful | useful | useful | K-shard available but reduction-free M/N also possible | — |
| bounded_loop_command | **useful** | useful | useful | useful | useful | useful | useful | useful | useful | useful | useful | K>1 cadence (configured/reference, not IR-recovered; needs a loop-preserving capture; per-workload K in predicate_audit) | loop-preserving capture (K/cadence are configured/reference) |
| loop_carried_state_handle | **useful** | useful | useful | useful | useful | useful | useful | useful | useful | useful | useful | K>1 cadence (configured/reference, not IR-recovered; needs a loop-preserving capture; per-workload K in predicate_audit) | loop-preserving capture (K/cadence are configured/reference) |
| partial_sum_object | **useful** | useful | useful | useful | useful | useful | useful | useful | useful | useful | useful | K-shard available but reduction-free M/N also possible | — |
| accumulator_commit | **possible** | possible | possible | possible | possible | possible | possible | possible | possible | possible | possible | available; not gated by a discriminating signal | — |
| async_queue | **possible** | possible | possible | possible | possible | possible | possible | possible | possible | possible | possible | available; not gated by a discriminating signal | — |
| decode_loop_controller | **possible** | possible | not_applicable | possible | possible | not_applicable | not_applicable | not_applicable | not_applicable | possible | not_applicable | available; not gated by a discriminating signal | loop-preserving capture (K/cadence are configured/reference) |
| dma_engine | **possible** | possible | possible | possible | possible | possible | possible | possible | possible | possible | possible | available; not gated by a discriminating signal | — |
| double_buffered_action_chunk | **possible** | possible | possible | possible | possible | possible | possible | possible | possible | not_applicable | not_applicable | available; not gated by a discriminating signal | loop-preserving capture (K/cadence are configured/reference) |
| event_token | **possible** | possible | possible | possible | possible | possible | possible | possible | possible | possible | possible | available; not gated by a discriminating signal | — |
| multi_stream_dma_descriptor | **possible** | possible | possible | possible | possible | possible | possible | possible | possible | possible | possible | available; not gated by a discriminating signal | — |
| persistent_command_buffer | **possible** | possible | possible | possible | possible | possible | possible | possible | possible | possible | possible | available; not gated by a discriminating signal | — |
| prefetch_descriptor | **possible** | possible | possible | possible | possible | possible | possible | possible | possible | possible | possible | available; not gated by a discriminating signal | — |
| producer_consumer_queue | **possible** | possible | possible | possible | possible | possible | possible | possible | possible | not_applicable | not_applicable | available; not gated by a discriminating signal | loop-preserving capture (K/cadence are configured/reference) |
| region_level_dispatch | **possible** | possible | possible | possible | possible | possible | possible | possible | possible | possible | possible | available; not gated by a discriminating signal | — |
| fused_dequant_matmul | **blocked** | blocked | blocked | blocked | blocked | blocked | blocked | blocked | blocked | blocked | blocked | low-bit/packed structure dequantized in the capture | low-bit recapture (packed weights + scales + per-format accuracy) |
| kv_cache_object | **blocked** | blocked | not_applicable | blocked | blocked | not_applicable | not_applicable | not_applicable | not_applicable | blocked | not_applicable | attention/KV lowered; structure not recovered | loop-preserving, attention-not-lowered capture |
| native_lowbit_matmul | **blocked** | blocked | blocked | blocked | blocked | blocked | blocked | blocked | blocked | blocked | blocked | low-bit/packed structure dequantized in the capture | low-bit recapture (packed weights + scales + per-format accuracy) |
| packed_lowbit_tensor | **blocked** | blocked | blocked | blocked | blocked | blocked | blocked | blocked | blocked | blocked | blocked | low-bit/packed structure dequantized in the capture | low-bit recapture (packed weights + scales + per-format accuracy) |
| prefix_kv_object | **blocked** | blocked | not_applicable | blocked | blocked | not_applicable | not_applicable | not_applicable | not_applicable | blocked | not_applicable | attention/KV lowered; structure not recovered | loop-preserving, attention-not-lowered capture |
| resident_packed_weight_object | **blocked** | blocked | blocked | blocked | blocked | blocked | blocked | blocked | blocked | blocked | blocked | low-bit/packed structure dequantized in the capture | low-bit recapture (packed weights + scales + per-format accuracy) |
| scale_object | **blocked** | blocked | blocked | blocked | blocked | blocked | blocked | blocked | blocked | blocked | blocked | low-bit/packed structure dequantized in the capture | low-bit recapture (packed weights + scales + per-format accuracy) |

## 4. Primitive-set frontier


> A primitive SET covers an op if ANY member tiles it under 10% pad waste. The headline search-space result: one primitive is not enough — the best single primitive leaves a workload badly covered (low worst-workload), while a {tile + GEMV-lane} pair covers the corpus. Structural coverage only, no performance.

| set size | best primitive set | worst-workload | macro (mean) | micro (MAC-wt) | max regret |
|---|---|---|---|---|---|
| 1 | gemv_lane_64 | 0.13 | 0.80 | 0.64 | 0.87 |
| 2 | gemv_lane_64 + tile_8x16 | 1.00 | 1.00 | 1.00 | 0.00 |
| 3 | gemv_lane_128 + gemv_lane_64 + tile_8x16 | 1.00 | 1.00 | 1.00 | 0.00 |

## 5. Operator Pareto hotspots


> How many top ops are needed to reach a MAC threshold — whether DSE should size for a few giant ops or many even ones. Structural (MAC/byte share), no performance.

| workload | n_ops | k@50%MAC | k@80%MAC | k@90%MAC | k@95%MAC | top-op MAC share |
|---|---|---|---|---|---|---|
| bitvla | 30 | 5 | 10 | 12 | 14 | 11% |
| groot_n1d7 | 116 | 27 | 56 | 77 | 87 | 2% |
| molmoact | 34 | 4 | 10 | 16 | 20 | 13% |
| openvla | 30 | 6 | 11 | 13 | 16 | 9% |
| pi05 | 777 | 34 | 66 | 193 | 341 | 2% |
| rdt | 21 | 1 | 1 | 3 | 9 | 87% |
| rdt2 | 26 | 6 | 14 | 17 | 19 | 12% |
| smolvla | 302 | 23 | 76 | 116 | 163 | 2% |
| tiny_llama | 30 | 6 | 10 | 14 | 18 | 10% |
| xr0 | 19 | 5 | 7 | 8 | 10 | 12% |

## 6. Capture fidelity (what the flat capture erased)


> The likely central result: which structural features the flat capture preserves vs erased. `strong`=recovered from IR; `assumed (config K)`=loop count is a reference value, not captured; `erased`=lowered/dequantized away; `measured (host)`=real host measurement; `not_claimed`=needs a target design. Findings that depend on `assumed`/`erased` rows are capture-limited.

| feature | bitvla | groot_n1d7 | molmoact | openvla | pi05 | rdt | rdt2 | smolvla | tiny_llama | xr0 |
|---|---|---|---|---|---|---|---|---|---|---|
| op_shapes_MNK | strong | strong | strong | strong | strong | strong | strong | strong | strong | strong |
| region_roles | strong | strong | strong | strong | strong | strong | strong | strong | strong | strong |
| dtype_information | strong | strong | strong | strong | strong | strong | strong | strong | strong | strong |
| attention_bmm_qkT_attnV | n/a (attention-free here) | recovered (32 ops) | n/a (attention-free here) | n/a (attention-free here) | recovered (232 ops) | recovered (25 ops) | n/a (attention-free here) | recovered (84 ops) | n/a (attention-free here) | recovered (14 ops) |
| softmax | recovered | recovered | recovered | recovered | recovered | n/a | recovered | recovered | recovered | n/a |
| normalization | n/a (norm as elementwise primitives) | recovered | n/a (norm as elementwise primitives) | n/a (norm as elementwise primitives) | recovered | n/a (norm as elementwise primitives) | n/a (norm as elementwise primitives) | recovered | n/a (norm as elementwise primitives) | n/a (norm as elementwise primitives) |
| K_or_decode_loop | recovered (K=7, IR scf.for) | recovered (K=4, IR scf.for) | recovered (K=8, IR scf.for) | recovered (K=7, IR scf.for) | recovered (K=10, IR scf.for) | recovered (K=5, IR scf.for) | recovered (K=5, IR scf.for) | recovered (K=10, IR scf.for) | recovered (K=7, IR scf.for) | recovered (K=5, IR scf.for) |
| kv_cache_state | recovered (79872 B, IR iter_arg) | n/a (prefix-KV invariant, closed-over) | recovered (262144 B, IR iter_arg) | recovered (221184 B, IR iter_arg) | n/a (prefix-KV invariant, closed-over) | n/a (prefix-KV invariant, closed-over) | n/a (prefix-KV invariant, closed-over) | n/a (prefix-KV invariant, closed-over) | recovered (61440 B, IR iter_arg) | n/a (prefix-KV invariant, closed-over) |
| loop_carried_state | recovered (5 iter_args: counter,kv_cache,token_buffer) | recovered (2 iter_args: counter,latent) | recovered (5 iter_args: counter,kv_cache,token_buffer) | recovered (5 iter_args: counter,kv_cache,token_buffer) | recovered (2 iter_args: counter,latent) | recovered (3 iter_args: counter,latent) | recovered (2 iter_args: counter,latent) | recovered (2 iter_args: counter,latent) | recovered (5 iter_args: counter,kv_cache,token_buffer) | recovered (2 iter_args: counter,latent) |
| packed_lowbit_layout | erased | erased | erased | erased | erased | erased | erased | erased | erased | erased |
| scale_metadata | erased | erased | erased | erased | erased | erased | erased | erased | erased | erased |
| host_dispatch_count | measured (host) | measured (host) | measured (host) | measured (host) | measured (host) | measured (host) | measured (host) | measured (host) | measured (host) | measured (host) |
| target_latency_cycles | not_claimed | not_claimed | not_claimed | not_claimed | not_claimed | not_claimed | not_claimed | not_claimed | not_claimed | not_claimed |

**Per-workload DSE risk:**

- **bitvla** (autoregressive_vla, severity low): lost async_backbone_head_overlap, action_chunk_horizon, replan_deadline; hides axes decode_kv_cache_path, autonomous_K_loop, command_batching, resident_prefix_kv, backbone_head_partition, async_chunk_overlap
- **groot_n1d7** (diffusion, severity low): lost action_chunk_horizon, replan_deadline; hides axes resident_action_head_weights, autonomous_K_loop, command_batching, async_chunk_overlap, backbone_head_partition
- **molmoact** (autoregressive_vla, severity low): lost async_backbone_head_overlap, action_chunk_horizon, replan_deadline; hides axes decode_kv_cache_path, autonomous_K_loop, command_batching, resident_prefix_kv, backbone_head_partition, async_chunk_overlap
- **openvla** (autoregressive_vla, severity low): lost async_backbone_head_overlap, action_chunk_horizon, replan_deadline; hides axes decode_kv_cache_path, autonomous_K_loop, command_batching, resident_prefix_kv, backbone_head_partition, async_chunk_overlap
- **pi05** (flow_matching, severity low): lost async_backbone_head_overlap, action_chunk_horizon, replan_deadline; hides axes resident_action_head_weights, autonomous_K_loop, command_batching, resident_prefix_kv, backbone_head_partition, async_chunk_overlap
- **rdt** (diffusion, severity low): lost action_chunk_horizon, replan_deadline; hides axes resident_action_head_weights, autonomous_K_loop, command_batching, async_chunk_overlap, backbone_head_partition
- **rdt2** (diffusion, severity low): lost action_chunk_horizon, replan_deadline; hides axes resident_action_head_weights, autonomous_K_loop, command_batching, async_chunk_overlap, backbone_head_partition
- **smolvla** (flow_matching, severity low): lost async_backbone_head_overlap, action_chunk_horizon, replan_deadline; hides axes resident_action_head_weights, autonomous_K_loop, command_batching, resident_prefix_kv, backbone_head_partition, async_chunk_overlap
- **tiny_llama** (llm, severity low): lost async_backbone_head_overlap; hides axes decode_kv_cache_path, autonomous_K_loop, command_batching, resident_prefix_kv, backbone_head_partition, async_chunk_overlap
- **xr0** (diffusion, severity low): lost async_backbone_head_overlap; hides axes resident_action_head_weights, autonomous_K_loop, command_batching, resident_prefix_kv, backbone_head_partition, async_chunk_overlap

## 7. Decision-question scorecard


> The few decisions a future DSE tool must make, each answered from the workload analysis with its caveat. A metric earns its place only by answering one of these.

| # | decision question | answer (from analysis) | caveat |
|---|---|---|---|
| Q1 | Q1 best single primitive (worst-workload coverage)? | gemv_lane_64 -> worst 0.13, macro 0.80 | no single primitive covers every workload |
| Q2 | Q2 best 2-primitive set? | gemv_lane_64+tile_8x16 -> worst 1.00 (vs 0.13 single) | search primitive SETS, not one tile |
| Q3 | Q3 capacity x dtype residency thresholds? | see decision_capacity_dtype plot (int4<int8<bf16 budget to fit) | repeated-head weights only; K is configured/reference |
| Q4 | Q4 sharding axis for top-MAC ops? | M/N reduction-free vs K partial-sum (see decision_sharding_per_top_op) | communication bytes, not latency |
| Q5 | Q5 which abstractions are NECESSARY (not just possible)? | 4 necessary, 5 useful, 11 possible, 7 blocked, 0 N/A | strict predicate; low-bit abstractions blocked by capture |
| Q6 | Q6 which conclusions are driven by one workload (RDT)? | dense-MAC dominance macro 0.1395 vs micro 0.0397; collapses if removed: none | micro view is biased by RDT's 87%-of-workload op |
| Q7 | Q7 which claims depend on configured K (capture fidelity)? | all residency / loop / command claims (K is config/reference) | needs a loop-preserving capture; see capture_fidelity_matrix |

## 8. Leave-one-workload-out robustness


> Anti-overfitting: each cross-workload finding recomputed dropping one workload. A finding that flips is corpus-specific, not general.

## best_2_primitive_set

- all: ['gemv_lane_64', 'tile_8x16']
- all_worst: 0.9977
- all_macro: 0.9998
- loo_changes_winner: ['smolvla']
- robust: False

## dense_gemm_mac_dominance

- macro: 0.1395
- micro: 0.0397
- micro_loo: {'bitvla': 0.0397, 'groot_n1d7': 0.04, 'molmoact': 0.0398, 'openvla': 0.0397, 'pi05': 0.5076, 'rdt': 0.0253, 'rdt2': 0.0397, 'smolvla': 0.0155, 'tiny_llama': 0.0397, 'xr0': 0.0397}
- collapses_if_removed: []
- note: consistent across views

## residency_pressure_rank

- all: ['molmoact', 'pi05', 'groot_n1d7', 'tiny_llama', 'smolvla', 'rdt', 'rdt2', 'xr0', 'bitvla', 'openvla']
- top: molmoact
- note: absolute bytes are small/random-init; ranking is the robust signal


## Appendix — abstraction support breadth (possible-placement view only)


> Replaces raw support counts with corpus **coverage**: for each candidate system abstraction, the fraction of workloads / MACs / weight bytes / regions that imply it, its compiler-proof status, and an overfit flag (single-workload support is corpus-overfit risk). Support breadth, not a performance ranking.

| abstraction | workloads | MAC cov | byte cov | proof | overfit |
|---|---|---|---|---|---|
| resident_weight_object | 10 (100%) | 100% | 100% | unknown | low |
| async_queue | 10 (100%) | 100% | 100% | assumed | low |
| dma_engine | 10 (100%) | 100% | 100% | no_proof_axis | low |
| event_token | 10 (100%) | 100% | 100% | no_proof_axis | low |
| region_level_dispatch | 10 (100%) | 100% | 100% | no_proof_axis | low |
| partial_sum_object | 10 (100%) | 100% | 100% | no_proof_axis | low |
| bounded_loop_command | 10 (100%) | 100% | 100% | assumed | low |
| loop_carried_state_handle | 10 (100%) | 100% | 100% | assumed | low |
| multi_stream_dma_descriptor | 10 (100%) | 100% | 100% | no_proof_axis | low |
| persistent_command_buffer | 10 (100%) | 100% | 100% | no_proof_axis | low |
| prefetch_descriptor | 10 (100%) | 100% | 100% | no_proof_axis | low |
| accumulator_commit | 10 (100%) | 100% | 100% | no_proof_axis | low |
| accumulator_merge | 10 (100%) | 100% | 100% | no_proof_axis | low |
| fused_dequant_matmul | 10 (100%) | 100% | 100% | no_proof_axis | low |
| native_lowbit_matmul | 10 (100%) | 100% | 100% | no_proof_axis | low |
| packed_lowbit_tensor | 10 (100%) | 100% | 100% | unknown | low |
| resident_packed_weight_object | 10 (100%) | 100% | 100% | no_proof_axis | low |
| scale_object | 10 (100%) | 100% | 100% | unknown | low |
| skinny_gemm_or_gemv_engine | 10 (100%) | 100% | 100% | no_proof_axis | low |
| fused_requant_epilogue | 9 (90%) | 100% | 95% | no_proof_axis | medium |
| epilogue_requant_unit | 9 (90%) | 100% | 95% | no_proof_axis | medium |
| producer_consumer_queue | 8 (80%) | 100% | 95% | no_proof_axis | medium |
| double_buffered_action_chunk | 8 (80%) | 100% | 95% | no_proof_axis | medium |
| matrix_engine | 3 (30%) | 99% | 57% | no_proof_axis | medium |
| decode_loop_controller | 4 (40%) | 0% | 33% | no_proof_axis | medium |
| kv_cache_object | 4 (40%) | 0% | 33% | no_proof_axis | medium |
| prefix_kv_object | 4 (40%) | 0% | 33% | no_proof_axis | medium |

## Decision-impact plots (what changes if DSE picks differently)

Structural what-if curves (bytes / coverage / counts — never latency or speedup). PNGs under `generated_plots/`.

- **Decision: single primitive choice -> coverage** (`generated_plots/decision_primitive_choice.png`)  
  If DSE builds only ONE compute primitive, how much of each workload's MACs it can tile under 10% waste — the worst-case bar shows no single primitive covers every workload, so the search space needs both a tile and a GEMV lane.
- **Decision: weight residency -> bytes moved vs loop count** (`generated_plots/decision_weight_residency.png`)  
  Weight bytes moved as the head loop count grows: reload-every-step (linear) vs keep-resident (flat). The vertical gap at each workload's real K is the avoidable reload a residency knob removes (bytes, not bandwidth).
- **Decision: on-chip capacity + dtype -> weights resident** (`generated_plots/decision_capacity_dtype.png`)  
  How many workloads become fully weight-resident as the on-chip capacity budget grows, per storage dtype — low-bit dtypes reach full residency at a smaller budget, quantifying the capacity-vs-dtype trade in the search space.
- **Decision: shard axis + count -> extra data-movement bytes** (`generated_plots/decision_sharding_cost.png`)  
  Extra data-movement bytes added by sharding 2/4/8 ways along M, N, or K: M/N shards are reduction-free (broadcast only) while K shards add partial-sum traffic — the cost side of the parallelization decision.
- **Decision: shard top-MAC ops -> extra bytes / output bytes** (`generated_plots/decision_sharding_per_top_op.png`)  
  For the top-MAC ops, extra sharding bytes normalized by the op's output bytes, per M/N/K axis — which axis partitions a hot op cheaply (the per-operator view, not a corpus aggregate).

## What a DSE tool ingests — knob catalog

> The structural search-space dimensions the workload-contract analysis discovered, consolidated as the bridge a future DSE engine consumes (alongside the per-workload abstraction-axis `dse_search_space_template.yaml`). **Structural only — no speedup, no chosen design.**

| knob group | phase | enabled | knobs | gated by |
|---|---|---|---|---|
| compute_primitive_shape | P5 | True | tile_8x8, tile_8x16, tile_16x16, tile_16x32, tile_32x32, gemv_lane_64, gemv_lane_128, gemv_lane_256 | structural tile/lane coverage of the real operator geometry (no perf) |
| intra_op_sharding | P7 | True | shard_axis in {M,N,K}, shard_count in [2, 4, 8] | 2143 reduction-free M/N shards available; K-sharding needs partial-sum reduction |
| inter_op_parallelism | P7 | False | num_engines | avg inter-op parallelism 1.426x (low -> limited; favors intra-op sharding) |
| processing_unit_set | P7/P8 | True | dma_engine, epilogue_requant_unit, loop_controller, matrix_engine, scalar_control_unit, skinny_gemm_or_gemv_engine | distinct operator families (dense GEMM + skinny/GEMV) + epilogue + DMA |
| pipeline_overlap | P8 | True | async_queue, bounded_loop_command, double_buffered_action_chunk, event_token, loop_carried_state_handle, prefix_state_object, producer_consumer_queue, resident_weight_object | candidate overlaps gated on recovered structure (backbone compute / control loop); per-phase timing needed to schedule |
| memory_residency | P9 | True | resident_weight_object, weight_dtype in ['bf16', 'fp8', 'int8', 'int4', 'fp6'], prefetch_depth | weight-dominated memory pressure; bandwidth needs a design YAML |
| dma_streams | P9 | True | multi_stream_dma_descriptor, double_buffered_activation_tile, prefetch_weight_once | 3 byte-carrying streams/region (weight/activation/output) |
| epilogue_fusion | P10 | True | epilogue_op_set subset of ['bias', 'activation', 'scale', 'clamp', 'cast'], accumulator_dtype, requant_in_epilogue | directly-fused epilogue slot proven (addmm bias); low-bit/scale gated by a low-bit capture + accuracy |
| hw_sw_boundary_placement | P12 | True | resident_weight_object@{compiler/runtime/command/isa/microcode/datapath}, region_level_dispatch@{compiler/runtime/command/isa/microcode/datapath}, event_token@{compiler/runtime/command/isa/microcode/datapath}, async_queue@{compiler/runtime/command/isa/microcode/datapath}, dma_engine@{compiler/runtime/command/isa/microcode/datapath}, partial_sum_object@{compiler/runtime/command/isa/microcode/datapath} | boundary placement is a search-space axis (Merlin does not choose); see boundary_candidate_contracts.yaml + boundary_dse_knobs.yaml |

Each knob group is evidence-labeled and grounded in the per-phase artifacts (P5 geometry/coverage, P7 sharding/hierarchy, P8 pipeline, P9 memory/DMA, P10 epilogue). The measurements each knob needs before a *quantitative* DSE decision (per-unit throughput, bandwidth, accuracy for low-bit, per-phase timing) are named in those artifacts. **No speedup is claimed.**

## How to evaluate this yourself

- Every headline row traces through `unified_fact_table.csv` (`metric_name -> source_artifact -> verifying_check`).
- The numbers are recomputed independently by `merlin/benchmarks/dse_guidance/verify_implementation.py` (run it; exit 0 = all checks pass).
- Regenerate this whole folder with `merlin-dse-guidance --insight-mining` (add `--workload <name>` for one network).
