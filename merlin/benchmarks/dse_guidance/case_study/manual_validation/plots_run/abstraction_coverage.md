# Abstraction coverage (all)

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
