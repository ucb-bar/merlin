# Interface contract sketches (HAL / command / ISA)

> Possible high-level interface sketches the workload evidence suggests. **These are sketches, not a final ISA/HAL design**, and make no speedup/area claim. The DSE tool would refine and choose.

## Runtime / HAL object sketch — `resident_weight_object`

- fields: `dtype`, `shape`, `layout`, `lifetime`, `size_bytes`, `scale_object_handle` (if quantized)
- operations: `load`, `pin`, `reuse`, `evict`
- evidence: weights are loop-invariant across the K-loop (`resident_action_head_weights` = proven_for_workload)

## Command ISA sketch — `bounded_loop_command`

- fields: `trip_count`, `body_region_handle`, `loop_carried_state_handles`, `invariant_state_handles`, `event_in`, `event_out`
- evidence: the repeated head is a bounded K-loop with loop-invariant weights (`autonomous_K_loop` = assumed)

## Accelerator ISA primitive sketch — `matmul_packed_lowbit`

- fields: `lhs_dtype`, `rhs_storage_dtype`, `accumulator_dtype`, `scale_object`, `output_dtype`, `tile_shape`, `epilogue_mode`
- evidence: **blocked** — the capture is dequantized f32; this primitive needs a low-bit capture (packed layout + scales) + per-format accuracy before it is a candidate (`resident_packed_lowbit_weights` = unknown)

## Accelerator ISA primitive sketch — `gemv_dot_lanes`

- fields: `lane_width`, `num_lanes`, `accumulator_depth`, `reduction_tree_width`
- evidence: GEMV/skinny shapes dominate the decode workloads (P5 geometry); a square matrix engine covers them poorly (P5 regret)

These sketches correspond to the `runtime_object_candidates.yaml`, `command_isa_candidates.yaml`, and `isa_candidate_primitives.yaml` lists.
