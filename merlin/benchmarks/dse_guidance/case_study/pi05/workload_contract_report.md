# Workload contract analysis — pi05

> Merlin recovers the HW/SW contract a flat capture erases and hands a future DSE engine the abstractions the workload needs, the requirements any design must meet, and what is still missing. It does **not** pick a design or claim a speedup.

- class: **flow_matching_action_head**  ·  K=10, H=50, control_rate=50.0 Hz

## 1. Recovered structure

- repeated head (structural_scf_for): 167 matmuls, 1720 MB weights, 15.7 GMAC/step, reused x10
- backbone (once/replan): 610 matmuls

## 2. Numerical contract

- storage **f32**, compute **f32**; lost: none (severity low)

## 3. Requirements (hardware-independent)

- macs_per_replan = 1.569e+11 MAC (recovered_from_ir)
- resident_capacity_required = 1.720e+09 B (recovered_from_ir)
- avoidable_weight_reload_bytes = 1.548e+10 B (recovered_from_ir)
- required_compute_rate = 1.569e+11 MAC/s (derived_requirement)
- required_weight_bandwidth = 1.720e+10 B/s (derived_requirement)
- required_command_rate = 1.670e+03 dispatch/s (derived_requirement)
- resident capacity by format: bf16=860MB, fp8=430MB, int8=430MB, int4=215MB, fp6=322MB

## 4. HW/SW abstraction candidates

| abstraction | DSE knobs | blocked by |
|-------------|-----------|------------|
| resident_weight_object | local_memory_capacity, packed_weight_format, dma_bandwidth, residency/replacement_policy, weight_prefetch_overlap | missing_calibration |
| bounded_loop_command + loop_carried_state_handle | device_loop_controller, dependency_tracking, resident_state_handles | missing_calibration |
| phase_boundary / region_handle (slow/fast split) | engine_partition, time_share_vs_split, state_crossing_transport | missing_calibration |
| double_buffered_replan_state + async_queue | double_buffering, async_scheduling, deadline_slack_use | missing_calibration |
| command_buffer | submission_granularity, persistent_command_graph | missing_calibration |
| packed_weight_cache / layout_persistent_buffer | packed_layout_object, repack_avoidance | missing_calibration |
| packed_lowbit_tensor + resident_weight_object + scale_object | resident_capacity_at_format, packed_layout_support, scale_object, low_bit_matmul_datapath | missing accuracy sweep + low-bit kernel calibration + resident-capacity model |
| accumulator_object + requant_epilogue | in_hw_epilogue_commit, accumulator_visibility | missing intermediate-materialization measurement |

_None claim a speedup/accuracy number — see `abstraction_candidates.yaml`._

## 5. Measurement plan

- measurable now (accuracy): fused requant/activation bit-exactness, per-format accuracy vs fp32 (cos/argmax) sweep
- measurable now (runtime proxy): batched_submit_ns, command_encode_ns, dispatches_per_replan, epilogue dispatch count, host_submit_ns, per_step_host_dispatch_ns, per_step_sync_ns
- needs target design: K, action_head_weight_bytes, backbone_cost, chunk_execution_time, deadline_slack, dequant/pack cost, head_cost_per_step, i32 intermediate bytes, loop_carried_state_bytes, low-bit weight bytes, measured memory bandwidth, pack_bytes, pack_cost_per_step, pack_count_per_replan, repacks_avoided, replan_latency, resident capacity at each format, resident_capacity_required, state_crossing_bytes, weight_reload_bytes_per_step

## 6. DSE readiness

- ready to rank designs: **False**
  - missing: quantization accuracy gates (per candidate low-bit format)
  - missing: real (target) command-submit / sync latency
  - missing: K / control-rate from the real deployment (currently reference values)
