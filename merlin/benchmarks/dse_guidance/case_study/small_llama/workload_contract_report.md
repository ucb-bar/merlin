# Workload contract analysis — small_llama

> Merlin recovers the HW/SW contract a flat capture erases and hands a future DSE engine the abstractions the workload needs, the requirements any design must meet, and what is still missing. It does **not** pick a design or claim a speedup.

- class: **autoregressive_decode**  ·  K=32, H=32, control_rate=30.0 Hz

## 1. Recovered structure

- repeated head (prov_fqn): 15 matmuls, 2 MB weights, 0.0 GMAC/step, reused x32

## 2. Numerical contract

- storage **f32**, compute **f32**; lost: none (severity low)

## 3. Requirements (hardware-independent)

- macs_per_replan = 1.096e+08 MAC (recovered_from_ir)
- resident_capacity_required = 1.712e+06 B (recovered_from_ir)
- avoidable_weight_reload_bytes = 5.308e+07 B (recovered_from_ir)
- required_compute_rate = 1.027e+08 MAC/s (derived_requirement)
- required_weight_bandwidth = 5.136e+07 B/s (derived_requirement)
- required_command_rate = 4.500e+02 dispatch/s (derived_requirement)
- resident capacity by format: bf16=1MB, fp8=0MB, int8=0MB, int4=0MB, fp6=0MB

## 4. HW/SW abstraction candidates

| abstraction | DSE knobs | blocked by |
|-------------|-----------|------------|
| resident_weight_object | local_memory_capacity, packed_weight_format, dma_bandwidth, residency/replacement_policy, weight_prefetch_overlap | missing_calibration |
| bounded_loop_command + loop_carried_state_handle | device_loop_controller, dependency_tracking, resident_state_handles | missing_calibration |
| phase_boundary / region_handle (slow/fast split) | engine_partition, time_share_vs_split, state_crossing_transport | missing_calibration |
| double_buffered_replan_state + async_queue | double_buffering, async_scheduling, deadline_slack_use | missing_calibration |
| command_buffer | submission_granularity, persistent_command_graph | missing_calibration |
| packed_weight_cache / layout_persistent_buffer | packed_layout_object, repack_avoidance | missing_calibration |
| decode_kv_cache_path | — | missing_calibration |
| packed_lowbit_tensor + resident_weight_object + scale_object | resident_capacity_at_format, packed_layout_support, scale_object, low_bit_matmul_datapath | missing accuracy sweep + low-bit kernel calibration + resident-capacity model |
| accumulator_object + requant_epilogue | in_hw_epilogue_commit, accumulator_visibility | missing intermediate-materialization measurement |
| resident_KV_cache (quantized) | kv_precision, kv_capacity, decode_gemv_datapath | missing KV-size profile + KV-quant accuracy sweep |

_None claim a speedup/accuracy number — see `abstraction_candidates.yaml`._

## 5. Measurement plan

- measurable now (accuracy): KV-quant accuracy vs fp16 KV, fused requant/activation bit-exactness, per-format accuracy vs fp32 (cos/argmax) sweep
- measurable now (runtime proxy): batched_submit_ns, command_encode_ns, dispatches_per_replan, epilogue dispatch count, host_submit_ns, per_step_host_dispatch_ns, per_step_sync_ns
- needs target design: K, KV bytes per step, KV growth over decode, action_head_weight_bytes, backbone_cost, chunk_execution_time, deadline_slack, dequant/pack cost, gemv_shape_distribution, head_cost_per_step, i32 intermediate bytes, kv_growth_bytes, loop_carried_state_bytes, low-bit weight bytes, measured memory bandwidth, pack_bytes, pack_cost_per_step, pack_count_per_replan, repacks_avoided, replan_latency, resident capacity at each format, resident_capacity_required, state_crossing_bytes, tokens_decoded, weight_reload_bytes_per_step

## 6. DSE readiness

- ready to rank designs: **False**
  - missing: real (target) command-submit / sync latency
  - missing: K / control-rate from the real deployment (currently reference values)
