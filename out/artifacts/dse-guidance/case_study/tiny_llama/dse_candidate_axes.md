# Structural DSE candidate axes — tiny_llama

- workload class: **autoregressive_decode**  ·  K=7, H=7, control_rate=30.0 Hz

> These are **structural** candidates derived from the recovered workload contract. They are valid without calibration. Each lists what must be proven/built and **what to measure before any quantitative ranking** — no cycle numbers are claimed here.

## resident_action_head_weights
- **signal**: temporal_reuse — action-head weights are loop-invariant across K=7 steps; a flat capture sees one use and hides this axis
- **evidence**: `{'K': 7, 'loop_invariant_state': ['weights'], 'head_phases': ['head']}`
- **needs (compiler)**: action-head weights immutable and live across the K-loop (resident-pack candidate); backbone weights excluded
- **needs (hw/runtime)**: resident weight store + pack-once/use-many interface
- **measure first**: action_head_weight_bytes, weight_reload_bytes_per_step, pack_cost_per_step, resident_capacity_required, measured memory bandwidth
- **attributed IR facts (Level-1)**: `{'matmul_count': 15, 'macs_per_invocation': 153616384, 'weight_bytes': 614465536, 'activation_bytes_per_invocation': 422912, 'invocations': 7, 'macs_total': 1075314688, 'activation_bytes_total': 2960384}`
- **status**: legality=structural, benefit=unquantified (blocked by: missing_calibration)
- **could be wrong if**: weights exceed resident capacity; packing already hoisted out of the loop; DMA/weight traffic is not the bottleneck; host dispatch dominates total latency

## autonomous_K_loop
- **signal**: temporal_reuse — a bounded K=7 loop with loop-carried state can run device-side
- **evidence**: `{'K': 7, 'loop_carried_state': []}`
- **needs (compiler)**: bounded K and a loop body expressible as a device-resident program
- **needs (hw/runtime)**: device-side loop controller + dependency tracking
- **measure first**: K, per_step_host_dispatch_ns, per_step_sync_ns, loop_carried_state_bytes
- **status**: legality=structural, benefit=unquantified (blocked by: missing_calibration)
- **could be wrong if**: K is data-dependent / unbounded; per-step host overhead is already negligible; on-device control costs area not justified by the saving

## backbone_head_partition
- **signal**: rate_mismatch — backbone runs once per replan while the head runs K times — the rates differ, so a slow/fast partition is a candidate
- **evidence**: `{'backbone_phases': ['backbone'], 'head_phases': ['head'], 'K': 7, 'H': 7, 'control_rate_hz': 30.0, 'replan_deadline_ms': 233.33333333333334}`
- **needs (compiler)**: a clean state boundary between the once-per-replan backbone and the K-times head (the crossing tensors are enumerable)
- **needs (hw/runtime)**: slow-path/fast-path engine split or time-shared scheduling
- **measure first**: backbone_cost, head_cost_per_step, state_crossing_bytes, deadline_slack
- **status**: legality=structural, benefit=unquantified (blocked by: missing_calibration)
- **could be wrong if**: backbone and head costs are comparable (no clear slow/fast split); state crossing is too large to move; a single engine already meets the deadline

## async_chunk_overlap
- **signal**: rate_mismatch — the robot executes H actions at the control rate, opening a window to overlap the next replan with chunk execution
- **evidence**: `{'H': 7, 'control_rate_hz': 30.0, 'replan_deadline_ms': 233.33333333333334}`
- **needs (compiler)**: the next replan is independent of the current chunk's execution
- **needs (hw/runtime)**: double-buffered action chunks + async backbone/head scheduling
- **measure first**: replan_latency, chunk_execution_time, deadline_slack
- **status**: legality=structural, benefit=unquantified (blocked by: missing_calibration)
- **could be wrong if**: the next replan depends on executed-chunk feedback; there is no deadline slack to exploit; double buffering exceeds memory

## command_batching
- **signal**: cpu_coupling — the K=7 head steps issue repeated host submits a flat capture hides
- **evidence**: `{'K': 7, 'note': 'K per-step submits are collapsible to one buffer'}`
- **needs (compiler)**: the K-loop dependency graph is static and known at submit time
- **needs (hw/runtime)**: persistent/batched command buffer submit
- **measure first**: dispatches_per_replan, host_submit_ns, command_encode_ns, batched_submit_ns
- **status**: legality=structural, benefit=unquantified (blocked by: missing_calibration)
- **could be wrong if**: host dispatch is not on the critical path; batching raises per-submit latency; the runtime already coalesces submits

## packed_layout_preservation
- **signal**: layout_packing — the same (packed/quantized) weight layout is consumed every step; preserving it across dispatches avoids re-packing
- **evidence**: `{'K': 7, 'quantized': False}`
- **needs (compiler)**: a packed/quantized weight layout is produced once and consumed by the same op family across the loop without re-pack
- **needs (hw/runtime)**: packed layout as a first-class, dispatch-crossing object
- **measure first**: pack_count_per_replan, pack_bytes, repacks_avoided
- **attributed IR facts (Level-1)**: `{'matmul_count': 15, 'macs_per_invocation': 153616384, 'weight_bytes': 614465536, 'activation_bytes_per_invocation': 422912, 'invocations': 7, 'macs_total': 1075314688, 'activation_bytes_total': 2960384}`
- **status**: legality=structural, benefit=unquantified (blocked by: missing_calibration)
- **could be wrong if**: packing is not repeated; the op family changes layout between uses; dequant happens upstream regardless

## decode_kv_cache_path
- **signal**: dynamic_loop — autoregressive decode with a growing KV cache and batch=1 GEMV shapes wants a decode/KV-cache-optimized path, not GEMM throughput
- **evidence**: `{'workload_class': 'autoregressive_decode', 'K': 7}`
- **needs (compiler)**: autoregressive decode with a growing KV cache and batch=1 GEMV shapes
- **needs (hw/runtime)**: decode/GEMV-optimized datapath + resident KV cache object
- **measure first**: tokens_decoded, kv_growth_bytes, gemv_shape_distribution
- **status**: legality=structural, benefit=unquantified (blocked by: missing_calibration)
- **could be wrong if**: the head is not autoregressive; throughput-GEMM utilisation is already adequate; KV fits trivially
