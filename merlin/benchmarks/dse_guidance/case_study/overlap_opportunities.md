# Pipeline overlap opportunities

> Which phases run at different cadences and could structurally overlap. **Structural candidates only** — `can_overlap: yes` means *structurally permitted by the recovered dependencies and rates*, not scheduled and not a speedup.

## Multi-rate structure

| workload | phases | cadences |
|---|---|---|
| rdt | 2 | K_times_per_replan, once_per_replan |
| openvla | 2 | once_per_replan, token_loop |
| small_llama | 2 | once_per_replan, token_loop |
| tiny_llama | 2 | once_per_replan, token_loop |

## Candidate overlaps (common across workloads)

| candidate | can_overlap | required abstractions | buffers |
|---|---|---|---|
| backbone(next replan) ‖ action_execution(current chunk) | yes | double_buffered_action_chunk, async_queue, event_token, prefix_state_object | 2 |
| dma_prefetch(resident weights) ‖ head | yes | resident_weight_object, async_queue | 2 |
| head ‖ head | yes | bounded_loop_command, loop_carried_state_handle, resident_weight_object | 1 |
| control_tick_consumer ‖ replan_inference(next) | yes | double_buffered_action_chunk, producer_consumer_queue, event_token | 2 |
| decode_token_step ‖ kv_cache_movement | unknown | producer_consumer_queue, loop_carried_state_handle | unavailable |

## Abstractions repeatedly required for overlap

- `double_buffered_action_chunk` — required by 8 candidate overlaps
- `async_queue` — required by 8 candidate overlaps
- `event_token` — required by 8 candidate overlaps
- `resident_weight_object` — required by 8 candidate overlaps
- `prefix_state_object` — required by 4 candidate overlaps
- `bounded_loop_command` — required by 4 candidate overlaps
- `loop_carried_state_handle` — required by 4 candidate overlaps
- `producer_consumer_queue` — required by 4 candidate overlaps

## Findings

- **Backbone/head and control/inference decouple by cadence** — the once-per-replan backbone and the K-times head run at different rates, and the control loop consumes actions at yet another rate; these are candidate overlaps a future DSE should consider.
- **The K-loop is representable as a bounded device-side loop** (loop-invariant weights + bounded trip count) — `requires bounded_loop_command`.
- **`double_buffered_action_chunk` / `async_queue` recur** as the abstractions overlap needs — `requires event/queue abstraction`.
- **Blocked by missing timing/dependency evidence:** per-phase wall-clock periods, host dispatch/sync latency, DRAM bandwidth, and (for KV pipelining) attention structure are all `unavailable` — they block quantitative scheduling, not the structural overlap candidates.

**Caveat (structural, not realized):** these are candidate overlaps the structure permits. **No speedup**, schedule, or deadline-met claim is made.
