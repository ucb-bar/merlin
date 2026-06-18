# HW/SW boundary-placement report

> The boundary search space the workload evidence implies: for each abstraction, where it could sit and what each placement requires. **Merlin generates the options; the DSE tool chooses. No speedup / cycles / area / energy and no chosen design is claimed.** `boundary_pressure_score` is evidence breadth, not performance.

## Strongly-suggested boundary placements (by evidence breadth)

| abstraction | top level(s) | supporting workloads | pressure (evidence) |
|---|---|---|---|
| resident_weight_object | runtime_hal_object, command_buffer_or_command_isa | groot_n1d7, molmoact, openvla, rdt, rdt2, small_llama, tiny_llama | 12 |
| region_level_dispatch | command_buffer_or_command_isa, device_microcode_or_controller | groot_n1d7, molmoact, openvla, rdt, rdt2, small_llama, tiny_llama | 11 |
| event_token | runtime_hal_object, command_buffer_or_command_isa | groot_n1d7, molmoact, openvla, rdt, rdt2, small_llama, tiny_llama | 11 |
| async_queue | runtime_hal_object, command_buffer_or_command_isa | groot_n1d7, molmoact, openvla, rdt, rdt2, small_llama, tiny_llama | 11 |
| dma_engine | runtime_hal_object, command_buffer_or_command_isa | groot_n1d7, molmoact, openvla, rdt, rdt2, small_llama, tiny_llama | 11 |
| partial_sum_object | accelerator_isa, device_microcode_or_controller, fixed_hardware_datapath | groot_n1d7, molmoact, openvla, rdt, rdt2, small_llama, tiny_llama | 10 |
| loop_carried_state_handle | command_buffer_or_command_isa, device_microcode_or_controller | groot_n1d7, molmoact, openvla, rdt, rdt2, small_llama, tiny_llama | 9 |
| bounded_loop_command | command_buffer_or_command_isa, device_microcode_or_controller | groot_n1d7, molmoact, openvla, rdt, rdt2, small_llama, tiny_llama | 9 |
| persistent_command_buffer | command_buffer_or_command_isa, device_microcode_or_controller | groot_n1d7, molmoact, openvla, rdt, rdt2, small_llama, tiny_llama | 9 |
| producer_consumer_queue | runtime_hal_object, command_buffer_or_command_isa | groot_n1d7, molmoact, openvla, rdt, rdt2 | 9 |

## Where all software/hardware placements are plausible (the genuine design axes)

`resident_weight_object`, `region_level_dispatch`, `dma_engine`, `prefetch_descriptor`

## Software-only management may explode command count

- `bounded_loop_command` / `region_level_dispatch`: a pure host loop submits a command per step (K×matmuls dispatches); a command buffer or device controller would remove the host re-dispatch. **requires command/ISA semantics.**
- `multi_stream_dma_descriptor`: software-issued per-tile DMA explodes; a descriptor engine batches it.

## Hardware-only management may hide semantics the compiler knows

- `resident_weight_object`: a hardware cache rediscovers reuse the compiler already proved (loop-invariant weights) — a `resident_weight_object` keeps the lifetime explicit.
- `partial_sum_object` / `accumulator_merge`: hardware-internal K-sharding hides the reduction; a command/ISA-level shard keeps it visible to the compiler.

## ISA / HAL objects the evidence suggests

- runtime objects: see `runtime_object_candidates.yaml`; command ops: `command_isa_candidates.yaml`; ISA primitives: `isa_candidate_primitives.yaml`; sketches: `interface_contract_sketches.md`.

## Blocked / unavailable (honest)

- `resident_packed_weight_object`, `packed_lowbit_tensor`, `scale_object`, `prefix_kv_object`, `fused_dequant_matmul`, `native_lowbit_matmul`, `kv_cache_object` — packed low-bit / scale / KV structure is erased or lowered in the capture; these placements are `blocked`/`unavailable` until a low-bit (packed weights + scales) or loop-preserving capture exists.

## Missing measurements before choosing a boundary

- per-unit throughput / latency / area / energy (a design YAML), per-format low-bit accuracy, per-phase timing, and host command/sync latency — named per certificate. Merlin does not choose; it bounds the search space.
