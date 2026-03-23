# cuda_tile HAL Driver — Future Features

Features intentionally skipped in the initial fork. Each can be added by
forking the corresponding file from the upstream CUDA HAL and adding it to
`CMakeLists.txt`.

## CUDA Graph Command Buffers

**Source**: `graph_command_buffer.c/h` (871 lines)

Captures command buffer operations into a CUDA graph for replay. Reduces launch
latency when the same sequence of kernels is dispatched repeatedly. No benefit
until cuda_tile supports multi-kernel fusion pipelines.

**To add**:
1. Fork `graph_command_buffer.c/h` → `cuda_tile_graph_command_buffer.c/h`
2. Add to `CMakeLists.txt` SRCS
3. In `cuda_tile_device.c`, add `command_buffer_mode` param and graph path
   in `create_command_buffer`

## NCCL Multi-GPU Collectives

**Source**: `nccl_channel.c/h` + `nccl_dynamic_symbols.c/h` (~800 lines)

Multi-GPU collective operations (allreduce, broadcast, etc.) via NCCL.
Single-GPU only for now.

**To add**:
1. Fork all NCCL files → `cuda_tile_nccl_*.c/h`
2. Add `nccl::headers` dep to `dynamic_symbols` library
3. Re-add `nccl_symbols` field to driver and device structs
4. Implement `create_channel` in device vtable

## Async Memory Pools

**Source**: `memory_pools.c/h` (315 lines)

`CUmemoryPool`-based async allocation for reduced allocation latency.
Simple `cuMemAlloc` path works; pools are a latency optimization.

**To add**:
1. Fork `memory_pools.c/h` → full `cuda_tile_memory_pools.c/h`
   (replacing the current stub header)
2. Add to `CMakeLists.txt` SRCS
3. Re-add `async_allocations` param and pool init in device creation
4. Update `queue_alloca`/`queue_dealloca` to use pool path
