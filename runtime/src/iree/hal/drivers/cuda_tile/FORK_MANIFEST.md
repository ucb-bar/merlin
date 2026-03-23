# CUDA Tile HAL Driver — Fork Manifest

Forked from IREE CUDA HAL driver.

- **Source**: `third_party/iree_bar/runtime/src/iree/hal/drivers/cuda/`
- **Source commit**: `2b7dd40`
- **Fork date**: 2026-03-22
- **Fork script**: `tools/fork_cuda_hal.sh`

## Forked Files

| Source (cuda/) | Target (cuda_tile/) |
|---|---|
| `stream_command_buffer.c` | `cuda_tile_stream_command_buffer.c` |
| `timepoint_pool.h` | `cuda_tile_timepoint_pool.h` |
| `timepoint_pool.c` | `cuda_tile_timepoint_pool.c` |
| `stream_command_buffer.h` | `cuda_tile_stream_command_buffer.h` |
| `registration/driver_module.h` | `registration/driver_module.h` |
| `registration/driver_module.c` | `registration/driver_module.c` |
| `event_semaphore.c` | `cuda_tile_event_semaphore.c` |
| `event_semaphore.h` | `cuda_tile_event_semaphore.h` |
| `cuda_dynamic_symbol_table.h` | `cuda_tile_dynamic_symbol_table.h` |
| `cuda_status_util.h` | `cuda_tile_status_util.h` |
| `cuda_status_util.c` | `cuda_tile_status_util.c` |
| `cuda_buffer.c` | `cuda_tile_buffer.c` |
| `cuda_buffer.h` | `cuda_tile_buffer.h` |
| `cuda_dynamic_symbols.h` | `cuda_tile_dynamic_symbols.h` |
| `cuda_dynamic_symbols.c` | `cuda_tile_dynamic_symbols.c` |
| `cuda_headers.h` | `cuda_tile_headers.h` |
| `nop_executable_cache.c` | `cuda_tile_nop_executable_cache.c` |
| `cuda_device.h` | `cuda_tile_device.h` |
| `cuda_device.c` | `cuda_tile_device.c` |
| `nop_executable_cache.h` | `cuda_tile_nop_executable_cache.h` |
| `event_pool.c` | `cuda_tile_event_pool.c` |
| `event_pool.h` | `cuda_tile_event_pool.h` |
| `api.h` | `api.h` |
| `cuda_driver.c` | `cuda_tile_driver.c` |
| `cuda_allocator.h` | `cuda_tile_allocator.h` |
| `cuda_allocator.c` | `cuda_tile_allocator.c` |

## Custom Files (not forked)

| File | Description |
|---|---|
| `native_executable.c` | CTL1 FlatBuffer reader (custom) |
| `native_executable.h` | CTL1 executable API (custom) |
| `CMakeLists.txt` | Build configuration (custom) |

## Skipped Files (see FUTURE.md)

- `graph_command_buffer.c/h` — CUDA graph capture
- `nccl_channel.c/h` + `nccl_dynamic_symbols.c/h` — Multi-GPU collectives
- `memory_pools.c/h` — Async memory pool allocation
