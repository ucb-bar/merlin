# Copyright 2026 UCB-BAR
#
# cuda_tile HAL driver — standalone device for cuda_tile IR executables.
# Forked from IREE's CUDA HAL driver with cuda_tile dispatch semantics
# (block_dims={1,1,1}, CTL1 FlatBuffer format).

set(IREE_PACKAGE_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(IREE_PACKAGE_ROOT_PREFIX "iree/hal/drivers/cuda_tile")

# The cuda_tile driver uses IREE-style full include paths
# (e.g. "iree/hal/drivers/cuda_tile/...") which require the merlin
# runtime/src directory in the include search path.
get_filename_component(_MERLIN_RUNTIME_SRC
  "${CMAKE_CURRENT_LIST_DIR}/../../../.." ABSOLUTE)
include_directories("${_MERLIN_RUNTIME_SRC}")

iree_add_all_subdirs()

iree_cc_library(
  NAME
    dynamic_symbols
  HDRS
    "cuda_tile_dynamic_symbols.h"
    "cuda_tile_status_util.h"
    "cuda_tile_nccl_dynamic_symbols.h"
    "cuda_tile_nccl_status_util.h"
  TEXTUAL_HDRS
    "cuda_tile_dynamic_symbol_table.h"
    "cuda_tile_nccl_dynamic_symbol_table.h"
  SRCS
    "cuda_tile_dynamic_symbols.c"
    "cuda_tile_headers.h"
    "cuda_tile_status_util.c"
    "cuda_tile_nccl_dynamic_symbols.c"
    "cuda_tile_nccl_headers.h"
    "cuda_tile_nccl_status_util.c"
  DEPS
    iree::base
    iree::base::internal::dynamic_library
    iree_cuda::headers
    nccl::headers
  PUBLIC
)

iree_cc_library(
  NAME
    cuda_tile
  HDRS
    "api.h"
  SRCS
    "api.h"
    "cuda_tile_allocator.c"
    "cuda_tile_allocator.h"
    "cuda_tile_buffer.c"
    "cuda_tile_buffer.h"
    "cuda_tile_device.c"
    "cuda_tile_device.h"
    "cuda_tile_driver.c"
    "cuda_tile_event_pool.c"
    "cuda_tile_event_pool.h"
    "cuda_tile_event_semaphore.c"
    "cuda_tile_event_semaphore.h"
    "cuda_tile_graph_command_buffer.c"
    "cuda_tile_graph_command_buffer.h"
    "cuda_tile_memory_pools.c"
    "cuda_tile_memory_pools.h"
    "cuda_tile_nccl_channel.c"
    "cuda_tile_nccl_channel.h"
    "cuda_tile_nop_executable_cache.c"
    "cuda_tile_nop_executable_cache.h"
    "cuda_tile_stream_command_buffer.c"
    "cuda_tile_stream_command_buffer.h"
    "cuda_tile_timepoint_pool.c"
    "cuda_tile_timepoint_pool.h"
    "native_executable.c"
    "native_executable.h"
  DEPS
    ::dynamic_symbols
    iree::base
    iree::base::core_headers
    iree::base::internal
    iree::base::internal::arena
    iree::base::internal::atomic_slist
    iree::base::internal::event_pool
    iree::base::internal::flatcc::parsing
    iree::base::internal::synchronization
    iree::base::internal::threading
    iree::base::internal::wait_handle
    iree::hal
    iree::hal::utils::collective_batch
    iree::hal::utils::deferred_command_buffer
    iree::hal::utils::deferred_work_queue
    iree::hal::utils::executable_debug_info
    iree::hal::utils::executable_header
    iree::hal::utils::file_transfer
    iree::hal::utils::files
    iree::hal::utils::queue_emulation
    iree::hal::utils::queue_host_call_emulation
    iree::hal::utils::resource_set
    iree::hal::utils::semaphore_base
    iree::hal::utils::stream_tracing
    iree::schemas::cuda_tile_executable_def_c_fbs
    iree::schemas::executable_debug_info_c_fbs
  PUBLIC
)
