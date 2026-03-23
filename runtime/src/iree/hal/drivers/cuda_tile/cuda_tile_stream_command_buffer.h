// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_CUDA_TILE_STREAM_COMMAND_BUFFER_H_
#define IREE_HAL_DRIVERS_CUDA_TILE_STREAM_COMMAND_BUFFER_H_

#include "iree/base/internal/arena.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/cuda_tile/cuda_tile_dynamic_symbols.h"
#include "iree/hal/drivers/cuda_tile/cuda_tile_headers.h"
#include "iree/hal/utils/stream_tracing.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a command buffer that immediately issues commands against the given
// CUDA |stream|. No NCCL support (removed from cuda_tile driver).
iree_status_t iree_hal_cuda_tile_stream_command_buffer_create(
    iree_hal_allocator_t* device_allocator,
    const iree_hal_cuda_tile_dynamic_symbols_t* cuda_symbols,
    iree_hal_stream_tracing_context_t* tracing_context,
    iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_host_size_t binding_capacity, CUstream stream,
    iree_arena_block_pool_t* block_pool, iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_command_buffer);

// Returns true if |command_buffer| is a CUDA stream-based command buffer.
bool iree_hal_cuda_tile_stream_command_buffer_isa(
    iree_hal_command_buffer_t* command_buffer);

// Notify tracing system of submitted commands.
void iree_hal_cuda_tile_stream_notify_submitted_commands(
    iree_hal_command_buffer_t* base_command_buffer);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_CUDA_TILE_STREAM_COMMAND_BUFFER_H_
