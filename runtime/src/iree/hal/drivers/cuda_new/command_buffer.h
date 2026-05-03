// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_CUDA_NEW_COMMAND_BUFFER_H_
#define IREE_HAL_DRIVERS_CUDA_NEW_COMMAND_BUFFER_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/hal/api.h"
#include "dynamic_symbols.h"
#include "headers.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

iree_status_t iree_hal_cuda_new_stream_command_buffer_create(
	iree_hal_allocator_t *device_allocator,
	const iree_hal_cuda_new_dynamic_symbols_t *syms,
	CUcontext cu_context,
	iree_hal_command_buffer_mode_t mode,
	iree_hal_command_category_t command_categories,
	iree_host_size_t binding_capacity, CUstream stream,
	iree_arena_block_pool_t *block_pool, iree_allocator_t host_allocator,
	iree_hal_command_buffer_t **out_command_buffer);

bool iree_hal_cuda_new_stream_command_buffer_isa(
	iree_hal_command_buffer_t *command_buffer);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // IREE_HAL_DRIVERS_CUDA_NEW_COMMAND_BUFFER_H_
