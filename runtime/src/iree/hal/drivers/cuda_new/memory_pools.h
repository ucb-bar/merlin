// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_CUDA_NEW_MEMORY_POOLS_H_
#define IREE_HAL_DRIVERS_CUDA_NEW_MEMORY_POOLS_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "dynamic_symbols.h"
#include "headers.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef struct iree_hal_cuda_new_memory_pools_t {
	CUmemoryPool device_local;
	CUmemoryPool other;
	iree_hal_device_t *parent_device;
	const iree_hal_cuda_new_dynamic_symbols_t *syms;
	iree_allocator_t host_allocator;
} iree_hal_cuda_new_memory_pools_t;

iree_status_t iree_hal_cuda_new_memory_pools_initialize(
	iree_hal_device_t *parent_device,
	const iree_hal_cuda_new_dynamic_symbols_t *syms,
	CUdevice cu_device, iree_allocator_t host_allocator,
	iree_hal_cuda_new_memory_pools_t *out_pools);

void iree_hal_cuda_new_memory_pools_deinitialize(
	iree_hal_cuda_new_memory_pools_t *pools);

iree_status_t iree_hal_cuda_new_memory_pools_trim(
	iree_hal_cuda_new_memory_pools_t *pools);

iree_status_t iree_hal_cuda_new_memory_pools_alloca(
	iree_hal_cuda_new_memory_pools_t *pools, CUstream stream,
	iree_hal_allocator_pool_t pool, iree_hal_buffer_params_t params,
	iree_device_size_t allocation_size, iree_hal_alloca_flags_t flags,
	iree_hal_buffer_t **out_buffer);

iree_status_t iree_hal_cuda_new_memory_pools_dealloca(
	iree_hal_cuda_new_memory_pools_t *pools, CUstream stream,
	iree_hal_buffer_t *buffer, iree_hal_dealloca_flags_t flags);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // IREE_HAL_DRIVERS_CUDA_NEW_MEMORY_POOLS_H_
