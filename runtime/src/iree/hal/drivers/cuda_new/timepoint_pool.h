// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_CUDA_NEW_TIMEPOINT_POOL_H_
#define IREE_HAL_DRIVERS_CUDA_NEW_TIMEPOINT_POOL_H_

#include "iree/async/semaphore.h"
#include "iree/base/api.h"
#include "event_pool.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef struct iree_hal_cuda_new_timepoint_pool_t
	iree_hal_cuda_new_timepoint_pool_t;

typedef enum iree_hal_cuda_new_timepoint_kind_e {
	IREE_HAL_CUDA_NEW_TIMEPOINT_KIND_NONE = 0,
	IREE_HAL_CUDA_NEW_TIMEPOINT_KIND_DEVICE_SIGNAL,
	IREE_HAL_CUDA_NEW_TIMEPOINT_KIND_DEVICE_WAIT,
} iree_hal_cuda_new_timepoint_kind_t;

typedef struct iree_hal_cuda_new_timepoint_t {
	iree_async_semaphore_timepoint_t base;
	iree_allocator_t host_allocator;
	iree_hal_cuda_new_timepoint_pool_t *pool;
	iree_hal_cuda_new_timepoint_kind_t kind;
	iree_hal_cuda_new_event_t *event;
} iree_hal_cuda_new_timepoint_t;

iree_status_t iree_hal_cuda_new_timepoint_pool_allocate(
	iree_hal_cuda_new_event_pool_t *event_pool,
	iree_host_size_t available_capacity, iree_allocator_t host_allocator,
	iree_hal_cuda_new_timepoint_pool_t **out_timepoint_pool);

void iree_hal_cuda_new_timepoint_pool_free(
	iree_hal_cuda_new_timepoint_pool_t *timepoint_pool);

iree_status_t iree_hal_cuda_new_timepoint_pool_acquire(
	iree_hal_cuda_new_timepoint_pool_t *timepoint_pool,
	iree_hal_cuda_new_timepoint_kind_t kind,
	iree_hal_cuda_new_timepoint_t **out_timepoint);

void iree_hal_cuda_new_timepoint_pool_release(
	iree_hal_cuda_new_timepoint_pool_t *timepoint_pool,
	iree_hal_cuda_new_timepoint_t *timepoint);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // IREE_HAL_DRIVERS_CUDA_NEW_TIMEPOINT_POOL_H_
