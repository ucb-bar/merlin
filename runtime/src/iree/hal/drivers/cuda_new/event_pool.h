// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_CUDA_NEW_EVENT_POOL_H_
#define IREE_HAL_DRIVERS_CUDA_NEW_EVENT_POOL_H_

#include "iree/base/api.h"
#include "dynamic_symbols.h"
#include "headers.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef struct iree_hal_cuda_new_event_t iree_hal_cuda_new_event_t;
typedef struct iree_hal_cuda_new_event_pool_t iree_hal_cuda_new_event_pool_t;

CUevent iree_hal_cuda_new_event_handle(
	const iree_hal_cuda_new_event_t *event);

void iree_hal_cuda_new_event_retain(iree_hal_cuda_new_event_t *event);

void iree_hal_cuda_new_event_release(iree_hal_cuda_new_event_t *event);

iree_status_t iree_hal_cuda_new_event_pool_allocate(
	const iree_hal_cuda_new_dynamic_symbols_t *syms,
	iree_host_size_t available_capacity, iree_allocator_t host_allocator,
	iree_hal_cuda_new_event_pool_t **out_event_pool);

void iree_hal_cuda_new_event_pool_retain(
	iree_hal_cuda_new_event_pool_t *event_pool);

void iree_hal_cuda_new_event_pool_release(
	iree_hal_cuda_new_event_pool_t *event_pool);

iree_status_t iree_hal_cuda_new_event_pool_acquire(
	iree_hal_cuda_new_event_pool_t *event_pool,
	iree_host_size_t event_count,
	iree_hal_cuda_new_event_t **out_events);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // IREE_HAL_DRIVERS_CUDA_NEW_EVENT_POOL_H_
