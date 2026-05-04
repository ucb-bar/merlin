// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_CUDA_NEW_SEMAPHORE_H_
#define IREE_HAL_DRIVERS_CUDA_NEW_SEMAPHORE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "dynamic_symbols.h"
#include "headers.h"
#include "timepoint_pool.h"

typedef struct iree_async_proactor_t iree_async_proactor_t;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

iree_status_t iree_hal_cuda_new_semaphore_create(
	iree_async_proactor_t *proactor, uint64_t initial_value,
	const iree_hal_cuda_new_dynamic_symbols_t *syms,
	iree_hal_cuda_new_timepoint_pool_t *timepoint_pool,
	iree_allocator_t host_allocator,
	iree_hal_semaphore_t **out_semaphore);

// Acquires a timepoint for device signal and returns the CUevent to record.
iree_status_t iree_hal_cuda_new_semaphore_acquire_signal_event(
	iree_hal_semaphore_t *semaphore, uint64_t to_value,
	CUevent *out_event);

// Acquires a timepoint for device wait and returns the CUevent to wait on.
// Returns iree_ok_status() with *out_event=NULL if already satisfied.
iree_status_t iree_hal_cuda_new_semaphore_acquire_wait_event(
	iree_hal_semaphore_t *semaphore, uint64_t min_value,
	CUevent *out_event);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // IREE_HAL_DRIVERS_CUDA_NEW_SEMAPHORE_H_
