// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_RADIANCE_SEMAPHORE_H_
#define IREE_HAL_DRIVERS_RADIANCE_SEMAPHORE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

iree_status_t iree_hal_radiance_semaphore_create(
	iree_hal_queue_affinity_t queue_affinity, uint64_t initial_value,
	iree_hal_semaphore_flags_t flags, iree_allocator_t host_allocator,
	iree_hal_semaphore_t **out_semaphore);

bool iree_hal_radiance_semaphore_isa(iree_hal_semaphore_t *semaphore);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // IREE_HAL_DRIVERS_RADIANCE_SEMAPHORE_H_
