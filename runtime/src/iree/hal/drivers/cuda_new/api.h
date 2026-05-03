// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_CUDA_NEW_API_H_
#define IREE_HAL_DRIVERS_CUDA_NEW_API_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_cuda_new_logical_device_options_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_cuda_new_logical_device_options_t {
	iree_host_size_t queue_count;
	iree_host_size_t arena_block_size;
	int32_t stream_tracing;
	bool async_allocations;
} iree_hal_cuda_new_logical_device_options_t;

IREE_API_EXPORT void iree_hal_cuda_new_logical_device_options_initialize(
	iree_hal_cuda_new_logical_device_options_t *out_options);

//===----------------------------------------------------------------------===//
// iree_hal_cuda_new_driver_options_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_cuda_new_driver_options_t {
	int default_device_index;
	iree_hal_cuda_new_logical_device_options_t default_device_options;
} iree_hal_cuda_new_driver_options_t;

IREE_API_EXPORT void iree_hal_cuda_new_driver_options_initialize(
	iree_hal_cuda_new_driver_options_t *out_options);

IREE_API_EXPORT iree_status_t iree_hal_cuda_new_driver_create(
	iree_string_view_t identifier,
	const iree_hal_cuda_new_driver_options_t *options,
	iree_allocator_t host_allocator, iree_hal_driver_t **out_driver);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // IREE_HAL_DRIVERS_CUDA_NEW_API_H_
