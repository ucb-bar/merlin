// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_CUDA_NEW_BUFFER_H_
#define IREE_HAL_DRIVERS_CUDA_NEW_BUFFER_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "headers.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef enum iree_hal_cuda_new_buffer_type_e {
	IREE_HAL_CUDA_NEW_BUFFER_TYPE_DEVICE = 0,
	IREE_HAL_CUDA_NEW_BUFFER_TYPE_HOST,
	IREE_HAL_CUDA_NEW_BUFFER_TYPE_HOST_REGISTERED,
	IREE_HAL_CUDA_NEW_BUFFER_TYPE_EXTERNAL,
} iree_hal_cuda_new_buffer_type_t;

iree_status_t iree_hal_cuda_new_buffer_wrap(
	iree_hal_buffer_placement_t placement, iree_hal_memory_type_t memory_type,
	iree_hal_memory_access_t allowed_access,
	iree_hal_buffer_usage_t allowed_usage, iree_device_size_t allocation_size,
	iree_device_size_t byte_offset, iree_device_size_t byte_length,
	iree_hal_cuda_new_buffer_type_t buffer_type, CUdeviceptr device_ptr,
	void *host_ptr, iree_hal_buffer_release_callback_t release_callback,
	iree_allocator_t host_allocator, iree_hal_buffer_t **out_buffer);

iree_hal_cuda_new_buffer_type_t iree_hal_cuda_new_buffer_type(
	const iree_hal_buffer_t *buffer);

CUdeviceptr iree_hal_cuda_new_buffer_device_pointer(
	const iree_hal_buffer_t *buffer);

void *iree_hal_cuda_new_buffer_host_pointer(const iree_hal_buffer_t *buffer);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // IREE_HAL_DRIVERS_CUDA_NEW_BUFFER_H_
