// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_CUDA_NEW_ALLOCATOR_H_
#define IREE_HAL_DRIVERS_CUDA_NEW_ALLOCATOR_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "dynamic_symbols.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

iree_status_t iree_hal_cuda_new_allocator_create(
	iree_hal_device_t *parent_device,
	const iree_hal_cuda_new_dynamic_symbols_t *syms, CUdevice device,
	CUstream stream, iree_allocator_t host_allocator,
	iree_hal_allocator_t **out_allocator);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // IREE_HAL_DRIVERS_CUDA_NEW_ALLOCATOR_H_
