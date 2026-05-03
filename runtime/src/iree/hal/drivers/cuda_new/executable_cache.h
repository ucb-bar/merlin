// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_CUDA_NEW_EXECUTABLE_CACHE_H_
#define IREE_HAL_DRIVERS_CUDA_NEW_EXECUTABLE_CACHE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "dynamic_symbols.h"
#include "headers.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

iree_status_t iree_hal_cuda_new_executable_cache_create(
	iree_string_view_t identifier,
	const iree_hal_cuda_new_dynamic_symbols_t *syms, CUdevice device,
	CUcontext cu_context, iree_allocator_t host_allocator,
	iree_hal_executable_cache_t **out_executable_cache);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // IREE_HAL_DRIVERS_CUDA_NEW_EXECUTABLE_CACHE_H_
