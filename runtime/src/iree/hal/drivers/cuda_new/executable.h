// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_CUDA_NEW_EXECUTABLE_H_
#define IREE_HAL_DRIVERS_CUDA_NEW_EXECUTABLE_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/tracing.h"
#include "iree/hal/api.h"
#include "dynamic_symbols.h"
#include "headers.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#define IREE_HAL_CUDA_NEW_MAX_DISPATCH_BINDING_COUNT 16
#define IREE_HAL_CUDA_NEW_MAX_DISPATCH_CONSTANT_COUNT 64

typedef struct iree_hal_cuda_new_kernel_debug_info_t {
	iree_string_view_t function_name;
	iree_string_view_t source_filename;
	uint32_t source_line;
} iree_hal_cuda_new_kernel_debug_info_t;

typedef struct iree_hal_cuda_new_kernel_params_t {
	CUfunction function;

	uint32_t constant_count;
	uint32_t binding_count;

	// Grid dims baked at compile time from static tensor shapes.
	uint32_t grid_dims[3];

	// Block dims ({1,1,1} for CTL1, from executable for PTXE).
	uint32_t block_dims[3];

	// CTA cluster dims for Hopper (0,0,0 = no clustering).
	uint32_t cluster_dims[3];

	uint32_t block_shared_memory_size;

	IREE_TRACE(iree_hal_cuda_new_kernel_debug_info_t debug_info;)
} iree_hal_cuda_new_kernel_params_t;

iree_status_t iree_hal_cuda_new_executable_infer_format(
	iree_const_byte_span_t executable_data,
	iree_host_size_t executable_format_capacity, char *executable_format,
	iree_host_size_t *out_inferred_size);

iree_status_t iree_hal_cuda_new_executable_create(
	const iree_hal_cuda_new_dynamic_symbols_t *syms, CUdevice device,
	CUcontext cu_context,
	const iree_hal_executable_params_t *executable_params,
	iree_allocator_t host_allocator, iree_hal_executable_t **out_executable);

iree_status_t iree_hal_cuda_new_executable_create_ptxe(
	const iree_hal_cuda_new_dynamic_symbols_t *syms, CUdevice device,
	CUcontext cu_context,
	const iree_hal_executable_params_t *executable_params,
	iree_allocator_t host_allocator, iree_hal_executable_t **out_executable);

iree_status_t iree_hal_cuda_new_executable_lookup_kernel_params(
	iree_hal_executable_t *executable,
	iree_hal_executable_export_ordinal_t export_ordinal,
	const iree_hal_cuda_new_kernel_params_t **out_params);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // IREE_HAL_DRIVERS_CUDA_NEW_EXECUTABLE_H_
