// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Launch ABI contract for cuda_new kernel dispatch.
//
// These concepts must remain distinct throughout the dispatch pipeline:
//   - HAL binding ordinal: index into the HAL binding table
//   - Subspan byte offset: offset within a HAL binding (e.g. weight vs bias
//     sharing one buffer)
//   - Kernel binding index: positional index in the kernel parameter list
//   - Constants: scalar values passed via the constants table
//
// This matters because the compiler may pack multiple logical tensors into a
// single HAL binding with different subspan offsets. The dispatch code must
// compute final device pointers as: binding_ptr[ordinal] + subspan_offset.

#ifndef IREE_HAL_DRIVERS_CUDA_NEW_LAUNCH_ABI_H_
#define IREE_HAL_DRIVERS_CUDA_NEW_LAUNCH_ABI_H_

#include <stdint.h>

#include "headers.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// Maps a kernel parameter slot to its source in the HAL dispatch.
typedef struct iree_hal_cuda_new_binding_ref_t {
	uint32_t hal_binding_ordinal;
	uint64_t subspan_byte_offset;
} iree_hal_cuda_new_binding_ref_t;

// Fully resolved launch parameters for a single kernel dispatch.
typedef struct iree_hal_cuda_new_launch_params_t {
	CUfunction function;

	uint32_t grid_x;
	uint32_t grid_y;
	uint32_t grid_z;

	uint32_t block_x;
	uint32_t block_y;
	uint32_t block_z;

	uint32_t shared_memory_bytes;

	uint32_t binding_count;
	CUdeviceptr binding_ptrs[16];

	uint32_t constant_count;
	uint32_t constants[64];
} iree_hal_cuda_new_launch_params_t;

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // IREE_HAL_DRIVERS_CUDA_NEW_LAUNCH_ABI_H_
