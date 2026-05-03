// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_CUDA_NEW_TARGET_CAPS_H_
#define IREE_HAL_DRIVERS_CUDA_NEW_TARGET_CAPS_H_

#include <stdbool.h>
#include <stdint.h>

typedef struct iree_hal_cuda_new_target_caps_t {
	int cuda_ordinal;
	int compute_capability_major;
	int compute_capability_minor;
	uint32_t warp_size;
	uint32_t max_threads_per_block;
	uint32_t max_block_dims[3];
	uint32_t max_grid_dims[3];
	uint32_t max_shared_memory_per_block;
	uint32_t pointer_size_bits;
	// CUDA 12.x / SM 9.0+ features.
	bool supports_clusters;
	uint32_t max_cluster_size;
	bool supports_tma;
	bool supports_async_copy;
} iree_hal_cuda_new_target_caps_t;

void iree_hal_cuda_new_target_caps_initialize_defaults(
	iree_hal_cuda_new_target_caps_t *out_caps);

#endif // IREE_HAL_DRIVERS_CUDA_NEW_TARGET_CAPS_H_
