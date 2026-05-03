// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "target_caps.h"

#include <string.h>

void iree_hal_cuda_new_target_caps_initialize_defaults(
	iree_hal_cuda_new_target_caps_t *out_caps) {
	if (!out_caps) return;
	memset(out_caps, 0, sizeof(*out_caps));
	out_caps->compute_capability_major = 8;
	out_caps->compute_capability_minor = 0;
	out_caps->warp_size = 32;
	out_caps->max_threads_per_block = 1024;
	out_caps->max_block_dims[0] = 1024;
	out_caps->max_block_dims[1] = 1024;
	out_caps->max_block_dims[2] = 64;
	out_caps->max_grid_dims[0] = 2147483647;
	out_caps->max_grid_dims[1] = 65535;
	out_caps->max_grid_dims[2] = 65535;
	out_caps->max_shared_memory_per_block = 49152;
	out_caps->pointer_size_bits = 64;
	out_caps->supports_clusters = false;
	out_caps->max_cluster_size = 0;
	out_caps->supports_tma = false;
	out_caps->supports_async_copy = true;
}
