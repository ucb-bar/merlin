// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "target_caps.h"

void iree_hal_radiance_target_caps_initialize_defaults(
	iree_hal_radiance_target_caps_t *out_caps) {
	if (!out_caps)
		return;
	out_caps->warp_size = 16;
	out_caps->warp_slots_per_core = 8;
	out_caps->shared_memory_per_cluster = 128 * 1024;
	out_caps->max_registers_per_thread = 128;
	out_caps->max_threads_per_block = 1024;
}
