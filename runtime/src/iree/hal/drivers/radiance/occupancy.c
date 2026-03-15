// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "occupancy.h"

void iree_hal_radiance_occupancy_estimate(
	const iree_hal_radiance_target_caps_t *caps, uint32_t regs_per_thread,
	uint32_t shmem_per_block, uint32_t threads_per_block,
	iree_hal_radiance_occupancy_result_t *out_result) {
	if (!caps || !out_result)
		return;

	if (regs_per_thread == 0)
		regs_per_thread = 1;
	if (threads_per_block == 0)
		threads_per_block = 1;

	uint32_t by_regs = caps->max_registers_per_thread / regs_per_thread;
	if (by_regs == 0)
		by_regs = 1;

	uint32_t resident_warps = caps->warp_slots_per_core;
	if (resident_warps > by_regs)
		resident_warps = by_regs;
	if (resident_warps == 0)
		resident_warps = 1;

	uint32_t max_tbs_by_smem = UINT32_MAX;
	if (shmem_per_block > 0) {
		max_tbs_by_smem = caps->shared_memory_per_cluster / shmem_per_block;
		if (max_tbs_by_smem == 0)
			max_tbs_by_smem = 1;
	}

	uint32_t max_threads = caps->warp_slots_per_core * caps->warp_size;
	uint32_t max_tbs_by_threads = max_threads / threads_per_block;
	if (max_tbs_by_threads == 0)
		max_tbs_by_threads = 1;

	uint32_t max_tbs = max_tbs_by_smem;
	if (max_tbs > max_tbs_by_threads)
		max_tbs = max_tbs_by_threads;

	out_result->resident_warps_per_core = resident_warps;
	out_result->max_threadblocks_per_cluster = max_tbs;
}
