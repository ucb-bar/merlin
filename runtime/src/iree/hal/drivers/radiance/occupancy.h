// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_RADIANCE_OCCUPANCY_H_
#define IREE_HAL_DRIVERS_RADIANCE_OCCUPANCY_H_

#include <stdint.h>

#include "target_caps.h"

typedef struct iree_hal_radiance_occupancy_result_t {
	uint32_t resident_warps_per_core;
	uint32_t max_threadblocks_per_cluster;
} iree_hal_radiance_occupancy_result_t;

void iree_hal_radiance_occupancy_estimate(
	const iree_hal_radiance_target_caps_t *caps, uint32_t regs_per_thread,
	uint32_t shmem_per_block, uint32_t threads_per_block,
	iree_hal_radiance_occupancy_result_t *out_result);

#endif // IREE_HAL_DRIVERS_RADIANCE_OCCUPANCY_H_
