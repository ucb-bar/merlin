// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_RADIANCE_TARGET_CAPS_H_
#define IREE_HAL_DRIVERS_RADIANCE_TARGET_CAPS_H_

#include <stdint.h>

typedef struct iree_hal_radiance_target_caps_t {
	uint32_t warp_size;
	uint32_t warp_slots_per_core;
	uint32_t shared_memory_per_cluster;
	uint32_t max_registers_per_thread;
	uint32_t max_threads_per_block;
} iree_hal_radiance_target_caps_t;

void iree_hal_radiance_target_caps_initialize_defaults(
	iree_hal_radiance_target_caps_t *out_caps);

#endif // IREE_HAL_DRIVERS_RADIANCE_TARGET_CAPS_H_
