// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_RADIANCE_DISPATCH_BUILDER_H_
#define IREE_HAL_DRIVERS_RADIANCE_DISPATCH_BUILDER_H_

#include <stdint.h>

#include "iree/base/api.h"

#include "transport/transport.h"

typedef struct iree_hal_radiance_dispatch_params_t {
	uint8_t stream_id;
	uint32_t start_pc;
	uint32_t kernel_pc;
	uint32_t grid_x;
	uint32_t grid_y;
	uint32_t grid_z;
	uint32_t block_x;
	uint32_t block_y;
	uint32_t block_z;
	uint8_t regs_per_thread;
	uint32_t shmem_per_block;
	iree_const_byte_span_t packed_params;
} iree_hal_radiance_dispatch_params_t;

iree_status_t iree_hal_radiance_dispatch_builder_build_launch(
	const iree_hal_radiance_dispatch_params_t *dispatch_params,
	iree_hal_radiance_launch_params_t *out_launch_params);

#endif // IREE_HAL_DRIVERS_RADIANCE_DISPATCH_BUILDER_H_
