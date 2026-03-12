// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "dispatch_builder.h"

iree_status_t iree_hal_radiance_dispatch_builder_build_launch(
	const iree_hal_radiance_dispatch_params_t *dispatch_params,
	iree_hal_radiance_launch_params_t *out_launch_params) {
	IREE_ASSERT_ARGUMENT(dispatch_params);
	IREE_ASSERT_ARGUMENT(out_launch_params);
	if (dispatch_params->grid_x == 0 || dispatch_params->block_x == 0) {
		return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
			"dispatch requires non-zero grid_x/block_x");
	}
	memset(out_launch_params, 0, sizeof(*out_launch_params));
	out_launch_params->stream_id = dispatch_params->stream_id;
	out_launch_params->start_pc = dispatch_params->start_pc;
	out_launch_params->kernel_pc = dispatch_params->kernel_pc;
	out_launch_params->grid_x = dispatch_params->grid_x;
	out_launch_params->grid_y = dispatch_params->grid_y;
	out_launch_params->grid_z = dispatch_params->grid_z;
	out_launch_params->block_x = dispatch_params->block_x;
	out_launch_params->block_y = dispatch_params->block_y;
	out_launch_params->block_z = dispatch_params->block_z;
	out_launch_params->regs_per_thread = dispatch_params->regs_per_thread;
	out_launch_params->shmem_per_block = dispatch_params->shmem_per_block;
	out_launch_params->params_data = dispatch_params->packed_params;
	return iree_ok_status();
}
