// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "submission.h"

static iree_status_t iree_hal_radiance_submission_dispatch_record(
	const iree_hal_radiance_recorded_cmd_t *command,
	iree_hal_radiance_transport_t *transport) {
	iree_hal_radiance_dispatch_params_t dispatch_params = {};
	dispatch_params.stream_id = command->stream_id;
	dispatch_params.start_pc = (uint32_t)command->arg0;
	dispatch_params.kernel_pc = (uint32_t)command->arg1;
	dispatch_params.grid_x = 1;
	dispatch_params.grid_y = 1;
	dispatch_params.grid_z = 1;
	dispatch_params.block_x = 16;
	dispatch_params.block_y = 1;
	dispatch_params.block_z = 1;
	dispatch_params.regs_per_thread = 32;
	dispatch_params.shmem_per_block = 0;
	dispatch_params.packed_params = iree_const_byte_span_empty();

	iree_hal_radiance_launch_params_t launch_params;
	IREE_RETURN_IF_ERROR(iree_hal_radiance_dispatch_builder_build_launch(
		&dispatch_params, &launch_params));
	return iree_hal_radiance_transport_submit_dispatch(
		transport, &launch_params);
}

iree_status_t iree_hal_radiance_submission_replay(
	const iree_hal_radiance_command_buffer_t *command_buffer,
	iree_hal_radiance_transport_t *transport) {
	IREE_ASSERT_ARGUMENT(command_buffer);
	IREE_ASSERT_ARGUMENT(transport);

	for (iree_host_size_t i = 0; i < command_buffer->count; ++i) {
		const iree_hal_radiance_recorded_cmd_t *command =
			&command_buffer->commands[i];
		switch (command->type) {
			case IREE_HAL_RADIANCE_RECORDED_CMD_FILL:
				IREE_RETURN_IF_ERROR(iree_hal_radiance_transport_submit_fill(
					transport, command->stream_id, command->dst,
					(uint32_t)command->arg0, (uint32_t)command->arg1));
				break;
			case IREE_HAL_RADIANCE_RECORDED_CMD_COPY:
			case IREE_HAL_RADIANCE_RECORDED_CMD_UPDATE:
				IREE_RETURN_IF_ERROR(iree_hal_radiance_transport_submit_copy(
					transport, command->stream_id, command->src, command->dst,
					(uint32_t)command->arg0,
					IREE_HAL_RADIANCE_COPY_DIRECTION_H2D));
				break;
			case IREE_HAL_RADIANCE_RECORDED_CMD_DISPATCH:
				IREE_RETURN_IF_ERROR(
					iree_hal_radiance_submission_dispatch_record(
						command, transport));
				break;
			case IREE_HAL_RADIANCE_RECORDED_CMD_BARRIER:
			case IREE_HAL_RADIANCE_RECORDED_CMD_WAIT_SEMAPHORE:
			case IREE_HAL_RADIANCE_RECORDED_CMD_SIGNAL_SEMAPHORE:
				IREE_RETURN_IF_ERROR(iree_hal_radiance_transport_synchronize(
					transport, command->stream_id));
				break;
			default:
				return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
					"unsupported recorded command type=%d", (int)command->type);
		}
	}

	return iree_ok_status();
}
