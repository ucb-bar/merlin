// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "command_buffer.h"

iree_status_t iree_hal_radiance_command_buffer_initialize(
	iree_allocator_t host_allocator,
	iree_hal_radiance_command_buffer_t *out_command_buffer) {
	IREE_ASSERT_ARGUMENT(out_command_buffer);
	memset(out_command_buffer, 0, sizeof(*out_command_buffer));
	out_command_buffer->host_allocator = host_allocator;
	return iree_ok_status();
}

void iree_hal_radiance_command_buffer_deinitialize(
	iree_hal_radiance_command_buffer_t *command_buffer) {
	if (!command_buffer)
		return;
	if (command_buffer->commands) {
		iree_allocator_free(
			command_buffer->host_allocator, command_buffer->commands);
	}
	memset(command_buffer, 0, sizeof(*command_buffer));
}

iree_status_t iree_hal_radiance_command_buffer_append(
	iree_hal_radiance_command_buffer_t *command_buffer,
	const iree_hal_radiance_recorded_cmd_t *command) {
	IREE_ASSERT_ARGUMENT(command_buffer);
	IREE_ASSERT_ARGUMENT(command);

	if (command_buffer->count == command_buffer->capacity) {
		const iree_host_size_t old_capacity = command_buffer->capacity;
		const iree_host_size_t new_capacity =
			old_capacity == 0 ? 16 : old_capacity * 2;
		IREE_RETURN_IF_ERROR(
			iree_allocator_realloc_array(command_buffer->host_allocator,
				new_capacity, sizeof(*command_buffer->commands),
				(void **)&command_buffer->commands));
		command_buffer->capacity = new_capacity;
	}

	command_buffer->commands[command_buffer->count++] = *command;
	return iree_ok_status();
}
