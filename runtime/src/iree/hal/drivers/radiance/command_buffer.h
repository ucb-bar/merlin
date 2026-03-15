// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_RADIANCE_COMMAND_BUFFER_H_
#define IREE_HAL_DRIVERS_RADIANCE_COMMAND_BUFFER_H_

#include <stdint.h>

#include "iree/base/api.h"

typedef enum iree_hal_radiance_recorded_cmd_type_e {
	IREE_HAL_RADIANCE_RECORDED_CMD_FILL = 0,
	IREE_HAL_RADIANCE_RECORDED_CMD_UPDATE = 1,
	IREE_HAL_RADIANCE_RECORDED_CMD_COPY = 2,
	IREE_HAL_RADIANCE_RECORDED_CMD_DISPATCH = 3,
	IREE_HAL_RADIANCE_RECORDED_CMD_BARRIER = 4,
	IREE_HAL_RADIANCE_RECORDED_CMD_WAIT_SEMAPHORE = 5,
	IREE_HAL_RADIANCE_RECORDED_CMD_SIGNAL_SEMAPHORE = 6,
} iree_hal_radiance_recorded_cmd_type_t;

typedef struct iree_hal_radiance_recorded_cmd_t {
	iree_hal_radiance_recorded_cmd_type_t type;
	uint8_t stream_id;
	uint64_t src;
	uint64_t dst;
	uint64_t arg0;
	uint64_t arg1;
} iree_hal_radiance_recorded_cmd_t;

typedef struct iree_hal_radiance_command_buffer_t {
	iree_allocator_t host_allocator;
	iree_hal_radiance_recorded_cmd_t *commands;
	iree_host_size_t count;
	iree_host_size_t capacity;
} iree_hal_radiance_command_buffer_t;

iree_status_t iree_hal_radiance_command_buffer_initialize(
	iree_allocator_t host_allocator,
	iree_hal_radiance_command_buffer_t *out_command_buffer);

void iree_hal_radiance_command_buffer_deinitialize(
	iree_hal_radiance_command_buffer_t *command_buffer);

iree_status_t iree_hal_radiance_command_buffer_append(
	iree_hal_radiance_command_buffer_t *command_buffer,
	const iree_hal_radiance_recorded_cmd_t *command);

#endif // IREE_HAL_DRIVERS_RADIANCE_COMMAND_BUFFER_H_
