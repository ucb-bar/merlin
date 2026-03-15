// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_RADIANCE_BUFFER_H_
#define IREE_HAL_DRIVERS_RADIANCE_BUFFER_H_

#include <stdint.h>

#include "iree/hal/api.h"

typedef struct iree_hal_radiance_buffer_t {
	uint64_t device_address;
	iree_device_size_t byte_length;
	iree_hal_memory_type_t memory_type;
	iree_hal_buffer_usage_t allowed_usage;
} iree_hal_radiance_buffer_t;

void iree_hal_radiance_buffer_initialize(uint64_t device_address,
	iree_device_size_t byte_length, iree_hal_memory_type_t memory_type,
	iree_hal_buffer_usage_t allowed_usage,
	iree_hal_radiance_buffer_t *out_buffer);

#endif // IREE_HAL_DRIVERS_RADIANCE_BUFFER_H_
