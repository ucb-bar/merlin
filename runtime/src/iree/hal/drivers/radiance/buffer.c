// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "buffer.h"

void iree_hal_radiance_buffer_initialize(uint64_t device_address,
	iree_device_size_t byte_length, iree_hal_memory_type_t memory_type,
	iree_hal_buffer_usage_t allowed_usage,
	iree_hal_radiance_buffer_t *out_buffer) {
	IREE_ASSERT_ARGUMENT(out_buffer);
	out_buffer->device_address = device_address;
	out_buffer->byte_length = byte_length;
	out_buffer->memory_type = memory_type;
	out_buffer->allowed_usage = allowed_usage;
}
