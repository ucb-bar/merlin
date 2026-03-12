// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "allocator.h"

void iree_hal_radiance_allocator_initialize(
	iree_hal_radiance_allocator_t *out_allocator) {
	memset(out_allocator, 0, sizeof(*out_allocator));
	out_allocator->next_device_address = 0x30000000ull;
}

iree_status_t iree_hal_radiance_allocator_alloc_device(
	iree_hal_radiance_allocator_t *allocator, uint32_t bytes,
	uint64_t *out_device_address) {
	IREE_ASSERT_ARGUMENT(allocator);
	IREE_ASSERT_ARGUMENT(out_device_address);
	*out_device_address = allocator->next_device_address;
	allocator->next_device_address += ((uint64_t)bytes + 63u) & ~63ull;
	return iree_ok_status();
}
