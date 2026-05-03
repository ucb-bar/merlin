// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "logical_device.h"

iree_status_t iree_hal_cuda_new_logical_device_create(
	iree_string_view_t identifier,
	const iree_hal_cuda_new_logical_device_options_t *options,
	const iree_hal_cuda_new_dynamic_symbols_t *syms,
	const iree_hal_cuda_new_physical_device_t *physical_device,
	iree_allocator_t host_allocator, iree_hal_device_t **out_device) {
	(void)identifier;
	(void)options;
	(void)syms;
	(void)physical_device;
	(void)host_allocator;
	(void)out_device;
	return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
		"cuda_new logical device creation not yet implemented (spike 2)");
}
