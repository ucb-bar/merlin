// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "transport.h"

iree_status_t iree_hal_radiance_transport_kmod_create(
	const iree_hal_radiance_device_options_t *options,
	iree_allocator_t host_allocator,
	iree_hal_radiance_transport_t **out_transport) {
	if (out_transport)
		*out_transport = NULL;
	(void)options;
	(void)host_allocator;
	return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
		"radiance kmod transport is not implemented; use radiance://rpc or "
		"radiance://direct for bring-up");
}
