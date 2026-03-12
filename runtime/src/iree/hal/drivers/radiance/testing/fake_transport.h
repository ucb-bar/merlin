// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_RADIANCE_TESTING_FAKE_TRANSPORT_H_
#define IREE_HAL_DRIVERS_RADIANCE_TESTING_FAKE_TRANSPORT_H_

#include "../transport/transport.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef struct iree_hal_radiance_fake_transport_stats_t {
	uint64_t alloc_count;
	uint64_t copy_count;
	uint64_t fill_count;
	uint64_t dispatch_count;
	uint64_t sync_count;
	uint32_t last_dispatch_grid_x;
	uint32_t last_dispatch_block_x;
	uint32_t last_dispatch_param_bytes;
} iree_hal_radiance_fake_transport_stats_t;

iree_status_t iree_hal_radiance_fake_transport_create(
	iree_hal_radiance_fake_transport_stats_t *stats,
	iree_allocator_t host_allocator,
	iree_hal_radiance_transport_t **out_transport);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // IREE_HAL_DRIVERS_RADIANCE_TESTING_FAKE_TRANSPORT_H_
