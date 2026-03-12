// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fake_transport.h"

typedef struct iree_hal_radiance_fake_transport_t {
	iree_hal_radiance_transport_t base;
	iree_hal_radiance_fake_transport_stats_t *stats;
	uint64_t next_device_address;
} iree_hal_radiance_fake_transport_t;

static const iree_hal_radiance_transport_vtable_t
	iree_hal_radiance_fake_transport_vtable;

static iree_hal_radiance_fake_transport_t *
iree_hal_radiance_fake_transport_cast(
	iree_hal_radiance_transport_t *base_transport) {
	return (iree_hal_radiance_fake_transport_t *)base_transport;
}

static void iree_hal_radiance_fake_transport_destroy(
	iree_hal_radiance_transport_t *base_transport) {
	iree_hal_radiance_fake_transport_t *transport =
		iree_hal_radiance_fake_transport_cast(base_transport);
	iree_allocator_t host_allocator = base_transport->host_allocator;
	iree_allocator_free(host_allocator, transport);
}

static iree_status_t iree_hal_radiance_fake_transport_alloc_device(
	iree_hal_radiance_transport_t *base_transport, uint32_t bytes,
	uint64_t *out_device_address) {
	iree_hal_radiance_fake_transport_t *transport =
		iree_hal_radiance_fake_transport_cast(base_transport);
	*out_device_address = transport->next_device_address;
	transport->next_device_address += ((uint64_t)bytes + 63u) & ~63ull;
	++transport->stats->alloc_count;
	return iree_ok_status();
}

static iree_status_t iree_hal_radiance_fake_transport_submit_copy(
	iree_hal_radiance_transport_t *base_transport, uint8_t stream_id,
	uint64_t src_address, uint64_t dst_address, uint32_t length,
	iree_hal_radiance_copy_direction_t direction) {
	iree_hal_radiance_fake_transport_t *transport =
		iree_hal_radiance_fake_transport_cast(base_transport);
	(void)stream_id;
	(void)src_address;
	(void)dst_address;
	(void)length;
	(void)direction;
	++transport->stats->copy_count;
	return iree_ok_status();
}

static iree_status_t iree_hal_radiance_fake_transport_submit_fill(
	iree_hal_radiance_transport_t *base_transport, uint8_t stream_id,
	uint64_t dst_address, uint32_t value, uint32_t length) {
	iree_hal_radiance_fake_transport_t *transport =
		iree_hal_radiance_fake_transport_cast(base_transport);
	(void)stream_id;
	(void)dst_address;
	(void)value;
	(void)length;
	++transport->stats->fill_count;
	return iree_ok_status();
}

static iree_status_t iree_hal_radiance_fake_transport_submit_dispatch(
	iree_hal_radiance_transport_t *base_transport,
	const iree_hal_radiance_launch_params_t *launch_params) {
	iree_hal_radiance_fake_transport_t *transport =
		iree_hal_radiance_fake_transport_cast(base_transport);
	++transport->stats->dispatch_count;
	transport->stats->last_dispatch_grid_x = launch_params->grid_x;
	transport->stats->last_dispatch_block_x = launch_params->block_x;
	transport->stats->last_dispatch_param_bytes =
		(uint32_t)launch_params->params_data.data_length;
	return iree_ok_status();
}

static iree_status_t iree_hal_radiance_fake_transport_synchronize(
	iree_hal_radiance_transport_t *base_transport, uint8_t stream_id) {
	iree_hal_radiance_fake_transport_t *transport =
		iree_hal_radiance_fake_transport_cast(base_transport);
	(void)stream_id;
	++transport->stats->sync_count;
	return iree_ok_status();
}

static const iree_hal_radiance_transport_vtable_t
	iree_hal_radiance_fake_transport_vtable = {
		.destroy = iree_hal_radiance_fake_transport_destroy,
		.alloc_device = iree_hal_radiance_fake_transport_alloc_device,
		.submit_copy = iree_hal_radiance_fake_transport_submit_copy,
		.submit_fill = iree_hal_radiance_fake_transport_submit_fill,
		.submit_dispatch = iree_hal_radiance_fake_transport_submit_dispatch,
		.synchronize = iree_hal_radiance_fake_transport_synchronize,
};

iree_status_t iree_hal_radiance_fake_transport_create(
	iree_hal_radiance_fake_transport_stats_t *stats,
	iree_allocator_t host_allocator,
	iree_hal_radiance_transport_t **out_transport) {
	IREE_ASSERT_ARGUMENT(stats);
	IREE_ASSERT_ARGUMENT(out_transport);
	*out_transport = NULL;

	iree_hal_radiance_fake_transport_t *transport = NULL;
	IREE_RETURN_IF_ERROR(iree_allocator_malloc(
		host_allocator, sizeof(*transport), (void **)&transport));
	memset(transport, 0, sizeof(*transport));
	transport->base.vtable = &iree_hal_radiance_fake_transport_vtable;
	transport->base.host_allocator = host_allocator;
	transport->stats = stats;
	transport->next_device_address = 0x40000000ull;

	*out_transport = &transport->base;
	return iree_ok_status();
}
