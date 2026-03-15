// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "transport.h"

typedef struct iree_hal_radiance_direct_submit_transport_t {
	iree_hal_radiance_transport_t base;
	uint64_t next_device_address;
	uint64_t tail_command_id;
} iree_hal_radiance_direct_submit_transport_t;

static const iree_hal_radiance_transport_vtable_t
	iree_hal_radiance_direct_submit_transport_vtable;

static iree_hal_radiance_direct_submit_transport_t *
iree_hal_radiance_direct_submit_transport_cast(
	iree_hal_radiance_transport_t *base_transport) {
	return (iree_hal_radiance_direct_submit_transport_t *)base_transport;
}

static uint64_t iree_hal_radiance_align64(uint64_t value, uint64_t alignment) {
	return (value + alignment - 1u) & ~(alignment - 1u);
}

static iree_status_t iree_hal_radiance_direct_submit_alloc_device(
	iree_hal_radiance_transport_t *base_transport, uint32_t bytes,
	uint64_t *out_device_address) {
	iree_hal_radiance_direct_submit_transport_t *transport =
		iree_hal_radiance_direct_submit_transport_cast(base_transport);
	IREE_ASSERT_ARGUMENT(out_device_address);

	const uint64_t aligned_bytes = iree_hal_radiance_align64(bytes, 64);
	*out_device_address = transport->next_device_address;
	transport->next_device_address += aligned_bytes;
	return iree_ok_status();
}

static iree_status_t iree_hal_radiance_direct_submit_submit_copy(
	iree_hal_radiance_transport_t *base_transport, uint8_t stream_id,
	uint64_t src_address, uint64_t dst_address, uint32_t length,
	iree_hal_radiance_copy_direction_t direction) {
	iree_hal_radiance_direct_submit_transport_t *transport =
		iree_hal_radiance_direct_submit_transport_cast(base_transport);
	if (direction != IREE_HAL_RADIANCE_COPY_DIRECTION_H2D &&
		direction != IREE_HAL_RADIANCE_COPY_DIRECTION_D2H) {
		return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
			"unsupported copy direction=%d", (int)direction);
	}
	(void)stream_id;
	(void)src_address;
	(void)dst_address;
	(void)length;
	++transport->tail_command_id;
	return iree_ok_status();
}

static iree_status_t iree_hal_radiance_direct_submit_submit_fill(
	iree_hal_radiance_transport_t *base_transport, uint8_t stream_id,
	uint64_t dst_address, uint32_t value, uint32_t length) {
	iree_hal_radiance_direct_submit_transport_t *transport =
		iree_hal_radiance_direct_submit_transport_cast(base_transport);
	(void)stream_id;
	(void)dst_address;
	(void)value;
	(void)length;
	++transport->tail_command_id;
	return iree_ok_status();
}

static iree_status_t iree_hal_radiance_direct_submit_submit_dispatch(
	iree_hal_radiance_transport_t *base_transport,
	const iree_hal_radiance_launch_params_t *launch_params) {
	iree_hal_radiance_direct_submit_transport_t *transport =
		iree_hal_radiance_direct_submit_transport_cast(base_transport);
	IREE_ASSERT_ARGUMENT(launch_params);
	if (launch_params->grid_x == 0 || launch_params->block_x == 0) {
		return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
			"dispatch requires non-zero grid_x and block_x");
	}
	++transport->tail_command_id;
	return iree_ok_status();
}

static iree_status_t iree_hal_radiance_direct_submit_synchronize(
	iree_hal_radiance_transport_t *base_transport, uint8_t stream_id) {
	(void)base_transport;
	(void)stream_id;
	return iree_ok_status();
}

static void iree_hal_radiance_direct_submit_destroy(
	iree_hal_radiance_transport_t *base_transport) {
	iree_hal_radiance_direct_submit_transport_t *transport =
		iree_hal_radiance_direct_submit_transport_cast(base_transport);
	iree_allocator_t host_allocator = base_transport->host_allocator;
	iree_allocator_free(host_allocator, transport);
}

static const iree_hal_radiance_transport_vtable_t
	iree_hal_radiance_direct_submit_transport_vtable = {
		.destroy = iree_hal_radiance_direct_submit_destroy,
		.alloc_device = iree_hal_radiance_direct_submit_alloc_device,
		.submit_copy = iree_hal_radiance_direct_submit_submit_copy,
		.submit_fill = iree_hal_radiance_direct_submit_submit_fill,
		.submit_dispatch = iree_hal_radiance_direct_submit_submit_dispatch,
		.synchronize = iree_hal_radiance_direct_submit_synchronize,
};

iree_status_t iree_hal_radiance_transport_direct_submit_create(
	const iree_hal_radiance_device_options_t *options,
	iree_allocator_t host_allocator,
	iree_hal_radiance_transport_t **out_transport) {
	IREE_ASSERT_ARGUMENT(options);
	IREE_ASSERT_ARGUMENT(out_transport);
	*out_transport = NULL;

	iree_hal_radiance_direct_submit_transport_t *transport = NULL;
	IREE_RETURN_IF_ERROR(iree_allocator_malloc(
		host_allocator, sizeof(*transport), (void **)&transport));
	memset(transport, 0, sizeof(*transport));
	transport->base.vtable = &iree_hal_radiance_direct_submit_transport_vtable;
	transport->base.host_allocator = host_allocator;
	transport->next_device_address = 0x10000000ull;
	transport->tail_command_id = 0;

	(void)options;
	*out_transport = &transport->base;
	return iree_ok_status();
}
