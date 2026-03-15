// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "transport.h"

#ifndef MERLIN_RADIANCE_ENABLE_RPC_COMPAT
#define MERLIN_RADIANCE_ENABLE_RPC_COMPAT 1
#endif
#ifndef MERLIN_RADIANCE_ENABLE_DIRECT_SUBMIT
#define MERLIN_RADIANCE_ENABLE_DIRECT_SUBMIT 1
#endif
#ifndef MERLIN_RADIANCE_ENABLE_KMOD
#define MERLIN_RADIANCE_ENABLE_KMOD 1
#endif

iree_status_t iree_hal_radiance_transport_create(
	const iree_hal_radiance_device_options_t *options,
	iree_allocator_t host_allocator,
	iree_hal_radiance_transport_t **out_transport) {
	IREE_ASSERT_ARGUMENT(options);
	IREE_ASSERT_ARGUMENT(out_transport);
	*out_transport = NULL;
	switch (options->backend) {
		case IREE_HAL_RADIANCE_TRANSPORT_BACKEND_RPC_COMPAT:
#if MERLIN_RADIANCE_ENABLE_RPC_COMPAT
			return iree_hal_radiance_transport_rpc_compat_create(
				options, host_allocator, out_transport);
#else
			return iree_make_status(IREE_STATUS_UNAVAILABLE,
				"radiance rpc backend disabled at build time");
#endif
		case IREE_HAL_RADIANCE_TRANSPORT_BACKEND_DIRECT_SUBMIT:
#if MERLIN_RADIANCE_ENABLE_DIRECT_SUBMIT
			return iree_hal_radiance_transport_direct_submit_create(
				options, host_allocator, out_transport);
#else
			return iree_make_status(IREE_STATUS_UNAVAILABLE,
				"radiance direct backend disabled at build time");
#endif
		case IREE_HAL_RADIANCE_TRANSPORT_BACKEND_AUTO:
#if MERLIN_RADIANCE_ENABLE_DIRECT_SUBMIT
			return iree_hal_radiance_transport_direct_submit_create(
				options, host_allocator, out_transport);
#elif MERLIN_RADIANCE_ENABLE_RPC_COMPAT
			return iree_hal_radiance_transport_rpc_compat_create(
				options, host_allocator, out_transport);
#elif MERLIN_RADIANCE_ENABLE_KMOD
			return iree_hal_radiance_transport_kmod_create(
				options, host_allocator, out_transport);
#else
			return iree_make_status(IREE_STATUS_UNAVAILABLE,
				"all radiance backends disabled at build time");
#endif
		case IREE_HAL_RADIANCE_TRANSPORT_BACKEND_KMOD:
#if MERLIN_RADIANCE_ENABLE_KMOD
			return iree_hal_radiance_transport_kmod_create(
				options, host_allocator, out_transport);
#else
			return iree_make_status(IREE_STATUS_UNAVAILABLE,
				"radiance kmod backend disabled at build time");
#endif
		default:
			return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
				"unknown radiance backend=%d", options->backend);
	}
}

void iree_hal_radiance_transport_destroy(
	iree_hal_radiance_transport_t *transport) {
	if (!transport)
		return;
	transport->vtable->destroy(transport);
}

iree_status_t iree_hal_radiance_transport_alloc_device(
	iree_hal_radiance_transport_t *transport, uint32_t bytes,
	uint64_t *out_device_address) {
	return transport->vtable->alloc_device(
		transport, bytes, out_device_address);
}

iree_status_t iree_hal_radiance_transport_submit_copy(
	iree_hal_radiance_transport_t *transport, uint8_t stream_id,
	uint64_t src_address, uint64_t dst_address, uint32_t length,
	iree_hal_radiance_copy_direction_t direction) {
	return transport->vtable->submit_copy(
		transport, stream_id, src_address, dst_address, length, direction);
}

iree_status_t iree_hal_radiance_transport_submit_fill(
	iree_hal_radiance_transport_t *transport, uint8_t stream_id,
	uint64_t dst_address, uint32_t value, uint32_t length) {
	return transport->vtable->submit_fill(
		transport, stream_id, dst_address, value, length);
}

iree_status_t iree_hal_radiance_transport_submit_dispatch(
	iree_hal_radiance_transport_t *transport,
	const iree_hal_radiance_launch_params_t *launch_params) {
	return transport->vtable->submit_dispatch(transport, launch_params);
}

iree_status_t iree_hal_radiance_transport_synchronize(
	iree_hal_radiance_transport_t *transport, uint8_t stream_id) {
	return transport->vtable->synchronize(transport, stream_id);
}
