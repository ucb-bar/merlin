// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_RADIANCE_TRANSPORT_TRANSPORT_H_
#define IREE_HAL_DRIVERS_RADIANCE_TRANSPORT_TRANSPORT_H_

#include <stdint.h>

#include "iree/base/api.h"

#include "../api.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef enum iree_hal_radiance_copy_direction_e {
	IREE_HAL_RADIANCE_COPY_DIRECTION_H2D = 0,
	IREE_HAL_RADIANCE_COPY_DIRECTION_D2H = 1,
} iree_hal_radiance_copy_direction_t;

typedef struct iree_hal_radiance_launch_params_t {
	uint8_t stream_id;
	uint32_t start_pc;
	uint32_t kernel_pc;
	uint32_t grid_x;
	uint32_t grid_y;
	uint32_t grid_z;
	uint32_t block_x;
	uint32_t block_y;
	uint32_t block_z;
	uint8_t regs_per_thread;
	uint32_t shmem_per_block;
	iree_const_byte_span_t params_data;
} iree_hal_radiance_launch_params_t;

typedef struct iree_hal_radiance_transport_t iree_hal_radiance_transport_t;

typedef struct iree_hal_radiance_transport_vtable_t {
	void(IREE_API_PTR *destroy)(iree_hal_radiance_transport_t *transport);
	iree_status_t(IREE_API_PTR *alloc_device)(
		iree_hal_radiance_transport_t *transport, uint32_t bytes,
		uint64_t *out_device_address);
	iree_status_t(IREE_API_PTR *submit_copy)(
		iree_hal_radiance_transport_t *transport, uint8_t stream_id,
		uint64_t src_address, uint64_t dst_address, uint32_t length,
		iree_hal_radiance_copy_direction_t direction);
	iree_status_t(IREE_API_PTR *submit_fill)(
		iree_hal_radiance_transport_t *transport, uint8_t stream_id,
		uint64_t dst_address, uint32_t value, uint32_t length);
	iree_status_t(IREE_API_PTR *submit_dispatch)(
		iree_hal_radiance_transport_t *transport,
		const iree_hal_radiance_launch_params_t *launch_params);
	iree_status_t(IREE_API_PTR *synchronize)(
		iree_hal_radiance_transport_t *transport, uint8_t stream_id);
} iree_hal_radiance_transport_vtable_t;

struct iree_hal_radiance_transport_t {
	const iree_hal_radiance_transport_vtable_t *vtable;
	iree_allocator_t host_allocator;
};

iree_status_t iree_hal_radiance_transport_create(
	const iree_hal_radiance_device_options_t *options,
	iree_allocator_t host_allocator,
	iree_hal_radiance_transport_t **out_transport);

IREE_API_EXPORT void iree_hal_radiance_transport_destroy(
	iree_hal_radiance_transport_t *transport);

IREE_API_EXPORT iree_status_t iree_hal_radiance_transport_alloc_device(
	iree_hal_radiance_transport_t *transport, uint32_t bytes,
	uint64_t *out_device_address);

IREE_API_EXPORT iree_status_t iree_hal_radiance_transport_submit_copy(
	iree_hal_radiance_transport_t *transport, uint8_t stream_id,
	uint64_t src_address, uint64_t dst_address, uint32_t length,
	iree_hal_radiance_copy_direction_t direction);

IREE_API_EXPORT iree_status_t iree_hal_radiance_transport_submit_fill(
	iree_hal_radiance_transport_t *transport, uint8_t stream_id,
	uint64_t dst_address, uint32_t value, uint32_t length);

IREE_API_EXPORT iree_status_t iree_hal_radiance_transport_submit_dispatch(
	iree_hal_radiance_transport_t *transport,
	const iree_hal_radiance_launch_params_t *launch_params);

IREE_API_EXPORT iree_status_t iree_hal_radiance_transport_synchronize(
	iree_hal_radiance_transport_t *transport, uint8_t stream_id);

iree_status_t iree_hal_radiance_transport_rpc_compat_create(
	const iree_hal_radiance_device_options_t *options,
	iree_allocator_t host_allocator,
	iree_hal_radiance_transport_t **out_transport);

iree_status_t iree_hal_radiance_transport_direct_submit_create(
	const iree_hal_radiance_device_options_t *options,
	iree_allocator_t host_allocator,
	iree_hal_radiance_transport_t **out_transport);

iree_status_t iree_hal_radiance_transport_kmod_create(
	const iree_hal_radiance_device_options_t *options,
	iree_allocator_t host_allocator,
	iree_hal_radiance_transport_t **out_transport);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // IREE_HAL_DRIVERS_RADIANCE_TRANSPORT_TRANSPORT_H_
