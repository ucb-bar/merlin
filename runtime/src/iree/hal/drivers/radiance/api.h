// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_RADIANCE_API_H_
#define IREE_HAL_DRIVERS_RADIANCE_API_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef enum iree_hal_radiance_transport_backend_e {
	IREE_HAL_RADIANCE_TRANSPORT_BACKEND_AUTO = 0,
	IREE_HAL_RADIANCE_TRANSPORT_BACKEND_RPC_COMPAT = 1,
	IREE_HAL_RADIANCE_TRANSPORT_BACKEND_DIRECT_SUBMIT = 2,
	IREE_HAL_RADIANCE_TRANSPORT_BACKEND_KMOD = 3,
} iree_hal_radiance_transport_backend_t;

// Parameters configuring a radiance device.
typedef struct iree_hal_radiance_device_options_t {
	iree_hal_radiance_transport_backend_t backend;
	iree_string_view_t rpc_socket_path;
	iree_string_view_t direct_socket_path;
	uint8_t stream_id;
	uint8_t regs_per_thread;
	uint32_t shmem_per_block;
} iree_hal_radiance_device_options_t;

IREE_API_EXPORT void iree_hal_radiance_device_options_initialize(
	iree_hal_radiance_device_options_t *out_options);

IREE_API_EXPORT iree_status_t iree_hal_radiance_device_create(
	iree_string_view_t identifier,
	const iree_hal_radiance_device_options_t *options,
	iree_allocator_t host_allocator, iree_hal_device_t **out_device);

// Parameters configuring a radiance driver.
typedef struct iree_hal_radiance_driver_options_t {
	iree_hal_radiance_device_options_t default_device_options;
} iree_hal_radiance_driver_options_t;

IREE_API_EXPORT void iree_hal_radiance_driver_options_initialize(
	iree_hal_radiance_driver_options_t *out_options);

IREE_API_EXPORT iree_status_t iree_hal_radiance_driver_create(
	iree_string_view_t identifier,
	const iree_hal_radiance_driver_options_t *options,
	iree_allocator_t host_allocator, iree_hal_driver_t **out_driver);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // IREE_HAL_DRIVERS_RADIANCE_API_H_
