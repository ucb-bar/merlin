// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "driver.h"

#include <inttypes.h>
#include <string.h>

#include "api.h"

#define IREE_HAL_CUDA_NEW_DEVICE_ID_DEFAULT 0

//===----------------------------------------------------------------------===//
// iree_hal_cuda_new_logical_device_options_t
//===----------------------------------------------------------------------===//

IREE_API_EXPORT void iree_hal_cuda_new_logical_device_options_initialize(
	iree_hal_cuda_new_logical_device_options_t *out_options) {
	IREE_ASSERT_ARGUMENT(out_options);
	memset(out_options, 0, sizeof(*out_options));
	out_options->queue_count = 1;
	out_options->arena_block_size = 32 * 1024;
	out_options->stream_tracing = 0;
	out_options->async_allocations = false;
}

//===----------------------------------------------------------------------===//
// iree_hal_cuda_new_driver_options_t
//===----------------------------------------------------------------------===//

IREE_API_EXPORT void iree_hal_cuda_new_driver_options_initialize(
	iree_hal_cuda_new_driver_options_t *out_options) {
	IREE_ASSERT_ARGUMENT(out_options);
	memset(out_options, 0, sizeof(*out_options));
	iree_hal_cuda_new_logical_device_options_initialize(
		&out_options->default_device_options);
}

static iree_status_t iree_hal_cuda_new_driver_options_verify(
	const iree_hal_cuda_new_driver_options_t *options) {
	IREE_ASSERT_ARGUMENT(options);
	(void)options;
	return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_cuda_new_driver_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_cuda_new_driver_t {
	iree_hal_resource_t resource;
	iree_allocator_t host_allocator;

	iree_string_view_t identifier;
	iree_hal_cuda_new_driver_options_t options;

	// + trailing identifier string storage
} iree_hal_cuda_new_driver_t;

static const iree_hal_driver_vtable_t iree_hal_cuda_new_driver_vtable;

static iree_hal_cuda_new_driver_t *iree_hal_cuda_new_driver_cast(
	iree_hal_driver_t *base_value) {
	IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_cuda_new_driver_vtable);
	return (iree_hal_cuda_new_driver_t *)base_value;
}

IREE_API_EXPORT iree_status_t iree_hal_cuda_new_driver_create(
	iree_string_view_t identifier,
	const iree_hal_cuda_new_driver_options_t *options,
	iree_allocator_t host_allocator, iree_hal_driver_t **out_driver) {
	IREE_ASSERT_ARGUMENT(options);
	IREE_ASSERT_ARGUMENT(out_driver);
	*out_driver = NULL;
	IREE_TRACE_ZONE_BEGIN(z0);

	IREE_RETURN_AND_END_ZONE_IF_ERROR(
		z0, iree_hal_cuda_new_driver_options_verify(options));

	iree_hal_cuda_new_driver_t *driver = NULL;
	const iree_host_size_t total_size = sizeof(*driver) + identifier.size;
	IREE_RETURN_AND_END_ZONE_IF_ERROR(
		z0,
		iree_allocator_malloc(host_allocator, total_size, (void **)&driver));
	iree_hal_resource_initialize(&iree_hal_cuda_new_driver_vtable,
		&driver->resource);
	driver->host_allocator = host_allocator;
	iree_string_view_append_to_buffer(identifier, &driver->identifier,
		(char *)driver + total_size - identifier.size);
	memcpy(&driver->options, options, sizeof(*options));

	*out_driver = (iree_hal_driver_t *)driver;
	IREE_TRACE_ZONE_END(z0);
	return iree_ok_status();
}

static void iree_hal_cuda_new_driver_destroy(iree_hal_driver_t *base_driver) {
	iree_hal_cuda_new_driver_t *driver =
		iree_hal_cuda_new_driver_cast(base_driver);
	iree_allocator_t host_allocator = driver->host_allocator;
	IREE_TRACE_ZONE_BEGIN(z0);

	iree_allocator_free(host_allocator, driver);

	IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_cuda_new_driver_query_available_devices(
	iree_hal_driver_t *base_driver, iree_allocator_t host_allocator,
	iree_host_size_t *out_device_info_count,
	iree_hal_device_info_t **out_device_infos) {
	(void)base_driver;
	(void)host_allocator;
	*out_device_info_count = 0;
	*out_device_infos = NULL;
	return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
		"cuda_new device enumeration not yet implemented (spike 2)");
}

static iree_status_t iree_hal_cuda_new_driver_dump_device_info(
	iree_hal_driver_t *base_driver, iree_hal_device_id_t device_id,
	iree_string_builder_t *builder) {
	(void)base_driver;
	(void)device_id;
	return iree_string_builder_append_cstring(builder,
		"cuda_new: device info not yet available\n");
}

static iree_status_t iree_hal_cuda_new_driver_create_device_by_id(
	iree_hal_driver_t *base_driver, iree_hal_device_id_t device_id,
	iree_host_size_t param_count, const iree_string_pair_t *params,
	const iree_hal_device_create_params_t *create_params,
	iree_allocator_t host_allocator, iree_hal_device_t **out_device) {
	(void)base_driver;
	(void)device_id;
	(void)param_count;
	(void)params;
	(void)create_params;
	(void)host_allocator;
	(void)out_device;
	return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
		"cuda_new device creation not yet implemented (spike 2)");
}

static iree_status_t iree_hal_cuda_new_driver_create_device_by_path(
	iree_hal_driver_t *base_driver, iree_string_view_t driver_name,
	iree_string_view_t device_path, iree_host_size_t param_count,
	const iree_string_pair_t *params,
	const iree_hal_device_create_params_t *create_params,
	iree_allocator_t host_allocator, iree_hal_device_t **out_device) {
	(void)base_driver;
	(void)driver_name;
	(void)device_path;
	(void)param_count;
	(void)params;
	(void)create_params;
	(void)host_allocator;
	(void)out_device;
	return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
		"cuda_new device creation not yet implemented (spike 2)");
}

static const iree_hal_driver_vtable_t iree_hal_cuda_new_driver_vtable = {
	.destroy = iree_hal_cuda_new_driver_destroy,
	.query_available_devices =
		iree_hal_cuda_new_driver_query_available_devices,
	.dump_device_info = iree_hal_cuda_new_driver_dump_device_info,
	.create_device_by_id = iree_hal_cuda_new_driver_create_device_by_id,
	.create_device_by_path = iree_hal_cuda_new_driver_create_device_by_path,
};
