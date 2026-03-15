// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "driver.h"

#include <inttypes.h>

#include "api.h"
#include "device.h"

#define IREE_HAL_RADIANCE_DEVICE_ID_DEFAULT 0

typedef struct iree_hal_radiance_driver_t {
	iree_hal_resource_t resource;
	iree_allocator_t host_allocator;

	iree_string_view_t identifier;
	iree_hal_radiance_driver_options_t options;

	// + trailing identifier string storage.
} iree_hal_radiance_driver_t;

static const iree_hal_driver_vtable_t iree_hal_radiance_driver_vtable;

static iree_hal_radiance_driver_t *iree_hal_radiance_driver_cast(
	iree_hal_driver_t *base_value) {
	IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_radiance_driver_vtable);
	return (iree_hal_radiance_driver_t *)base_value;
}

IREE_API_EXPORT void iree_hal_radiance_driver_options_initialize(
	iree_hal_radiance_driver_options_t *out_options) {
	memset(out_options, 0, sizeof(*out_options));
	iree_hal_radiance_device_options_initialize(
		&out_options->default_device_options);
}

static iree_status_t iree_hal_radiance_driver_options_verify(
	const iree_hal_radiance_driver_options_t *options) {
	(void)options;
	return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_hal_radiance_driver_create(
	iree_string_view_t identifier,
	const iree_hal_radiance_driver_options_t *options,
	iree_allocator_t host_allocator, iree_hal_driver_t **out_driver) {
	IREE_ASSERT_ARGUMENT(options);
	IREE_ASSERT_ARGUMENT(out_driver);
	IREE_TRACE_ZONE_BEGIN(z0);
	*out_driver = NULL;

	IREE_RETURN_AND_END_ZONE_IF_ERROR(
		z0, iree_hal_radiance_driver_options_verify(options));

	iree_hal_radiance_driver_t *driver = NULL;
	const iree_host_size_t total_size = sizeof(*driver) + identifier.size;
	IREE_RETURN_AND_END_ZONE_IF_ERROR(z0,
		iree_allocator_malloc(host_allocator, total_size, (void **)&driver));
	iree_hal_resource_initialize(
		&iree_hal_radiance_driver_vtable, &driver->resource);
	driver->host_allocator = host_allocator;
	iree_string_view_append_to_buffer(identifier, &driver->identifier,
		(char *)driver + total_size - identifier.size);
	memcpy(&driver->options, options, sizeof(*options));

	*out_driver = (iree_hal_driver_t *)driver;
	IREE_TRACE_ZONE_END(z0);
	return iree_ok_status();
}

static void iree_hal_radiance_driver_destroy(iree_hal_driver_t *base_driver) {
	iree_hal_radiance_driver_t *driver =
		iree_hal_radiance_driver_cast(base_driver);
	iree_allocator_t host_allocator = driver->host_allocator;
	IREE_TRACE_ZONE_BEGIN(z0);
	iree_allocator_free(host_allocator, driver);
	IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_radiance_driver_query_available_devices(
	iree_hal_driver_t *base_driver, iree_allocator_t host_allocator,
	iree_host_size_t *out_device_info_count,
	iree_hal_device_info_t **out_device_infos) {
	(void)base_driver;
	static const iree_hal_device_info_t device_infos[1] = {
		{
			.device_id = IREE_HAL_RADIANCE_DEVICE_ID_DEFAULT,
			.name = iree_string_view_literal("default"),
		},
	};
	*out_device_info_count = IREE_ARRAYSIZE(device_infos);
	return iree_allocator_clone(host_allocator,
		iree_make_const_byte_span(device_infos, sizeof(device_infos)),
		(void **)out_device_infos);
}

static iree_status_t iree_hal_radiance_driver_dump_device_info(
	iree_hal_driver_t *base_driver, iree_hal_device_id_t device_id,
	iree_string_builder_t *builder) {
	iree_hal_radiance_driver_t *driver =
		iree_hal_radiance_driver_cast(base_driver);
	IREE_RETURN_IF_ERROR(iree_string_builder_append_format(builder,
		"Radiance device[%" PRIu64 "]: backend=%d stream=%u\n",
		(uint64_t)device_id,
		(int)driver->options.default_device_options.backend,
		(unsigned)driver->options.default_device_options.stream_id));
	return iree_ok_status();
}

static iree_status_t iree_hal_radiance_driver_apply_device_path(
	iree_string_view_t device_path,
	iree_hal_radiance_device_options_t *inout_options) {
	if (iree_string_view_is_empty(device_path))
		return iree_ok_status();
	if (iree_string_view_equal(device_path, IREE_SV("rpc"))) {
		inout_options->backend = IREE_HAL_RADIANCE_TRANSPORT_BACKEND_RPC_COMPAT;
		return iree_ok_status();
	}
	if (iree_string_view_equal(device_path, IREE_SV("direct"))) {
		inout_options->backend =
			IREE_HAL_RADIANCE_TRANSPORT_BACKEND_DIRECT_SUBMIT;
		return iree_ok_status();
	}
	if (iree_string_view_equal(device_path, IREE_SV("kmod"))) {
		inout_options->backend = IREE_HAL_RADIANCE_TRANSPORT_BACKEND_KMOD;
		return iree_ok_status();
	}
	return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
		"unsupported radiance device path '%.*s'", (int)device_path.size,
		device_path.data);
}

static iree_status_t iree_hal_radiance_driver_create_device_by_id(
	iree_hal_driver_t *base_driver, iree_hal_device_id_t device_id,
	iree_host_size_t param_count, const iree_string_pair_t *params,
	iree_allocator_t host_allocator, iree_hal_device_t **out_device) {
	iree_hal_radiance_driver_t *driver =
		iree_hal_radiance_driver_cast(base_driver);
	if (device_id != IREE_HAL_RADIANCE_DEVICE_ID_DEFAULT &&
		device_id != IREE_HAL_DEVICE_ID_DEFAULT) {
		return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
			"unsupported radiance device id: %" PRIu64, (uint64_t)device_id);
	}
	(void)param_count;
	(void)params;
	return iree_hal_radiance_device_create(driver->identifier,
		&driver->options.default_device_options, host_allocator, out_device);
}

static iree_status_t iree_hal_radiance_driver_create_device_by_path(
	iree_hal_driver_t *base_driver, iree_string_view_t driver_name,
	iree_string_view_t device_path, iree_host_size_t param_count,
	const iree_string_pair_t *params, iree_allocator_t host_allocator,
	iree_hal_device_t **out_device) {
	iree_hal_radiance_driver_t *driver =
		iree_hal_radiance_driver_cast(base_driver);

	iree_hal_radiance_device_options_t options =
		driver->options.default_device_options;
	IREE_RETURN_IF_ERROR(
		iree_hal_radiance_driver_apply_device_path(device_path, &options));
	(void)driver_name;
	(void)param_count;
	(void)params;
	return iree_hal_radiance_device_create(
		driver->identifier, &options, host_allocator, out_device);
}

static const iree_hal_driver_vtable_t iree_hal_radiance_driver_vtable = {
	.destroy = iree_hal_radiance_driver_destroy,
	.query_available_devices = iree_hal_radiance_driver_query_available_devices,
	.dump_device_info = iree_hal_radiance_driver_dump_device_info,
	.create_device_by_id = iree_hal_radiance_driver_create_device_by_id,
	.create_device_by_path = iree_hal_radiance_driver_create_device_by_path,
};
