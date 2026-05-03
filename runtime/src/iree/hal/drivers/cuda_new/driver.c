// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "driver.h"

#include <inttypes.h>
#include <string.h>

#include "api.h"
#include "dynamic_symbols.h"
#include "logical_device.h"
#include "physical_device.h"
#include "status_util.h"

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

	iree_hal_cuda_new_dynamic_symbols_t syms;

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

	iree_status_t status = iree_hal_cuda_new_dynamic_symbols_initialize(
		host_allocator, &driver->syms);

	if (iree_status_is_ok(status)) {
		IREE_CUDA_NEW_RETURN_AND_END_ZONE_IF_ERROR(z0, &driver->syms,
			cuInit(0), "cuInit");
	}

	if (iree_status_is_ok(status)) {
		*out_driver = (iree_hal_driver_t *)driver;
	} else {
		iree_hal_driver_release((iree_hal_driver_t *)driver);
	}
	IREE_TRACE_ZONE_END(z0);
	return status;
}

static void iree_hal_cuda_new_driver_destroy(iree_hal_driver_t *base_driver) {
	iree_hal_cuda_new_driver_t *driver =
		iree_hal_cuda_new_driver_cast(base_driver);
	iree_allocator_t host_allocator = driver->host_allocator;
	IREE_TRACE_ZONE_BEGIN(z0);

	iree_hal_cuda_new_dynamic_symbols_deinitialize(&driver->syms);
	iree_allocator_free(host_allocator, driver);

	IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_cuda_new_driver_query_available_devices(
	iree_hal_driver_t *base_driver, iree_allocator_t host_allocator,
	iree_host_size_t *out_device_info_count,
	iree_hal_device_info_t **out_device_infos) {
	iree_hal_cuda_new_driver_t *driver =
		iree_hal_cuda_new_driver_cast(base_driver);
	*out_device_info_count = 0;
	*out_device_infos = NULL;

	int device_count = 0;
	IREE_CUDA_NEW_RETURN_IF_ERROR(&driver->syms,
		cuDeviceGetCount(&device_count), "cuDeviceGetCount");
	if (device_count == 0) return iree_ok_status();

	#define IREE_HAL_CUDA_NEW_MAX_PATH_LENGTH 16
	iree_host_size_t name_storage_size =
		(iree_host_size_t)device_count * IREE_HAL_CUDA_NEW_MAX_DEVICE_NAME_LENGTH;
	iree_host_size_t path_storage_size =
		(iree_host_size_t)device_count * IREE_HAL_CUDA_NEW_MAX_PATH_LENGTH;
	iree_host_size_t total_size =
		(iree_host_size_t)device_count * sizeof(iree_hal_device_info_t) +
		name_storage_size + path_storage_size;
	iree_hal_device_info_t *device_infos = NULL;
	IREE_RETURN_IF_ERROR(
		iree_allocator_malloc(host_allocator, total_size, (void **)&device_infos));

	char *name_storage =
		(char *)device_infos +
		(iree_host_size_t)device_count * sizeof(iree_hal_device_info_t);
	char *path_storage = name_storage + name_storage_size;

	for (int i = 0; i < device_count; ++i) {
		CUdevice cu_device;
		iree_status_t status = IREE_CURESULT_TO_STATUS_NEW(
			&driver->syms, cuDeviceGet(&cu_device, i));
		if (!iree_status_is_ok(status)) {
			iree_allocator_free(host_allocator, device_infos);
			return status;
		}

		char *name_buf =
			name_storage + (iree_host_size_t)i * IREE_HAL_CUDA_NEW_MAX_DEVICE_NAME_LENGTH;
		status = IREE_CURESULT_TO_STATUS_NEW(&driver->syms,
			cuDeviceGetName(name_buf,
				IREE_HAL_CUDA_NEW_MAX_DEVICE_NAME_LENGTH, cu_device));
		if (!iree_status_is_ok(status)) {
			iree_allocator_free(host_allocator, device_infos);
			return status;
		}

		char *path_buf =
			path_storage + (iree_host_size_t)i * IREE_HAL_CUDA_NEW_MAX_PATH_LENGTH;
		int path_len = snprintf(path_buf, IREE_HAL_CUDA_NEW_MAX_PATH_LENGTH,
			"%d", i);

		device_infos[i].device_id = (iree_hal_device_id_t)i;
		device_infos[i].path = iree_make_string_view(path_buf, path_len);
		device_infos[i].name =
			iree_make_string_view(name_buf, strlen(name_buf));
	}
	#undef IREE_HAL_CUDA_NEW_MAX_PATH_LENGTH

	*out_device_info_count = (iree_host_size_t)device_count;
	*out_device_infos = device_infos;
	return iree_ok_status();
}

static iree_status_t iree_hal_cuda_new_driver_dump_device_info(
	iree_hal_driver_t *base_driver, iree_hal_device_id_t device_id,
	iree_string_builder_t *builder) {
	iree_hal_cuda_new_driver_t *driver =
		iree_hal_cuda_new_driver_cast(base_driver);

	iree_hal_cuda_new_physical_device_t phys;
	iree_status_t status = iree_hal_cuda_new_physical_device_initialize(
		&driver->syms, (int)device_id, &phys);
	if (!iree_status_is_ok(status)) return status;

	IREE_RETURN_IF_ERROR(iree_string_builder_append_format(builder,
		"cuda_new device[%d]: %s (sm_%d%d)\n", phys.cuda_ordinal,
		phys.device_name, phys.caps.compute_capability_major,
		phys.caps.compute_capability_minor));

	iree_hal_cuda_new_physical_device_deinitialize(&phys);
	return iree_ok_status();
}

static iree_status_t iree_hal_cuda_new_driver_create_device_by_ordinal(
	iree_hal_cuda_new_driver_t *driver, int cuda_ordinal,
	const iree_hal_device_create_params_t *create_params,
	iree_allocator_t host_allocator, iree_hal_device_t **out_device) {
	int device_count = 0;
	IREE_CUDA_NEW_RETURN_IF_ERROR(&driver->syms,
		cuDeviceGetCount(&device_count), "cuDeviceGetCount");
	if (cuda_ordinal < 0 || cuda_ordinal >= device_count) {
		return iree_make_status(IREE_STATUS_NOT_FOUND,
			"CUDA device ordinal %d out of range (have %d device(s))",
			cuda_ordinal, device_count);
	}

	iree_hal_cuda_new_physical_device_t phys;
	IREE_RETURN_IF_ERROR(iree_hal_cuda_new_physical_device_initialize(
		&driver->syms, cuda_ordinal, &phys));

	iree_status_t status = iree_hal_cuda_new_logical_device_create(
		(iree_hal_driver_t *)driver, driver->identifier,
		&driver->options.default_device_options, &driver->syms, &phys,
		create_params, host_allocator, out_device);

	iree_hal_cuda_new_physical_device_deinitialize(&phys);
	return status;
}

static iree_status_t iree_hal_cuda_new_driver_create_device_by_id(
	iree_hal_driver_t *base_driver, iree_hal_device_id_t device_id,
	iree_host_size_t param_count, const iree_string_pair_t *params,
	const iree_hal_device_create_params_t *create_params,
	iree_allocator_t host_allocator, iree_hal_device_t **out_device) {
	iree_hal_cuda_new_driver_t *driver =
		iree_hal_cuda_new_driver_cast(base_driver);
	(void)param_count;
	(void)params;
	(void)create_params;

	int ordinal = (device_id == IREE_HAL_CUDA_NEW_DEVICE_ID_DEFAULT ||
					  device_id == IREE_HAL_DEVICE_ID_DEFAULT)
					  ? driver->options.default_device_index
					  : (int)device_id;

	return iree_hal_cuda_new_driver_create_device_by_ordinal(
		driver, ordinal, create_params, host_allocator, out_device);
}

static iree_status_t iree_hal_cuda_new_driver_create_device_by_path(
	iree_hal_driver_t *base_driver, iree_string_view_t driver_name,
	iree_string_view_t device_path, iree_host_size_t param_count,
	const iree_string_pair_t *params,
	const iree_hal_device_create_params_t *create_params,
	iree_allocator_t host_allocator, iree_hal_device_t **out_device) {
	iree_hal_cuda_new_driver_t *driver =
		iree_hal_cuda_new_driver_cast(base_driver);
	(void)driver_name;
	(void)param_count;
	(void)params;
	(void)create_params;

	int ordinal = driver->options.default_device_index;
	if (!iree_string_view_is_empty(device_path)) {
		// Parse numeric ordinal from path (e.g. "0", "1").
		uint32_t parsed = 0;
		if (!iree_string_view_atoi_uint32(device_path, &parsed)) {
			return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
				"unsupported cuda_new device path '%.*s'; "
				"expected numeric ordinal",
				(int)device_path.size, device_path.data);
		}
		ordinal = (int)parsed;
	}

	return iree_hal_cuda_new_driver_create_device_by_ordinal(
		driver, ordinal, create_params, host_allocator, out_device);
}

static const iree_hal_driver_vtable_t iree_hal_cuda_new_driver_vtable = {
	.destroy = iree_hal_cuda_new_driver_destroy,
	.query_available_devices =
		iree_hal_cuda_new_driver_query_available_devices,
	.dump_device_info = iree_hal_cuda_new_driver_dump_device_info,
	.create_device_by_id = iree_hal_cuda_new_driver_create_device_by_id,
	.create_device_by_path = iree_hal_cuda_new_driver_create_device_by_path,
};
