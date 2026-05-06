// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "driver_module.h"

#include <stdlib.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"
#include "../api.h"

IREE_FLAG(
	bool, cuda_new_use_streams, true,
	"Use CUDA streams (instead of graphs) for executing command buffers.");

IREE_FLAG(
	bool, cuda_new_async_allocations, false,
	"Enables CUDA asynchronous stream-ordered allocations when supported.");

IREE_FLAG(
	int32_t, cuda_new_tracing, 0,
	"Controls the verbosity of stream tracing when Tracy is enabled.\n"
	"  0: stream tracing disabled.\n"
	"  1: coarse command buffer level tracing enabled.\n"
	"  2: fine-grained kernel level tracing enabled.");

IREE_FLAG(int32_t, cuda_new_default_index, 0,
	"Specifies the index of the default CUDA device to use.");

IREE_FLAG(bool, cuda_new_default_index_from_mpi, true,
	"Infers the default CUDA device index from PMI_RANK or\n"
	"OMPI_COMM_WORLD_LOCAL_RANK environment variables when set.");

static bool iree_hal_cuda_new_try_parse_env_i32(const char *var_name,
	int32_t *out_value) {
	const char *var_value = getenv(var_name);
	if (!var_value || strlen(var_value) == 0) return false;
	return iree_string_view_atoi_int32(iree_make_cstring_view(var_value),
		out_value);
}

static int32_t iree_hal_cuda_new_infer_device_index_from_env(
	int32_t default_index) {
	int32_t result = 0;
	if (iree_hal_cuda_new_try_parse_env_i32("PMI_RANK", &result) ||
		iree_hal_cuda_new_try_parse_env_i32("OMPI_COMM_WORLD_LOCAL_RANK",
			&result)) {
		return result;
	}
	return default_index;
}

static iree_status_t iree_hal_cuda_new_driver_factory_enumerate(void *self,
	iree_host_size_t *out_driver_info_count,
	const iree_hal_driver_info_t **out_driver_infos) {
	(void)self;
	static const iree_hal_driver_info_t default_driver_info = {
		.driver_name = IREE_SVL("cuda_new"),
		.full_name = IREE_SVL("CUDA HAL driver (cuda_new)"),
	};
	*out_driver_info_count = 1;
	*out_driver_infos = &default_driver_info;
	return iree_ok_status();
}

static iree_status_t iree_hal_cuda_new_driver_factory_try_create(void *self,
	iree_string_view_t driver_name, iree_allocator_t host_allocator,
	iree_hal_driver_t **out_driver) {
	(void)self;
	if (!iree_string_view_equal(driver_name, IREE_SV("cuda_new"))) {
		return iree_make_status(IREE_STATUS_UNAVAILABLE,
			"no driver '%.*s' is provided by this factory",
			(int)driver_name.size, driver_name.data);
	}

	iree_hal_cuda_new_driver_options_t options;
	iree_hal_cuda_new_driver_options_initialize(&options);

	options.default_device_options.use_graphs =
		!FLAG_cuda_new_use_streams;
	options.default_device_options.async_allocations =
		FLAG_cuda_new_async_allocations;
	options.default_device_options.stream_tracing =
		FLAG_cuda_new_tracing;

	options.default_device_index = FLAG_cuda_new_default_index;
	if (FLAG_cuda_new_default_index_from_mpi) {
		options.default_device_index =
			iree_hal_cuda_new_infer_device_index_from_env(
				options.default_device_index);
	}

	// Env var overrides for backwards compatibility.
	if (getenv("IREE_CUDA_NEW_USE_GRAPHS")) {
		options.default_device_options.use_graphs = true;
	}
	if (getenv("IREE_CUDA_NEW_ASYNC_ALLOCATIONS")) {
		options.default_device_options.async_allocations = true;
	}

	return iree_hal_cuda_new_driver_create(driver_name, &options,
		host_allocator, out_driver);
}

IREE_API_EXPORT iree_status_t iree_hal_cuda_new_driver_module_register(
	iree_hal_driver_registry_t *registry) {
	static const iree_hal_driver_factory_t factory = {
		.self = NULL,
		.enumerate = iree_hal_cuda_new_driver_factory_enumerate,
		.try_create = iree_hal_cuda_new_driver_factory_try_create,
	};
	return iree_hal_driver_registry_register_factory(registry, &factory);
}
