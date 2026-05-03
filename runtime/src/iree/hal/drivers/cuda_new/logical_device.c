// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "logical_device.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "allocator.h"
#include "command_buffer.h"
#include "event_pool.h"
#include "executable_cache.h"
#include "status_util.h"
#include "target_caps.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/hal/drivers/local_sync/sync_semaphore.h"
#include "iree/hal/utils/deferred_command_buffer.h"
#include "iree/hal/utils/file_registry.h"
#include "iree/hal/utils/file_transfer.h"
#include "iree/hal/utils/queue_emulation.h"

//===----------------------------------------------------------------------===//
// iree_hal_cuda_new_logical_device_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_cuda_new_logical_device_t {
	iree_hal_resource_t resource;
	iree_string_view_t identifier;
	iree_allocator_t host_allocator;

	iree_hal_driver_t *driver;
	const iree_hal_cuda_new_dynamic_symbols_t *syms;
	iree_hal_cuda_new_logical_device_options_t options;

	CUdevice cu_device;
	CUcontext cu_context;
	CUstream dispatch_stream;

	iree_hal_cuda_new_target_caps_t caps;
	iree_hal_allocator_t *device_allocator;
	iree_arena_block_pool_t block_pool;
	iree_hal_cuda_new_event_pool_t *event_pool;

	iree_async_proactor_pool_t *proactor_pool;
	iree_async_proactor_t *proactor;

	iree_hal_device_topology_info_t topology_info;

	// + trailing identifier string storage
} iree_hal_cuda_new_logical_device_t;

static const iree_hal_device_vtable_t iree_hal_cuda_new_device_vtable;

static iree_hal_cuda_new_logical_device_t *
iree_hal_cuda_new_logical_device_cast(iree_hal_device_t *base_value) {
	IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_cuda_new_device_vtable);
	return (iree_hal_cuda_new_logical_device_t *)base_value;
}

iree_status_t iree_hal_cuda_new_logical_device_create(
	iree_hal_driver_t *driver, iree_string_view_t identifier,
	const iree_hal_cuda_new_logical_device_options_t *options,
	const iree_hal_cuda_new_dynamic_symbols_t *syms,
	const iree_hal_cuda_new_physical_device_t *physical_device,
	const iree_hal_device_create_params_t *create_params,
	iree_allocator_t host_allocator, iree_hal_device_t **out_device) {
	IREE_ASSERT_ARGUMENT(driver);
	IREE_ASSERT_ARGUMENT(options);
	IREE_ASSERT_ARGUMENT(syms);
	IREE_ASSERT_ARGUMENT(physical_device);
	IREE_ASSERT_ARGUMENT(out_device);
	*out_device = NULL;
	IREE_TRACE_ZONE_BEGIN(z0);

	// Retain the primary context for this device.
	CUcontext cu_context = NULL;
	iree_status_t status = IREE_CURESULT_TO_STATUS_NEW(syms,
		cuDevicePrimaryCtxRetain(&cu_context, physical_device->cu_device),
		"cuDevicePrimaryCtxRetain");

	if (iree_status_is_ok(status)) {
		status = IREE_CURESULT_TO_STATUS_NEW(syms,
			cuCtxSetCurrent(cu_context), "cuCtxSetCurrent");
	}

	// Create the dispatch stream.
	CUstream dispatch_stream = NULL;
	if (iree_status_is_ok(status)) {
		status = IREE_CURESULT_TO_STATUS_NEW(syms,
			cuStreamCreate(&dispatch_stream, CU_STREAM_NON_BLOCKING),
			"cuStreamCreate");
	}

	// Allocate the device struct.
	iree_hal_cuda_new_logical_device_t *device = NULL;
	if (iree_status_is_ok(status)) {
		iree_host_size_t total_size = sizeof(*device) + identifier.size;
		status = iree_allocator_malloc(host_allocator, total_size,
			(void **)&device);
	}

	if (iree_status_is_ok(status)) {
		iree_hal_resource_initialize(&iree_hal_cuda_new_device_vtable,
			&device->resource);
		iree_string_view_append_to_buffer(identifier, &device->identifier,
			(char *)device + sizeof(*device));
		device->host_allocator = host_allocator;
		device->driver = driver;
		iree_hal_driver_retain(driver);
		device->syms = syms;
		device->options = *options;
		device->cu_device = physical_device->cu_device;
		device->cu_context = cu_context;
		device->dispatch_stream = dispatch_stream;
		device->caps = physical_device->caps;
		iree_arena_block_pool_initialize(
			options->arena_block_size, host_allocator, &device->block_pool);
		device->proactor_pool = create_params->proactor_pool;
		iree_async_proactor_pool_retain(device->proactor_pool);
		memset(&device->topology_info, 0, sizeof(device->topology_info));
	}

	// Create the allocator.
	if (iree_status_is_ok(status)) {
		status = iree_hal_cuda_new_allocator_create(
			(iree_hal_device_t *)device, syms, physical_device->cu_device,
			cu_context, dispatch_stream, host_allocator,
			&device->device_allocator);
	}

	// Acquire a proactor for semaphore support.
	if (iree_status_is_ok(status)) {
		status = iree_async_proactor_pool_get(
			device->proactor_pool, 0, &device->proactor);
	}

	if (iree_status_is_ok(status)) {
		status = iree_hal_cuda_new_event_pool_allocate(
			syms, /*available_capacity=*/64, host_allocator,
			&device->event_pool);
	}

	if (iree_status_is_ok(status)) {
		*out_device = (iree_hal_device_t *)device;
	} else {
		if (device) {
			if (device->device_allocator) {
				iree_hal_allocator_release(device->device_allocator);
			}
			if (device->proactor_pool) {
				iree_async_proactor_pool_release(device->proactor_pool);
			}
			if (device->event_pool) {
				iree_hal_cuda_new_event_pool_release(device->event_pool);
			}
			if (device->driver) {
				iree_hal_driver_release(device->driver);
			}
			iree_allocator_free(host_allocator, device);
		}
		if (dispatch_stream) {
			IREE_CUDA_NEW_IGNORE_ERROR(syms,
				cuStreamDestroy(dispatch_stream));
		}
		if (cu_context) {
			IREE_CUDA_NEW_IGNORE_ERROR(syms,
				cuDevicePrimaryCtxRelease(physical_device->cu_device));
		}
	}

	IREE_TRACE_ZONE_END(z0);
	return status;
}

//===----------------------------------------------------------------------===//
// Device lifecycle
//===----------------------------------------------------------------------===//

static void iree_hal_cuda_new_device_destroy(iree_hal_device_t *base_device) {
	iree_hal_cuda_new_logical_device_t *device =
		iree_hal_cuda_new_logical_device_cast(base_device);
	iree_allocator_t host_allocator = device->host_allocator;
	const iree_hal_cuda_new_dynamic_symbols_t *syms = device->syms;
	IREE_TRACE_ZONE_BEGIN(z0);

	iree_hal_allocator_release(device->device_allocator);
	iree_arena_block_pool_deinitialize(&device->block_pool);
	iree_hal_cuda_new_event_pool_release(device->event_pool);
	iree_async_proactor_pool_release(device->proactor_pool);

	IREE_CUDA_NEW_IGNORE_ERROR(syms,
		cuStreamDestroy(device->dispatch_stream));
	IREE_CUDA_NEW_IGNORE_ERROR(syms,
		cuDevicePrimaryCtxRelease(device->cu_device));

	iree_hal_driver_release(device->driver);
	iree_allocator_free(host_allocator, device);
	IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// Device queries
//===----------------------------------------------------------------------===//

static iree_string_view_t iree_hal_cuda_new_device_id(
	iree_hal_device_t *base_device) {
	iree_hal_cuda_new_logical_device_t *device =
		iree_hal_cuda_new_logical_device_cast(base_device);
	return device->identifier;
}

static iree_allocator_t iree_hal_cuda_new_device_host_allocator(
	iree_hal_device_t *base_device) {
	iree_hal_cuda_new_logical_device_t *device =
		iree_hal_cuda_new_logical_device_cast(base_device);
	return device->host_allocator;
}

static iree_hal_allocator_t *iree_hal_cuda_new_device_allocator(
	iree_hal_device_t *base_device) {
	iree_hal_cuda_new_logical_device_t *device =
		iree_hal_cuda_new_logical_device_cast(base_device);
	return device->device_allocator;
}

static void iree_hal_cuda_new_device_replace_allocator(
	iree_hal_device_t *base_device, iree_hal_allocator_t *new_allocator) {
	iree_hal_cuda_new_logical_device_t *device =
		iree_hal_cuda_new_logical_device_cast(base_device);
	iree_hal_allocator_retain(new_allocator);
	iree_hal_allocator_release(device->device_allocator);
	device->device_allocator = new_allocator;
}

static void iree_hal_cuda_new_device_replace_channel_provider(
	iree_hal_device_t *base_device,
	iree_hal_channel_provider_t *new_provider) {
	(void)base_device;
	(void)new_provider;
}

static iree_status_t iree_hal_cuda_new_device_trim(
	iree_hal_device_t *base_device) {
	iree_hal_cuda_new_logical_device_t *device =
		iree_hal_cuda_new_logical_device_cast(base_device);
	return iree_hal_allocator_trim(device->device_allocator);
}

static iree_status_t iree_hal_cuda_new_device_query_i64(
	iree_hal_device_t *base_device, iree_string_view_t category,
	iree_string_view_t key, int64_t *out_value) {
	iree_hal_cuda_new_logical_device_t *device =
		iree_hal_cuda_new_logical_device_cast(base_device);
	*out_value = 0;

	if (iree_string_view_equal(category, IREE_SV("hal.device.id"))) {
		*out_value =
			iree_string_view_match_pattern(device->identifier, key) ? 1 : 0;
		// Also accept cuda_tile device targets so existing VMFBs work.
		if (!*out_value) {
			*out_value =
				iree_string_view_match_pattern(IREE_SV("cuda_tile"), key)
					? 1
					: 0;
		}
		return iree_ok_status();
	}

	if (iree_string_view_equal(category, IREE_SV("hal.executable.format"))) {
		*out_value =
			(iree_string_view_equal(key, IREE_SV("cuda-tile-fb")) ||
			 iree_string_view_equal(key, IREE_SV("CTL1")))
				? 1
				: 0;
		return iree_ok_status();
	}

	if (iree_string_view_equal(category, IREE_SV("cuda.device"))) {
		if (iree_string_view_equal(key,
				IREE_SV("compute_capability_major"))) {
			*out_value = device->caps.compute_capability_major;
			return iree_ok_status();
		} else if (iree_string_view_equal(key,
					   IREE_SV("compute_capability_minor"))) {
			*out_value = device->caps.compute_capability_minor;
			return iree_ok_status();
		}
	}

	return iree_make_status(IREE_STATUS_NOT_FOUND,
		"unknown device configuration key value '%.*s :: %.*s'",
		(int)category.size, category.data, (int)key.size, key.data);
}

static iree_status_t iree_hal_cuda_new_device_query_capabilities(
	iree_hal_device_t *base_device,
	iree_hal_device_capabilities_t *out_capabilities) {
	memset(out_capabilities, 0, sizeof(*out_capabilities));
	return iree_ok_status();
}

static const iree_hal_device_topology_info_t *
iree_hal_cuda_new_device_topology_info(iree_hal_device_t *base_device) {
	iree_hal_cuda_new_logical_device_t *device =
		iree_hal_cuda_new_logical_device_cast(base_device);
	return &device->topology_info;
}

static iree_status_t iree_hal_cuda_new_device_refine_topology_edge(
	iree_hal_device_t *src_device, iree_hal_device_t *dst_device,
	iree_hal_topology_edge_t *edge) {
	return iree_ok_status();
}

static iree_status_t iree_hal_cuda_new_device_assign_topology_info(
	iree_hal_device_t *base_device,
	const iree_hal_device_topology_info_t *topology_info) {
	iree_hal_cuda_new_logical_device_t *device =
		iree_hal_cuda_new_logical_device_cast(base_device);
	device->topology_info = *topology_info;
	return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Unimplemented stubs
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_cuda_new_device_create_channel(
	iree_hal_device_t *base_device, iree_hal_queue_affinity_t queue_affinity,
	iree_hal_channel_params_t params, iree_hal_channel_t **out_channel) {
	return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
		"cuda_new channel not yet implemented");
}

static iree_status_t iree_hal_cuda_new_device_create_command_buffer(
	iree_hal_device_t *base_device, iree_hal_command_buffer_mode_t mode,
	iree_hal_command_category_t command_categories,
	iree_hal_queue_affinity_t queue_affinity,
	iree_host_size_t binding_capacity,
	iree_hal_command_buffer_t **out_command_buffer) {
	iree_hal_cuda_new_logical_device_t *device =
		iree_hal_cuda_new_logical_device_cast(base_device);
	return iree_hal_deferred_command_buffer_create(
		iree_hal_device_allocator(base_device), mode, command_categories,
		queue_affinity, binding_capacity, &device->block_pool,
		device->host_allocator, out_command_buffer);
}

static iree_status_t iree_hal_cuda_new_device_create_event(
	iree_hal_device_t *base_device, iree_hal_queue_affinity_t queue_affinity,
	iree_hal_event_flags_t flags, iree_hal_event_t **out_event) {
	return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
		"cuda_new event not yet implemented");
}

static iree_status_t iree_hal_cuda_new_device_create_executable_cache(
	iree_hal_device_t *base_device, iree_string_view_t identifier,
	iree_hal_executable_cache_t **out_executable_cache) {
	iree_hal_cuda_new_logical_device_t *device =
		iree_hal_cuda_new_logical_device_cast(base_device);
	return iree_hal_cuda_new_executable_cache_create(identifier,
		device->syms, device->cu_device, device->cu_context,
		device->host_allocator, out_executable_cache);
}

static iree_status_t iree_hal_cuda_new_device_import_file(
	iree_hal_device_t *base_device, iree_hal_queue_affinity_t queue_affinity,
	iree_hal_memory_access_t access, iree_io_file_handle_t *handle,
	iree_hal_external_file_flags_t flags, iree_hal_file_t **out_file) {
	return iree_hal_file_from_handle(
		iree_hal_device_allocator(base_device), queue_affinity, access,
		handle, /*proactor=*/NULL,
		iree_hal_device_host_allocator(base_device), out_file);
}

static iree_status_t iree_hal_cuda_new_device_create_semaphore(
	iree_hal_device_t *base_device, iree_hal_queue_affinity_t queue_affinity,
	uint64_t initial_value, iree_hal_semaphore_flags_t flags,
	iree_hal_semaphore_t **out_semaphore) {
	iree_hal_cuda_new_logical_device_t *device =
		iree_hal_cuda_new_logical_device_cast(base_device);
	return iree_hal_sync_semaphore_create(device->proactor, queue_affinity,
		initial_value, flags, device->host_allocator, out_semaphore);
}

static iree_hal_semaphore_compatibility_t
iree_hal_cuda_new_device_query_semaphore_compatibility(
	iree_hal_device_t *base_device, iree_hal_semaphore_t *semaphore) {
	return IREE_HAL_SEMAPHORE_COMPATIBILITY_HOST_ONLY;
}

static iree_status_t iree_hal_cuda_new_device_queue_alloca(
	iree_hal_device_t *base_device, iree_hal_queue_affinity_t queue_affinity,
	const iree_hal_semaphore_list_t wait_semaphore_list,
	const iree_hal_semaphore_list_t signal_semaphore_list,
	iree_hal_allocator_pool_t pool, iree_hal_buffer_params_t params,
	iree_device_size_t allocation_size, iree_hal_alloca_flags_t flags,
	iree_hal_buffer_t **IREE_RESTRICT out_buffer) {
	IREE_RETURN_IF_ERROR(iree_hal_semaphore_list_wait(
		wait_semaphore_list, iree_infinite_timeout(),
		IREE_ASYNC_WAIT_FLAG_NONE));

	iree_status_t status = iree_hal_allocator_allocate_buffer(
		iree_hal_device_allocator(base_device), params, allocation_size,
		out_buffer);

	if (iree_status_is_ok(status)) {
		status = iree_hal_semaphore_list_signal(signal_semaphore_list,
			/*frontier=*/NULL);
	} else {
		iree_hal_semaphore_list_fail(signal_semaphore_list,
			iree_status_clone(status));
	}
	return status;
}

static iree_status_t iree_hal_cuda_new_device_queue_dealloca(
	iree_hal_device_t *base_device, iree_hal_queue_affinity_t queue_affinity,
	const iree_hal_semaphore_list_t wait_semaphore_list,
	const iree_hal_semaphore_list_t signal_semaphore_list,
	iree_hal_buffer_t *buffer, iree_hal_dealloca_flags_t flags) {
	IREE_RETURN_IF_ERROR(iree_hal_semaphore_list_wait(
		wait_semaphore_list, iree_infinite_timeout(),
		IREE_ASYNC_WAIT_FLAG_NONE));

	iree_status_t status =
		iree_hal_semaphore_list_signal(signal_semaphore_list,
			/*frontier=*/NULL);
	if (!iree_status_is_ok(status)) {
		iree_hal_semaphore_list_fail(signal_semaphore_list,
			iree_status_clone(status));
	}
	return status;
}

static iree_status_t iree_hal_cuda_new_device_queue_read(
	iree_hal_device_t *base_device, iree_hal_queue_affinity_t queue_affinity,
	const iree_hal_semaphore_list_t wait_semaphore_list,
	const iree_hal_semaphore_list_t signal_semaphore_list,
	iree_hal_file_t *source_file, uint64_t source_offset,
	iree_hal_buffer_t *target_buffer, iree_device_size_t target_offset,
	iree_device_size_t length, iree_hal_read_flags_t flags) {
	iree_hal_file_transfer_options_t options = {
		.chunk_count = IREE_HAL_FILE_TRANSFER_CHUNK_COUNT_DEFAULT,
		.chunk_size = IREE_HAL_FILE_TRANSFER_CHUNK_SIZE_DEFAULT,
	};
	return iree_hal_device_queue_read_streaming(base_device, queue_affinity,
		wait_semaphore_list, signal_semaphore_list, source_file,
		source_offset, target_buffer, target_offset, length, flags,
		options);
}

static iree_status_t iree_hal_cuda_new_device_queue_write(
	iree_hal_device_t *base_device, iree_hal_queue_affinity_t queue_affinity,
	const iree_hal_semaphore_list_t wait_semaphore_list,
	const iree_hal_semaphore_list_t signal_semaphore_list,
	iree_hal_buffer_t *source_buffer, iree_device_size_t source_offset,
	iree_hal_file_t *target_file, uint64_t target_offset,
	iree_device_size_t length, iree_hal_write_flags_t flags) {
	iree_hal_file_transfer_options_t options = {
		.chunk_count = IREE_HAL_FILE_TRANSFER_CHUNK_COUNT_DEFAULT,
		.chunk_size = IREE_HAL_FILE_TRANSFER_CHUNK_SIZE_DEFAULT,
	};
	return iree_hal_device_queue_write_streaming(base_device, queue_affinity,
		wait_semaphore_list, signal_semaphore_list, source_buffer,
		source_offset, target_file, target_offset, length, flags,
		options);
}

static iree_status_t iree_hal_cuda_new_device_queue_host_call(
	iree_hal_device_t *base_device, iree_hal_queue_affinity_t queue_affinity,
	const iree_hal_semaphore_list_t wait_semaphore_list,
	const iree_hal_semaphore_list_t signal_semaphore_list,
	iree_hal_host_call_t call, const uint64_t args[4],
	iree_hal_host_call_flags_t flags) {
	return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
		"cuda_new queue_host_call not yet implemented");
}

static iree_status_t iree_hal_cuda_new_device_queue_execute(
	iree_hal_device_t *base_device, iree_hal_queue_affinity_t queue_affinity,
	const iree_hal_semaphore_list_t wait_semaphore_list,
	const iree_hal_semaphore_list_t signal_semaphore_list,
	iree_hal_command_buffer_t *command_buffer,
	iree_hal_buffer_binding_table_t binding_table,
	iree_hal_execute_flags_t flags) {
	iree_hal_cuda_new_logical_device_t *device =
		iree_hal_cuda_new_logical_device_cast(base_device);
	IREE_TRACE_ZONE_BEGIN(z0);

	IREE_RETURN_AND_END_ZONE_IF_ERROR(z0,
		IREE_CURESULT_TO_STATUS_NEW(device->syms,
			cuCtxSetCurrent(device->cu_context), "cuCtxSetCurrent"));

	IREE_RETURN_AND_END_ZONE_IF_ERROR(z0,
		iree_hal_semaphore_list_wait(wait_semaphore_list,
			iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

	iree_status_t status = iree_ok_status();

	if (command_buffer) {
		if (iree_hal_deferred_command_buffer_isa(command_buffer)) {
			// Create a transient stream command buffer and replay into it.
			iree_hal_command_buffer_t *stream_cb = NULL;
			status = iree_hal_cuda_new_stream_command_buffer_create(
				device->device_allocator, device->syms, device->cu_context,
				IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
				IREE_HAL_COMMAND_CATEGORY_ANY, /*binding_capacity=*/0,
				device->dispatch_stream, &device->block_pool,
				device->host_allocator, &stream_cb);
			if (iree_status_is_ok(status)) {
				status = iree_hal_deferred_command_buffer_apply(
					command_buffer, stream_cb, binding_table);
			}
			if (iree_status_is_ok(status)) {
				status = IREE_CURESULT_TO_STATUS_NEW(device->syms,
					cuStreamSynchronize(device->dispatch_stream),
					"cuStreamSynchronize");
			}
			if (stream_cb) {
				iree_hal_command_buffer_release(stream_cb);
			}
		} else if (iree_hal_cuda_new_stream_command_buffer_isa(
					   command_buffer)) {
			// Already executed inline on the stream; just sync.
			status = IREE_CURESULT_TO_STATUS_NEW(device->syms,
				cuStreamSynchronize(device->dispatch_stream),
				"cuStreamSynchronize");
		} else {
			status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
				"unsupported command buffer type");
		}
	}

	if (iree_status_is_ok(status)) {
		status = iree_hal_semaphore_list_signal(signal_semaphore_list,
			/*frontier=*/NULL);
	} else {
		iree_hal_semaphore_list_fail(signal_semaphore_list,
			iree_status_clone(status));
	}

	IREE_TRACE_ZONE_END(z0);
	return status;
}

static iree_status_t iree_hal_cuda_new_device_queue_flush(
	iree_hal_device_t *base_device,
	iree_hal_queue_affinity_t queue_affinity) {
	return iree_ok_status();
}

static iree_status_t iree_hal_cuda_new_device_profiling_begin(
	iree_hal_device_t *base_device,
	const iree_hal_device_profiling_options_t *options) {
	return iree_ok_status();
}

static iree_status_t iree_hal_cuda_new_device_profiling_flush(
	iree_hal_device_t *base_device) {
	return iree_ok_status();
}

static iree_status_t iree_hal_cuda_new_device_profiling_end(
	iree_hal_device_t *base_device) {
	return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Vtable
//===----------------------------------------------------------------------===//

static const iree_hal_device_vtable_t iree_hal_cuda_new_device_vtable = {
	.destroy = iree_hal_cuda_new_device_destroy,
	.id = iree_hal_cuda_new_device_id,
	.host_allocator = iree_hal_cuda_new_device_host_allocator,
	.device_allocator = iree_hal_cuda_new_device_allocator,
	.replace_device_allocator = iree_hal_cuda_new_device_replace_allocator,
	.replace_channel_provider =
		iree_hal_cuda_new_device_replace_channel_provider,
	.trim = iree_hal_cuda_new_device_trim,
	.query_i64 = iree_hal_cuda_new_device_query_i64,
	.query_capabilities = iree_hal_cuda_new_device_query_capabilities,
	.topology_info = iree_hal_cuda_new_device_topology_info,
	.refine_topology_edge = iree_hal_cuda_new_device_refine_topology_edge,
	.assign_topology_info = iree_hal_cuda_new_device_assign_topology_info,
	.create_channel = iree_hal_cuda_new_device_create_channel,
	.create_command_buffer = iree_hal_cuda_new_device_create_command_buffer,
	.create_event = iree_hal_cuda_new_device_create_event,
	.create_executable_cache =
		iree_hal_cuda_new_device_create_executable_cache,
	.import_file = iree_hal_cuda_new_device_import_file,
	.create_semaphore = iree_hal_cuda_new_device_create_semaphore,
	.query_semaphore_compatibility =
		iree_hal_cuda_new_device_query_semaphore_compatibility,
	.queue_alloca = iree_hal_cuda_new_device_queue_alloca,
	.queue_dealloca = iree_hal_cuda_new_device_queue_dealloca,
	.queue_fill = iree_hal_device_queue_emulated_fill,
	.queue_update = iree_hal_device_queue_emulated_update,
	.queue_copy = iree_hal_device_queue_emulated_copy,
	.queue_read = iree_hal_cuda_new_device_queue_read,
	.queue_write = iree_hal_cuda_new_device_queue_write,
	.queue_host_call = iree_hal_cuda_new_device_queue_host_call,
	.queue_dispatch = iree_hal_device_queue_emulated_dispatch,
	.queue_execute = iree_hal_cuda_new_device_queue_execute,
	.queue_flush = iree_hal_cuda_new_device_queue_flush,
	.profiling_begin = iree_hal_cuda_new_device_profiling_begin,
	.profiling_flush = iree_hal_cuda_new_device_profiling_flush,
	.profiling_end = iree_hal_cuda_new_device_profiling_end,
};
