// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "qnn_device.h"

#include <inttypes.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iree/async/util/proactor_pool.h"
#include "iree/base/internal/arena.h"
#include "iree/hal/local/inline_command_buffer.h"
#include "iree/hal/utils/deferred_command_buffer.h"
#include "iree/hal/utils/file_registry.h"
#include "iree/hal/utils/file_transfer.h"
#include "iree/hal/utils/queue_emulation.h"

#include "qnn_command_buffer.h"
#include "qnn_executable.h"
#include "qnn_executable_cache.h"
#include "qnn_semaphore.h"

// QNN SDK — needed for the QnnInterface vtable +
// Qnn_BackendHandle/DeviceHandle.
#include "QnnBackend.h"
#include "QnnCommon.h"
#include "QnnDevice.h"
#include "QnnInterface.h"

#define IREE_HAL_QNN_COMMAND_ARENA_BLOCK_SIZE (32 * 1024)

typedef struct iree_hal_qnn_device_t {
	iree_hal_resource_t resource;
	iree_string_view_t identifier;

	iree_allocator_t host_allocator;
	iree_hal_allocator_t *device_allocator;
	iree_hal_channel_provider_t *channel_provider;

	// Backend identity + dlopen'd libQnn<Backend>.so.
	iree_hal_qnn_backend_t backend;
	iree_dynamic_library_t *backend_lib;

	// Borrowed pointer to the driver. Used to fetch the cached
	// QnnSystemInterface when creating executables; not retained because
	// the device lives strictly inside the driver's lifetime.
	iree_hal_driver_t *parent_driver;

	// QNN SDK vtable + per-device QNN handles. Populated lazily on the first
	// executable_cache_create call so we don't pay the QnnBackend_create cost
	// for devices that never load executables (e.g. probe-only).
	const QnnInterface_t *qnn_interface;
	Qnn_LogHandle_t qnn_log_handle;
	Qnn_BackendHandle_t qnn_backend_handle;
	Qnn_DeviceHandle_t qnn_device_handle;
	bool qnn_handles_initialized;

	// Async I/O proactor pool retained from create_params; semaphores acquire a
	// proactor from this pool. Mirrors task_device's pattern.
	iree_async_proactor_pool_t *proactor_pool;
	iree_async_proactor_t *proactor;

	// Topology info populated by the runtime. Default-initialised to all-zero.
	iree_hal_device_topology_info_t topology_info;

	// Arena block pool used by deferred command buffers.
	iree_arena_block_pool_t command_block_pool;

	// Serializes queue execution. QNN's QnnGraph_execute is thread-unsafe per
	// graph handle; we serialise per-device.
	iree_slim_mutex_t queue_mutex;

	// + trailing identifier storage.
} iree_hal_qnn_device_t;

static const iree_hal_device_vtable_t iree_hal_qnn_device_vtable;

static void iree_hal_qnn_log_callback(
	const char *fmt, QnnLog_Level_t level, uint64_t timestamp, va_list args) {
	(void)timestamp;
	fprintf(stderr, "QNN[%d] ", (int)level);
	vfprintf(stderr, fmt, args);
	fprintf(stderr, "\n");
}

static QnnLog_Level_t iree_hal_qnn_parse_log_level(const char *value) {
	if (!value || !value[0])
		return QNN_LOG_LEVEL_WARN;
	if (strcmp(value, "error") == 0)
		return QNN_LOG_LEVEL_ERROR;
	if (strcmp(value, "warn") == 0)
		return QNN_LOG_LEVEL_WARN;
	if (strcmp(value, "info") == 0)
		return QNN_LOG_LEVEL_INFO;
	if (strcmp(value, "verbose") == 0)
		return QNN_LOG_LEVEL_VERBOSE;
	if (strcmp(value, "debug") == 0)
		return QNN_LOG_LEVEL_DEBUG;
	return QNN_LOG_LEVEL_WARN;
}

static iree_hal_qnn_device_t *iree_hal_qnn_device_cast(
	iree_hal_device_t *base_value) {
	IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_qnn_device_vtable);
	return (iree_hal_qnn_device_t *)base_value;
}

// -----------------------------------------------------------------------------
// vtable methods
// -----------------------------------------------------------------------------

static void iree_hal_qnn_device_destroy(iree_hal_device_t *base_device) {
	iree_hal_qnn_device_t *device = iree_hal_qnn_device_cast(base_device);
	IREE_TRACE_ZONE_BEGIN(z0);

	iree_hal_allocator_release(device->device_allocator);
	iree_hal_channel_provider_release(device->channel_provider);

	// Tear down per-device QNN handles in reverse order of construction.
	if (device->qnn_handles_initialized && device->qnn_interface) {
		if (device->qnn_device_handle) {
			Qnn_ErrorHandle_t rc =
				device->qnn_interface->QNN_INTERFACE_VER_NAME.deviceFree(
					device->qnn_device_handle);
			(void)rc;
		}
		if (device->qnn_backend_handle) {
			Qnn_ErrorHandle_t rc =
				device->qnn_interface->QNN_INTERFACE_VER_NAME.backendFree(
					device->qnn_backend_handle);
			(void)rc;
		}
		if (device->qnn_log_handle &&
			device->qnn_interface->QNN_INTERFACE_VER_NAME.logFree) {
			Qnn_ErrorHandle_t rc =
				device->qnn_interface->QNN_INTERFACE_VER_NAME.logFree(
					device->qnn_log_handle);
			(void)rc;
		}
	}

	// The proactor pool is borrowed (one per session); release the retain.
	if (device->proactor_pool) {
		iree_async_proactor_pool_release(device->proactor_pool);
	}

	// The backend_lib handle is shared with the driver. Release here decrements
	// the refcount; the driver keeps its own ref so a second device on the same
	// backend reuses the same load.
	if (device->backend_lib) {
		iree_dynamic_library_release(device->backend_lib);
	}

	// Release the parent driver retain (matches the retain in device_create).
	if (device->parent_driver) {
		iree_hal_driver_release(device->parent_driver);
		device->parent_driver = NULL;
	}

	iree_slim_mutex_deinitialize(&device->queue_mutex);
	iree_arena_block_pool_deinitialize(&device->command_block_pool);

	iree_allocator_free(device->host_allocator, device);
	IREE_TRACE_ZONE_END(z0);
}

static iree_string_view_t iree_hal_qnn_device_id(
	iree_hal_device_t *base_device) {
	iree_hal_qnn_device_t *device = iree_hal_qnn_device_cast(base_device);
	return device->identifier;
}

static iree_allocator_t iree_hal_qnn_device_host_allocator(
	iree_hal_device_t *base_device) {
	iree_hal_qnn_device_t *device = iree_hal_qnn_device_cast(base_device);
	return device->host_allocator;
}

static iree_hal_allocator_t *iree_hal_qnn_device_allocator(
	iree_hal_device_t *base_device) {
	iree_hal_qnn_device_t *device = iree_hal_qnn_device_cast(base_device);
	return device->device_allocator;
}

static void iree_hal_qnn_device_replace_device_allocator(
	iree_hal_device_t *base_device, iree_hal_allocator_t *new_allocator) {
	iree_hal_qnn_device_t *device = iree_hal_qnn_device_cast(base_device);
	iree_hal_allocator_retain(new_allocator);
	iree_hal_allocator_release(device->device_allocator);
	device->device_allocator = new_allocator;
}

static void iree_hal_qnn_device_replace_channel_provider(
	iree_hal_device_t *base_device, iree_hal_channel_provider_t *new_provider) {
	iree_hal_qnn_device_t *device = iree_hal_qnn_device_cast(base_device);
	iree_hal_channel_provider_retain(new_provider);
	iree_hal_channel_provider_release(device->channel_provider);
	device->channel_provider = new_provider;
}

static iree_status_t iree_hal_qnn_device_trim(iree_hal_device_t *base_device) {
	iree_hal_qnn_device_t *device = iree_hal_qnn_device_cast(base_device);
	return iree_hal_allocator_trim(device->device_allocator);
}

static iree_status_t iree_hal_qnn_device_query_i64(
	iree_hal_device_t *base_device, iree_string_view_t category,
	iree_string_view_t key, int64_t *out_value) {
	iree_hal_qnn_device_t *device = iree_hal_qnn_device_cast(base_device);
	*out_value = 0;

	if (iree_string_view_equal(category, IREE_SV("hal.device.id"))) {
		// The device identifier is the backend-qualified form ("qnn-gpu",
		// "qnn-htp"...), but #hal.device.target<"qnn", ...> in compiled
		// VMFBs queries only with the backend root ("qnn"). Match either
		// the full identifier or its "qnn" prefix so both work.
		if (iree_string_view_match_pattern(device->identifier, key) ||
			iree_string_view_starts_with(device->identifier, key)) {
			*out_value = 1;
		} else {
			*out_value = 0;
		}
		return iree_ok_status();
	}

	if (iree_string_view_equal(category, IREE_SV("hal.executable.format"))) {
		// Advertise support for: legacy "qnn-context-binary" and
		// backend-agnostic "qnn-graph", plus the suffixed form matching THIS
		// device's backend ("qnn-graph-hta" / "-gpu" / "-htp"). Refusing the
		// other backends' suffixes is what makes a multi-variant VMFB load
		// cleanly: the compiler-emitted __init queries each variant's format
		// against each bound device and skips ones that say no here.
		if (iree_string_view_equal(key, IREE_SV("qnn-context-binary")) ||
			iree_string_view_equal(key, IREE_SV("qnn-graph"))) {
			*out_value = 1;
			return iree_ok_status();
		}
		iree_string_view_t my_suffix = iree_string_view_empty();
		switch (device->backend) {
			case IREE_HAL_QNN_BACKEND_HTA:
				my_suffix = IREE_SV("qnn-graph-hta");
				break;
			case IREE_HAL_QNN_BACKEND_GPU:
				my_suffix = IREE_SV("qnn-graph-gpu");
				break;
			case IREE_HAL_QNN_BACKEND_HTP:
				my_suffix = IREE_SV("qnn-graph-htp");
				break;
			default:
				break;
		}
		*out_value = iree_string_view_equal(key, my_suffix) ? 1 : 0;
		return iree_ok_status();
	}

	if (iree_string_view_equal(category, IREE_SV("hal.device")) &&
		iree_string_view_equal(key, IREE_SV("concurrency"))) {
		// QNN dispatches are serialised per device; advertise concurrency=1.
		*out_value = 1;
		return iree_ok_status();
	}
	if (iree_string_view_equal(category, IREE_SV("hal.dispatch")) &&
		iree_string_view_equal(key, IREE_SV("concurrency"))) {
		*out_value = 1;
		return iree_ok_status();
	}

	return iree_make_status(IREE_STATUS_NOT_FOUND,
		"unknown device query key '%.*s :: %.*s'", (int)category.size,
		category.data, (int)key.size, key.data);
}

static iree_status_t iree_hal_qnn_device_create_channel(
	iree_hal_device_t *base_device, iree_hal_queue_affinity_t queue_affinity,
	iree_hal_channel_params_t params, iree_hal_channel_t **out_channel) {
	(void)base_device;
	(void)queue_affinity;
	(void)params;
	(void)out_channel;
	return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
		"collective channels are not implemented for QNN");
}

static iree_status_t iree_hal_qnn_device_create_command_buffer(
	iree_hal_device_t *base_device, iree_hal_command_buffer_mode_t mode,
	iree_hal_command_category_t command_categories,
	iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
	iree_hal_command_buffer_t **out_command_buffer) {
	iree_hal_qnn_device_t *device = iree_hal_qnn_device_cast(base_device);
	if (iree_all_bits_set(
			mode, IREE_HAL_COMMAND_BUFFER_MODE_ALLOW_INLINE_EXECUTION)) {
		return iree_hal_inline_command_buffer_create(
			iree_hal_device_allocator(base_device), mode, command_categories,
			queue_affinity, binding_capacity, device->host_allocator,
			out_command_buffer);
	}
	return iree_hal_deferred_command_buffer_create(
		iree_hal_device_allocator(base_device), mode, command_categories,
		queue_affinity, binding_capacity, &device->command_block_pool,
		device->host_allocator, out_command_buffer);
}

static iree_status_t iree_hal_qnn_device_create_event(
	iree_hal_device_t *base_device, iree_hal_queue_affinity_t queue_affinity,
	iree_hal_event_flags_t flags, iree_hal_event_t **out_event) {
	(void)base_device;
	(void)queue_affinity;
	(void)flags;
	(void)out_event;
	return iree_make_status(
		IREE_STATUS_UNIMPLEMENTED, "events are not yet implemented for QNN");
}

// Lazily acquires the QnnInterface vtable + per-device Qnn{Backend,Device}
// handles. Idempotent — each call after the first is a no-op.
static iree_status_t iree_hal_qnn_device_ensure_qnn_handles(
	iree_hal_qnn_device_t *device) {
	if (device->qnn_handles_initialized) {
		return iree_ok_status();
	}

	// Look up QnnInterface_getProviders on the dlopen'd backend lib.
	typedef Qnn_ErrorHandle_t (*QnnInterface_getProviders_fn_t)(
		const QnnInterface_t ***providers, uint32_t *num_providers);
	QnnInterface_getProviders_fn_t get_providers = NULL;
	IREE_RETURN_IF_ERROR(iree_dynamic_library_lookup_symbol(device->backend_lib,
		"QnnInterface_getProviders", (void **)&get_providers));

	const QnnInterface_t **providers = NULL;
	uint32_t num_providers = 0;
	Qnn_ErrorHandle_t rc = get_providers(&providers, &num_providers);
	if (rc != QNN_SUCCESS || num_providers == 0 || providers == NULL) {
		return iree_make_status(IREE_STATUS_INTERNAL,
			"QnnInterface_getProviders rc=%lld num=%u", (long long)rc,
			num_providers);
	}

	// Use the first provider — every backend lib we ship is single-provider.
	device->qnn_interface = providers[0];

	const char *log_level = getenv("IREE_HAL_QNN_LOG_LEVEL");
	if (log_level && device->qnn_interface->QNN_INTERFACE_VER_NAME.logCreate) {
		Qnn_ErrorHandle_t log_rc =
			device->qnn_interface->QNN_INTERFACE_VER_NAME.logCreate(
				iree_hal_qnn_log_callback,
				iree_hal_qnn_parse_log_level(log_level),
				&device->qnn_log_handle);
		if (log_rc != QNN_SUCCESS) {
			device->qnn_log_handle = NULL;
		}
	}

	// Create the QnnBackend (borrows the provider library; no config).
	rc = device->qnn_interface->QNN_INTERFACE_VER_NAME.backendCreate(
		device->qnn_log_handle, /*config=*/NULL, &device->qnn_backend_handle);
	if (rc != QNN_SUCCESS) {
		device->qnn_interface = NULL;
		return iree_make_status(
			IREE_STATUS_INTERNAL, "QnnBackend_create rc=%lld", (long long)rc);
	}

	// Create the QnnDevice. Some backends (CPU) don't support
	// device_create — pass through silently if unimplemented.
	if (device->qnn_interface->QNN_INTERFACE_VER_NAME.deviceCreate) {
		rc = device->qnn_interface->QNN_INTERFACE_VER_NAME.deviceCreate(
			device->qnn_log_handle, /*config=*/NULL,
			&device->qnn_device_handle);
		if (rc != QNN_SUCCESS) {
			// Non-fatal — many backends don't have a separate device handle.
			device->qnn_device_handle = NULL;
		}
	}

	device->qnn_handles_initialized = true;
	return iree_ok_status();
}

static iree_status_t iree_hal_qnn_device_create_executable_cache(
	iree_hal_device_t *base_device, iree_string_view_t identifier,
	iree_hal_executable_cache_t **out_executable_cache) {
	iree_hal_qnn_device_t *device = iree_hal_qnn_device_cast(base_device);
	IREE_RETURN_IF_ERROR(iree_hal_qnn_device_ensure_qnn_handles(device));
	// Lazy-load libQnnSystem.so via the (retained) parent driver — first
	// call pays for the dlopen + getProviders, subsequent calls share the
	// cached pointer. Earlier this segfaulted because the device wasn't
	// retaining the driver; that's fixed in device_create. Failure is
	// non-fatal: executable_create falls back to a per-call dlopen.
	const void *system_interface = NULL;
	if (device->parent_driver) {
		iree_status_t s = iree_hal_qnn_driver_get_system_interface(
			device->parent_driver, &system_interface);
		if (!iree_status_is_ok(s))
			iree_status_ignore(s);
	}
	return iree_hal_qnn_executable_cache_create(identifier, device->backend,
		(iree_hal_qnn_interface_t)device->qnn_interface,
		(iree_hal_qnn_backend_handle_t)device->qnn_backend_handle,
		(iree_hal_qnn_device_handle_t)device->qnn_device_handle,
		system_interface, device->host_allocator, out_executable_cache);
}

static iree_status_t iree_hal_qnn_device_import_file(
	iree_hal_device_t *base_device, iree_hal_queue_affinity_t queue_affinity,
	iree_hal_memory_access_t access, iree_io_file_handle_t *handle,
	iree_hal_external_file_flags_t flags, iree_hal_file_t **out_file) {
	(void)flags;
	return iree_hal_file_from_handle(iree_hal_device_allocator(base_device),
		queue_affinity, access, handle,
		/*proactor=*/NULL, iree_hal_device_host_allocator(base_device),
		out_file);
}

static iree_status_t iree_hal_qnn_device_create_semaphore(
	iree_hal_device_t *base_device, iree_hal_queue_affinity_t queue_affinity,
	uint64_t initial_value, iree_hal_semaphore_flags_t flags,
	iree_hal_semaphore_t **out_semaphore) {
	(void)queue_affinity;
	(void)flags;
	iree_hal_qnn_device_t *device = iree_hal_qnn_device_cast(base_device);
	return iree_hal_qnn_semaphore_create(
		device->proactor, initial_value, device->host_allocator, out_semaphore);
}

// New vtable methods present in the current iree_hal_device_vtable_t (mirroring
// task_device.c). All four return ok with empty/default state — QNN devices
// don't have NUMA affinity or P2P topology to report.
static iree_status_t iree_hal_qnn_device_query_capabilities(
	iree_hal_device_t *base_device,
	iree_hal_device_capabilities_t *out_capabilities) {
	(void)base_device;
	memset(out_capabilities, 0, sizeof(*out_capabilities));
	return iree_ok_status();
}

static const iree_hal_device_topology_info_t *iree_hal_qnn_device_topology_info(
	iree_hal_device_t *base_device) {
	iree_hal_qnn_device_t *device = iree_hal_qnn_device_cast(base_device);
	return &device->topology_info;
}

static iree_status_t iree_hal_qnn_device_refine_topology_edge(
	iree_hal_device_t *src_device, iree_hal_device_t *dst_device,
	iree_hal_topology_edge_t *edge) {
	(void)src_device;
	(void)dst_device;
	(void)edge;
	return iree_ok_status();
}

static iree_status_t iree_hal_qnn_device_assign_topology_info(
	iree_hal_device_t *base_device,
	const iree_hal_device_topology_info_t *topology_info) {
	iree_hal_qnn_device_t *device = iree_hal_qnn_device_cast(base_device);
	device->topology_info = *topology_info;
	return iree_ok_status();
}

static iree_hal_semaphore_compatibility_t
iree_hal_qnn_device_query_semaphore_compatibility(
	iree_hal_device_t *base_device, iree_hal_semaphore_t *semaphore) {
	(void)base_device;
	(void)semaphore;
	return IREE_HAL_SEMAPHORE_COMPATIBILITY_HOST_ONLY;
}

static iree_status_t iree_hal_qnn_device_queue_alloca(
	iree_hal_device_t *base_device, iree_hal_queue_affinity_t queue_affinity,
	const iree_hal_semaphore_list_t wait_semaphore_list,
	const iree_hal_semaphore_list_t signal_semaphore_list,
	iree_hal_allocator_pool_t pool, iree_hal_buffer_params_t params,
	iree_device_size_t allocation_size, iree_hal_alloca_flags_t flags,
	iree_hal_buffer_t **IREE_RESTRICT out_buffer) {
	(void)queue_affinity;
	(void)pool;
	(void)flags;
	IREE_RETURN_IF_ERROR(iree_hal_semaphore_list_wait(wait_semaphore_list,
		iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
	IREE_RETURN_IF_ERROR(iree_hal_allocator_allocate_buffer(
		iree_hal_device_allocator(base_device), params, allocation_size,
		out_buffer));
	IREE_RETURN_IF_ERROR(iree_hal_semaphore_list_signal(
		signal_semaphore_list, /*frontier=*/NULL));
	return iree_ok_status();
}

static iree_status_t iree_hal_qnn_device_queue_dealloca(
	iree_hal_device_t *base_device, iree_hal_queue_affinity_t queue_affinity,
	const iree_hal_semaphore_list_t wait_semaphore_list,
	const iree_hal_semaphore_list_t signal_semaphore_list,
	iree_hal_buffer_t *buffer, iree_hal_dealloca_flags_t flags) {
	(void)buffer;
	(void)flags;
	return iree_hal_device_queue_barrier(base_device, queue_affinity,
		wait_semaphore_list, signal_semaphore_list, IREE_HAL_EXECUTE_FLAG_NONE);
}

static iree_status_t iree_hal_qnn_device_queue_read(
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
		wait_semaphore_list, signal_semaphore_list, source_file, source_offset,
		target_buffer, target_offset, length, flags, options);
}

static iree_status_t iree_hal_qnn_device_queue_write(
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
		source_offset, target_file, target_offset, length, flags, options);
}

static iree_status_t iree_hal_qnn_device_queue_host_call(
	iree_hal_device_t *base_device, iree_hal_queue_affinity_t queue_affinity,
	const iree_hal_semaphore_list_t wait_semaphore_list,
	const iree_hal_semaphore_list_t signal_semaphore_list,
	iree_hal_host_call_t call, const uint64_t args[4],
	iree_hal_host_call_flags_t flags) {
	IREE_RETURN_IF_ERROR(iree_hal_semaphore_list_wait(wait_semaphore_list,
		iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

	const bool is_nonblocking =
		iree_any_bit_set(flags, IREE_HAL_HOST_CALL_FLAG_NON_BLOCKING);
	if (is_nonblocking) {
		IREE_RETURN_IF_ERROR(iree_hal_semaphore_list_signal(
			signal_semaphore_list, /*frontier=*/NULL));
	}

	iree_hal_host_call_context_t context = {
		.device = base_device,
		.queue_affinity = queue_affinity,
		.signal_semaphore_list = is_nonblocking
			? iree_hal_semaphore_list_empty()
			: signal_semaphore_list,
	};
	iree_status_t call_status = call.fn(call.user_data, args, &context);
	if (is_nonblocking || iree_status_is_deferred(call_status)) {
		return iree_ok_status();
	}
	if (iree_status_is_ok(call_status)) {
		return iree_hal_semaphore_list_signal(
			signal_semaphore_list, /*frontier=*/NULL);
	}
	if (!is_nonblocking) {
		iree_hal_semaphore_list_fail(signal_semaphore_list, call_status);
	} else {
		iree_status_ignore(call_status);
	}
	return iree_ok_status();
}

static iree_status_t iree_hal_qnn_device_apply_deferred_command_buffer(
	iree_hal_qnn_device_t *device, iree_hal_command_buffer_t *command_buffer,
	iree_hal_buffer_binding_table_t binding_table) {
	if (!command_buffer || iree_hal_inline_command_buffer_isa(command_buffer)) {
		return iree_ok_status();
	}
	if (!iree_hal_deferred_command_buffer_isa(command_buffer)) {
		return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
			"unsupported command buffer implementation");
	}

	// For dispatch-category buffers, use the QNN command buffer that
	// translates dispatch → QnnGraph_execute. For non-dispatch buffers
	// (e.g. transfer-only) we still need a target; use the standard inline
	// buffer since QNN doesn't need to intercept fill/copy.
	iree_hal_command_buffer_mode_t target_mode =
		iree_hal_command_buffer_mode(command_buffer) |
		IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT |
		IREE_HAL_COMMAND_BUFFER_MODE_ALLOW_INLINE_EXECUTION |
		(iree_hal_buffer_binding_table_is_empty(binding_table)
				? IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED
				: 0);
	iree_hal_command_category_t target_categories =
		iree_hal_command_buffer_allowed_categories(command_buffer);

	iree_hal_command_buffer_t *target_buffer = NULL;
	bool is_dispatch =
		iree_any_bit_set(target_categories, IREE_HAL_COMMAND_CATEGORY_DISPATCH);
	if (is_dispatch && device->qnn_handles_initialized) {
		IREE_RETURN_IF_ERROR(iree_hal_qnn_command_buffer_create(
			device->device_allocator, target_mode, target_categories,
			IREE_HAL_QUEUE_AFFINITY_ANY, /*binding_capacity=*/0,
			device->qnn_interface, device->host_allocator, &target_buffer));
	} else {
		iree_host_size_t storage_size = iree_hal_inline_command_buffer_size(
			target_mode, /*binding_capacity=*/0);
		iree_byte_span_t storage =
			iree_make_byte_span(iree_alloca(storage_size), storage_size);
		IREE_RETURN_IF_ERROR(iree_hal_inline_command_buffer_initialize(
			device->device_allocator, target_mode, target_categories,
			IREE_HAL_QUEUE_AFFINITY_ANY, /*binding_capacity=*/0,
			device->host_allocator, storage, &target_buffer));
	}

	iree_status_t status = iree_hal_deferred_command_buffer_apply(
		command_buffer, target_buffer, binding_table);

	if (is_dispatch && device->qnn_handles_initialized) {
		iree_hal_command_buffer_release(target_buffer);
	} else {
		iree_hal_inline_command_buffer_deinitialize(target_buffer);
	}
	return status;
}

static iree_status_t iree_hal_qnn_device_queue_execute(
	iree_hal_device_t *base_device, iree_hal_queue_affinity_t queue_affinity,
	const iree_hal_semaphore_list_t wait_semaphore_list,
	const iree_hal_semaphore_list_t signal_semaphore_list,
	iree_hal_command_buffer_t *command_buffer,
	iree_hal_buffer_binding_table_t binding_table,
	iree_hal_execute_flags_t flags) {
	(void)queue_affinity;
	(void)flags;
	iree_hal_qnn_device_t *device = iree_hal_qnn_device_cast(base_device);

	// PR-3: dispatch-category command buffers replay against the QNN
	// command buffer (qnn_command_buffer.c) which translates dispatches into
	// QnnGraph_execute calls.

	IREE_RETURN_IF_ERROR(iree_hal_semaphore_list_wait(wait_semaphore_list,
		iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

	iree_slim_mutex_lock(&device->queue_mutex);
	iree_status_t status = iree_hal_qnn_device_apply_deferred_command_buffer(
		device, command_buffer, binding_table);
	if (iree_status_is_ok(status)) {
		status = iree_hal_semaphore_list_signal(
			signal_semaphore_list, /*frontier=*/NULL);
	} else {
		iree_hal_semaphore_list_fail(
			signal_semaphore_list, iree_status_clone(status));
	}
	iree_slim_mutex_unlock(&device->queue_mutex);
	return status;
}

static iree_status_t iree_hal_qnn_device_queue_flush(
	iree_hal_device_t *base_device, iree_hal_queue_affinity_t queue_affinity) {
	(void)queue_affinity;
	(void)base_device;
	// QnnGraph_execute is synchronous so nothing to flush at this layer.
	return iree_ok_status();
}

static iree_status_t iree_hal_qnn_device_profiling_begin(
	iree_hal_device_t *base_device,
	const iree_hal_device_profiling_options_t *options) {
	(void)base_device;
	(void)options;
	return iree_ok_status();
}

static iree_status_t iree_hal_qnn_device_profiling_flush(
	iree_hal_device_t *base_device) {
	(void)base_device;
	return iree_ok_status();
}

static iree_status_t iree_hal_qnn_device_profiling_end(
	iree_hal_device_t *base_device) {
	(void)base_device;
	return iree_ok_status();
}

static const iree_hal_device_vtable_t iree_hal_qnn_device_vtable = {
	.destroy = iree_hal_qnn_device_destroy,
	.id = iree_hal_qnn_device_id,
	.host_allocator = iree_hal_qnn_device_host_allocator,
	.device_allocator = iree_hal_qnn_device_allocator,
	.replace_device_allocator = iree_hal_qnn_device_replace_device_allocator,
	.replace_channel_provider = iree_hal_qnn_device_replace_channel_provider,
	.trim = iree_hal_qnn_device_trim,
	.query_i64 = iree_hal_qnn_device_query_i64,
	.query_capabilities = iree_hal_qnn_device_query_capabilities,
	.topology_info = iree_hal_qnn_device_topology_info,
	.refine_topology_edge = iree_hal_qnn_device_refine_topology_edge,
	.assign_topology_info = iree_hal_qnn_device_assign_topology_info,
	.create_channel = iree_hal_qnn_device_create_channel,
	.create_command_buffer = iree_hal_qnn_device_create_command_buffer,
	.create_event = iree_hal_qnn_device_create_event,
	.create_executable_cache = iree_hal_qnn_device_create_executable_cache,
	.import_file = iree_hal_qnn_device_import_file,
	.create_semaphore = iree_hal_qnn_device_create_semaphore,
	.query_semaphore_compatibility =
		iree_hal_qnn_device_query_semaphore_compatibility,
	.queue_alloca = iree_hal_qnn_device_queue_alloca,
	.queue_dealloca = iree_hal_qnn_device_queue_dealloca,
	.queue_fill = iree_hal_device_queue_emulated_fill,
	.queue_update = iree_hal_device_queue_emulated_update,
	.queue_copy = iree_hal_device_queue_emulated_copy,
	.queue_read = iree_hal_qnn_device_queue_read,
	.queue_write = iree_hal_qnn_device_queue_write,
	.queue_host_call = iree_hal_qnn_device_queue_host_call,
	.queue_dispatch = iree_hal_device_queue_emulated_dispatch,
	.queue_execute = iree_hal_qnn_device_queue_execute,
	.queue_flush = iree_hal_qnn_device_queue_flush,
	.profiling_begin = iree_hal_qnn_device_profiling_begin,
	.profiling_flush = iree_hal_qnn_device_profiling_flush,
	.profiling_end = iree_hal_qnn_device_profiling_end,
};

IREE_API_EXPORT iree_status_t iree_hal_qnn_device_create(
	iree_string_view_t identifier, iree_hal_qnn_backend_t backend,
	iree_dynamic_library_t *backend_lib, iree_hal_driver_t *parent_driver,
	const iree_hal_device_create_params_t *create_params,
	iree_allocator_t host_allocator, iree_hal_device_t **out_device) {
	IREE_ASSERT_ARGUMENT(create_params);
	IREE_ASSERT_ARGUMENT(create_params->proactor_pool);
	IREE_ASSERT_ARGUMENT(out_device);
	IREE_TRACE_ZONE_BEGIN(z0);
	*out_device = NULL;

	iree_hal_qnn_device_t *device = NULL;
	const iree_host_size_t total_size = sizeof(*device) + identifier.size;
	IREE_RETURN_AND_END_ZONE_IF_ERROR(z0,
		iree_allocator_malloc(host_allocator, total_size, (void **)&device));
	memset(device, 0, total_size);

	iree_hal_resource_initialize(
		&iree_hal_qnn_device_vtable, &device->resource);
	iree_string_view_append_to_buffer(identifier, &device->identifier,
		(char *)device + total_size - identifier.size);
	device->host_allocator = host_allocator;
	device->backend = backend;
	device->backend_lib = backend_lib;
	if (backend_lib) {
		iree_dynamic_library_retain(backend_lib);
	}
	// Parent driver pointer — retained so the device can hop back for the
	// cached QnnSystemInterface even after the original driver-create caller
	// has released its ref. Released in device_destroy. NULL is fine —
	// executable creation falls back to a per-executable dlopen of
	// libQnnSystem.so when the device wasn't given a parent.
	device->parent_driver = parent_driver;
	if (parent_driver) {
		iree_hal_driver_retain(parent_driver);
	}

	// Retain the proactor pool and pick the first proactor (no NUMA affinity
	// for QNN backends — the entire SoC shares L3).
	device->proactor_pool = create_params->proactor_pool;
	iree_async_proactor_pool_retain(device->proactor_pool);
	iree_status_t status = iree_async_proactor_pool_get_for_node(
		device->proactor_pool, /*node_id=*/0, &device->proactor);

	iree_arena_block_pool_initialize(IREE_HAL_QNN_COMMAND_ARENA_BLOCK_SIZE,
		host_allocator, &device->command_block_pool);
	iree_slim_mutex_initialize(&device->queue_mutex);

	if (iree_status_is_ok(status)) {
		status = iree_hal_allocator_create_heap(IREE_SV("qnn"), host_allocator,
			host_allocator, &device->device_allocator);
	}
	if (!iree_status_is_ok(status)) {
		iree_hal_device_release((iree_hal_device_t *)device);
		IREE_TRACE_ZONE_END(z0);
		return status;
	}

	*out_device = (iree_hal_device_t *)device;
	IREE_TRACE_ZONE_END(z0);
	return iree_ok_status();
}
