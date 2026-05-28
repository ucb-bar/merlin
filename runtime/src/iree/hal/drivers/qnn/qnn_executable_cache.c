// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// QNN executable cache. No-op cache: QNN context binaries are produced
// offline by qnn-context-binary-generator and contain everything the runtime
// needs. prepare_executable just forwards the binary blob to
// iree_hal_qnn_executable_create. Mirrors amdgpu/executable_cache.c.

#include "qnn_executable_cache.h"

#include <stdbool.h>
#include <stddef.h>
#include <string.h>

#include "qnn_executable.h"
#include "qnn_graph_builder.h"

typedef struct iree_hal_qnn_executable_cache_t {
	iree_hal_resource_t resource;
	iree_allocator_t host_allocator;

	// Which QNN backend this cache is configured for. Used by
	// can_prepare_format to reject backend-suffixed variants intended for
	// other backends (e.g. an HTA cache rejects "qnn-graph-gpu").
	iree_hal_qnn_backend_t backend;

	// Borrowed from the device.
	iree_hal_qnn_interface_t qnn_interface;
	iree_hal_qnn_backend_handle_t backend_handle;
	iree_hal_qnn_device_handle_t device_handle;
	// Borrowed from the driver via the device. NULL when libQnnSystem.so
	// hasn't been loaded yet — the executable_create path falls back to a
	// per-call dlopen + getProviders so the cache stays usable.
	const void *system_interface;
} iree_hal_qnn_executable_cache_t;

static const iree_hal_executable_cache_vtable_t
	iree_hal_qnn_executable_cache_vtable;

static iree_hal_qnn_executable_cache_t *iree_hal_qnn_executable_cache_cast(
	iree_hal_executable_cache_t *base_value) {
	IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_qnn_executable_cache_vtable);
	return (iree_hal_qnn_executable_cache_t *)base_value;
}

iree_status_t iree_hal_qnn_executable_cache_create(
	iree_string_view_t identifier, iree_hal_qnn_backend_t backend,
	iree_hal_qnn_interface_t qnn_interface,
	iree_hal_qnn_backend_handle_t backend_handle,
	iree_hal_qnn_device_handle_t device_handle, const void *system_interface,
	iree_allocator_t host_allocator,
	iree_hal_executable_cache_t **out_executable_cache) {
	IREE_ASSERT_ARGUMENT(out_executable_cache);
	IREE_TRACE_ZONE_BEGIN(z0);
	*out_executable_cache = NULL;
	(void)identifier;

	iree_hal_qnn_executable_cache_t *cache = NULL;
	IREE_RETURN_AND_END_ZONE_IF_ERROR(z0,
		iree_allocator_malloc(host_allocator, sizeof(*cache), (void **)&cache));
	iree_hal_resource_initialize(
		&iree_hal_qnn_executable_cache_vtable, &cache->resource);
	cache->host_allocator = host_allocator;
	cache->backend = backend;
	cache->qnn_interface = qnn_interface;
	cache->backend_handle = backend_handle;
	cache->device_handle = device_handle;
	cache->system_interface = system_interface;

	*out_executable_cache = (iree_hal_executable_cache_t *)cache;
	IREE_TRACE_ZONE_END(z0);
	return iree_ok_status();
}

static void iree_hal_qnn_executable_cache_destroy(
	iree_hal_executable_cache_t *base_executable_cache) {
	iree_hal_qnn_executable_cache_t *cache =
		iree_hal_qnn_executable_cache_cast(base_executable_cache);
	iree_allocator_t host_allocator = cache->host_allocator;
	IREE_TRACE_ZONE_BEGIN(z0);
	iree_allocator_free(host_allocator, cache);
	IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_qnn_executable_cache_infer_format(
	iree_hal_executable_cache_t *base_executable_cache,
	iree_hal_executable_caching_mode_t caching_mode,
	iree_const_byte_span_t executable_data,
	iree_host_size_t executable_format_capacity, char *executable_format,
	iree_host_size_t *out_inferred_size) {
	(void)base_executable_cache;
	(void)caching_mode;
	// QNN context binaries don't have a stable magic-number prefix we can
	// sniff for, so the caller must already know they're feeding us a
	// qnn-context-binary blob.
	static const char kFormat[] = "qnn-context-binary";
	iree_host_size_t needed = sizeof(kFormat); // includes NUL.
	if (out_inferred_size)
		*out_inferred_size = needed;
	if (executable_format_capacity < needed) {
		return iree_make_status(
			IREE_STATUS_OUT_OF_RANGE, "executable_format buffer too small");
	}
	(void)executable_data;
	memcpy(executable_format, kFormat, needed);
	return iree_ok_status();
}

static bool iree_hal_qnn_executable_cache_can_prepare_format(
	iree_hal_executable_cache_t *base_executable_cache,
	iree_hal_executable_caching_mode_t caching_mode,
	iree_string_view_t executable_format) {
	iree_hal_qnn_executable_cache_t *cache =
		iree_hal_qnn_executable_cache_cast(base_executable_cache);
	(void)caching_mode;
	// Three accepted format families:
	// 1. "qnn-context-binary" — pre-built `.qnn-ctx` blob (HTA's only path
	//    on QAIRT 2.45 because its compiler is closed; produced by
	//    qnn-context-binary-generator on board). Backend-agnostic legacy.
	// 2. "qnn-graph"  — backend-agnostic in-compiler form (legacy / single-
	//    backend VMFBs). Accept on every backend for backward compat.
	// 3. "qnn-graph-{hta|gpu|htp}" — backend-suffixed in-compiler form.
	//    Reject if the suffix doesn't match THIS cache's backend so a
	//    multi-variant VMFB doesn't try to load (e.g.) the GPU variant
	//    onto an HTA-bound cache and fail at QnnGraph_addNode.
	if (iree_string_view_equal(executable_format,
			iree_make_cstring_view(IREE_HAL_QNN_CONTEXT_BINARY_FORMAT)) ||
		iree_string_view_equal(executable_format,
			iree_make_cstring_view(IREE_HAL_QNN_GRAPH_FORMAT))) {
		return true;
	}
	const char *my_suffix = NULL;
	switch (cache->backend) {
		case IREE_HAL_QNN_BACKEND_HTA:
			my_suffix = "qnn-graph-hta";
			break;
		case IREE_HAL_QNN_BACKEND_GPU:
			my_suffix = "qnn-graph-gpu";
			break;
		case IREE_HAL_QNN_BACKEND_HTP:
			my_suffix = "qnn-graph-htp";
			break;
		default:
			return false;
	}
	return iree_string_view_equal(
		executable_format, iree_make_cstring_view(my_suffix));
}

static iree_status_t iree_hal_qnn_executable_cache_prepare_executable(
	iree_hal_executable_cache_t *base_executable_cache,
	const iree_hal_executable_params_t *executable_params,
	iree_hal_executable_t **out_executable) {
	iree_hal_qnn_executable_cache_t *cache =
		iree_hal_qnn_executable_cache_cast(base_executable_cache);
	return iree_hal_qnn_executable_create(executable_params,
		cache->qnn_interface, cache->backend_handle, cache->device_handle,
		cache->system_interface, cache->host_allocator, out_executable);
}

static const iree_hal_executable_cache_vtable_t
	iree_hal_qnn_executable_cache_vtable = {
		.destroy = iree_hal_qnn_executable_cache_destroy,
		.infer_format = iree_hal_qnn_executable_cache_infer_format,
		.can_prepare_format = iree_hal_qnn_executable_cache_can_prepare_format,
		.prepare_executable = iree_hal_qnn_executable_cache_prepare_executable,
};
