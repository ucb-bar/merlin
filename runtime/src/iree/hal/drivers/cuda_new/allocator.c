// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "allocator.h"

#include <stddef.h>

#include "iree/base/api.h"
#include "buffer.h"
#include "status_util.h"

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_TRACKING
static const char *IREE_HAL_CUDA_NEW_ALLOCATOR_ID = "CUDA_NEW unpooled";
#endif // IREE_TRACING_FEATURE_ALLOCATION_TRACKING

typedef struct iree_hal_cuda_new_allocator_t {
	iree_hal_resource_t resource;
	iree_hal_device_t *parent_device;
	CUdevice device;
	CUstream stream;
	const iree_hal_cuda_new_dynamic_symbols_t *symbols;
	iree_allocator_t host_allocator;
	bool supports_concurrent_managed_access;
	IREE_STATISTICS(iree_hal_allocator_statistics_t statistics;)
} iree_hal_cuda_new_allocator_t;

static const iree_hal_allocator_vtable_t iree_hal_cuda_new_allocator_vtable;

static iree_hal_cuda_new_allocator_t *iree_hal_cuda_new_allocator_cast(
	iree_hal_allocator_t *base_value) {
	IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_cuda_new_allocator_vtable);
	return (iree_hal_cuda_new_allocator_t *)base_value;
}

iree_status_t iree_hal_cuda_new_allocator_create(
	iree_hal_device_t *parent_device,
	const iree_hal_cuda_new_dynamic_symbols_t *syms, CUdevice device,
	CUstream stream, iree_allocator_t host_allocator,
	iree_hal_allocator_t **out_allocator) {
	IREE_ASSERT_ARGUMENT(parent_device);
	IREE_ASSERT_ARGUMENT(syms);
	IREE_ASSERT_ARGUMENT(out_allocator);
	IREE_TRACE_ZONE_BEGIN(z0);

	int supports_concurrent_managed_access = 0;
	IREE_RETURN_AND_END_ZONE_IF_ERROR(z0,
		IREE_CURESULT_TO_STATUS_NEW(syms,
			cuDeviceGetAttribute(&supports_concurrent_managed_access,
				CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS, device),
			"cuDeviceGetAttribute"));

	iree_hal_cuda_new_allocator_t *allocator = NULL;
	IREE_RETURN_AND_END_ZONE_IF_ERROR(z0,
		iree_allocator_malloc(host_allocator, sizeof(*allocator),
			(void **)&allocator));

	iree_hal_resource_initialize(&iree_hal_cuda_new_allocator_vtable,
		&allocator->resource);
	allocator->parent_device = parent_device;
	allocator->device = device;
	allocator->stream = stream;
	allocator->symbols = syms;
	allocator->host_allocator = host_allocator;
	allocator->supports_concurrent_managed_access =
		supports_concurrent_managed_access != 0;
	*out_allocator = (iree_hal_allocator_t *)allocator;

	IREE_TRACE_ZONE_END(z0);
	return iree_ok_status();
}

static void iree_hal_cuda_new_allocator_destroy(
	iree_hal_allocator_t *IREE_RESTRICT base_allocator) {
	iree_hal_cuda_new_allocator_t *allocator =
		iree_hal_cuda_new_allocator_cast(base_allocator);
	IREE_TRACE_ZONE_BEGIN(z0);
	iree_allocator_free(allocator->host_allocator, allocator);
	IREE_TRACE_ZONE_END(z0);
}

static iree_allocator_t iree_hal_cuda_new_allocator_host_allocator(
	const iree_hal_allocator_t *IREE_RESTRICT base_allocator) {
	iree_hal_cuda_new_allocator_t *allocator =
		(iree_hal_cuda_new_allocator_t *)base_allocator;
	return allocator->host_allocator;
}

static iree_status_t iree_hal_cuda_new_allocator_trim(
	iree_hal_allocator_t *IREE_RESTRICT base_allocator) {
	return iree_ok_status();
}

static void iree_hal_cuda_new_allocator_query_statistics(
	iree_hal_allocator_t *IREE_RESTRICT base_allocator,
	iree_hal_allocator_statistics_t *IREE_RESTRICT out_statistics) {
	IREE_STATISTICS({
		iree_hal_cuda_new_allocator_t *allocator =
			iree_hal_cuda_new_allocator_cast(base_allocator);
		memcpy(out_statistics, &allocator->statistics, sizeof(*out_statistics));
	});
}

static iree_status_t iree_hal_cuda_new_allocator_query_memory_heaps(
	iree_hal_allocator_t *IREE_RESTRICT base_allocator,
	iree_host_size_t capacity,
	iree_hal_allocator_memory_heap_t *IREE_RESTRICT heaps,
	iree_host_size_t *IREE_RESTRICT out_count) {
	iree_hal_cuda_new_allocator_t *allocator =
		iree_hal_cuda_new_allocator_cast(base_allocator);

	iree_host_size_t count = 3;
	if (allocator->supports_concurrent_managed_access) {
		++count;
	}
	if (out_count) *out_count = count;
	if (capacity < count) {
		return iree_status_from_code(IREE_STATUS_OUT_OF_RANGE);
	}

	const iree_device_size_t max_allocation_size = ~(iree_device_size_t)0;
	const iree_device_size_t min_alignment = 64;

	int i = 0;

	heaps[i++] = (iree_hal_allocator_memory_heap_t){
		.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
		.allowed_usage =
			IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_DISPATCH,
		.max_allocation_size = max_allocation_size,
		.min_alignment = min_alignment,
	};

	if (allocator->supports_concurrent_managed_access) {
		heaps[i++] = (iree_hal_allocator_memory_heap_t){
			.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL |
					IREE_HAL_MEMORY_TYPE_HOST_VISIBLE |
					IREE_HAL_MEMORY_TYPE_HOST_COHERENT,
			.allowed_usage = IREE_HAL_BUFFER_USAGE_TRANSFER |
							 IREE_HAL_BUFFER_USAGE_DISPATCH |
							 IREE_HAL_BUFFER_USAGE_MAPPING,
			.max_allocation_size = max_allocation_size,
			.min_alignment = min_alignment,
		};
	}

	heaps[i++] = (iree_hal_allocator_memory_heap_t){
		.type = IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE |
				IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
				IREE_HAL_MEMORY_TYPE_HOST_COHERENT,
		.allowed_usage = IREE_HAL_BUFFER_USAGE_TRANSFER |
						 IREE_HAL_BUFFER_USAGE_DISPATCH |
						 IREE_HAL_BUFFER_USAGE_MAPPING,
		.max_allocation_size = max_allocation_size,
		.min_alignment = min_alignment,
	};

	heaps[i++] = (iree_hal_allocator_memory_heap_t){
		.type = IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE |
				IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
				IREE_HAL_MEMORY_TYPE_HOST_COHERENT |
				IREE_HAL_MEMORY_TYPE_HOST_CACHED,
		.allowed_usage = IREE_HAL_BUFFER_USAGE_TRANSFER |
						 IREE_HAL_BUFFER_USAGE_DISPATCH |
						 IREE_HAL_BUFFER_USAGE_MAPPING,
		.max_allocation_size = max_allocation_size,
		.min_alignment = min_alignment,
	};

	IREE_ASSERT(i == count);
	return iree_ok_status();
}

static iree_hal_buffer_compatibility_t
iree_hal_cuda_new_allocator_query_buffer_compatibility(
	iree_hal_allocator_t *IREE_RESTRICT base_allocator,
	iree_hal_buffer_params_t *IREE_RESTRICT params,
	iree_device_size_t *IREE_RESTRICT allocation_size) {
	iree_hal_cuda_new_allocator_t *allocator =
		iree_hal_cuda_new_allocator_cast(base_allocator);

	iree_hal_buffer_compatibility_t compatibility =
		IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE;

	if (iree_all_bits_set(params->type, IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE)) {
		compatibility |= IREE_HAL_BUFFER_COMPATIBILITY_IMPORTABLE;
	}

	if (iree_all_bits_set(params->type, IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE)) {
		if (iree_any_bit_set(params->usage, IREE_HAL_BUFFER_USAGE_TRANSFER)) {
			compatibility |= IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_TRANSFER;
		}
		if (iree_any_bit_set(params->usage,
				IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE)) {
			compatibility |= IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_DISPATCH;
		}
	}

	if (!allocator->supports_concurrent_managed_access &&
		iree_all_bits_set(params->type, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL |
											IREE_HAL_MEMORY_TYPE_HOST_VISIBLE)) {
		compatibility |= IREE_HAL_BUFFER_COMPATIBILITY_LOW_PERFORMANCE;
		params->type &= ~(IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL |
						  IREE_HAL_MEMORY_TYPE_HOST_VISIBLE);
		params->type |=
			IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
	}

	params->type &= ~IREE_HAL_MEMORY_TYPE_OPTIMAL;

	if (*allocation_size == 0) *allocation_size = 4;

	return compatibility;
}

static void iree_hal_cuda_new_buffer_free(
	const iree_hal_cuda_new_dynamic_symbols_t *syms,
	iree_hal_cuda_new_buffer_type_t buffer_type, CUdeviceptr device_ptr,
	void *host_ptr) {
	IREE_TRACE_ZONE_BEGIN(z0);
	switch (buffer_type) {
	case IREE_HAL_CUDA_NEW_BUFFER_TYPE_DEVICE:
		IREE_CUDA_NEW_IGNORE_ERROR(syms, cuMemFree(device_ptr));
		break;
	case IREE_HAL_CUDA_NEW_BUFFER_TYPE_HOST:
		IREE_CUDA_NEW_IGNORE_ERROR(syms, cuMemFreeHost(host_ptr));
		break;
	case IREE_HAL_CUDA_NEW_BUFFER_TYPE_HOST_REGISTERED:
		IREE_CUDA_NEW_IGNORE_ERROR(syms, cuMemHostUnregister(host_ptr));
		break;
	case IREE_HAL_CUDA_NEW_BUFFER_TYPE_EXTERNAL:
		break;
	}
	IREE_TRACE_ZONE_END(z0);
}

static void iree_hal_cuda_new_buffer_release_callback(void *user_data,
	iree_hal_buffer_t *buffer);

static iree_status_t iree_hal_cuda_new_allocator_allocate_buffer(
	iree_hal_allocator_t *IREE_RESTRICT base_allocator,
	const iree_hal_buffer_params_t *IREE_RESTRICT params,
	iree_device_size_t allocation_size,
	iree_hal_buffer_t **IREE_RESTRICT out_buffer) {
	iree_hal_cuda_new_allocator_t *allocator =
		iree_hal_cuda_new_allocator_cast(base_allocator);

	iree_hal_buffer_params_t compat_params = *params;
	iree_hal_buffer_compatibility_t compatibility =
		iree_hal_cuda_new_allocator_query_buffer_compatibility(
			base_allocator, &compat_params, &allocation_size);
	if (!iree_all_bits_set(compatibility,
			IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE)) {
		return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
			"allocator cannot allocate a buffer with the given parameters");
	}

	iree_status_t status = iree_ok_status();
	iree_hal_cuda_new_buffer_type_t buffer_type =
		IREE_HAL_CUDA_NEW_BUFFER_TYPE_DEVICE;
	void *host_ptr = NULL;
	CUdeviceptr device_ptr = 0;

	if (iree_all_bits_set(compat_params.type,
			IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL)) {
		if (iree_all_bits_set(compat_params.type,
				IREE_HAL_MEMORY_TYPE_HOST_VISIBLE)) {
			buffer_type = IREE_HAL_CUDA_NEW_BUFFER_TYPE_DEVICE;
			status = IREE_CURESULT_TO_STATUS_NEW(allocator->symbols,
				cuMemAllocManaged(&device_ptr, allocation_size,
					CU_MEM_ATTACH_GLOBAL));
			if (iree_status_is_ok(status) &&
				allocator->supports_concurrent_managed_access) {
				status = IREE_CURESULT_TO_STATUS_NEW(allocator->symbols,
					cuMemPrefetchAsync(device_ptr, allocation_size,
						allocator->device, allocator->stream));
			}
			host_ptr = (void *)device_ptr;
		} else {
			buffer_type = IREE_HAL_CUDA_NEW_BUFFER_TYPE_DEVICE;
			status = IREE_CURESULT_TO_STATUS_NEW(allocator->symbols,
				cuMemAlloc(&device_ptr, allocation_size));
		}
	} else {
		buffer_type = IREE_HAL_CUDA_NEW_BUFFER_TYPE_HOST;
		unsigned int flags = CU_MEMHOSTALLOC_DEVICEMAP;
		if (!iree_all_bits_set(compat_params.type,
				IREE_HAL_MEMORY_TYPE_HOST_CACHED)) {
			flags |= CU_MEMHOSTALLOC_WRITECOMBINED;
		}
		status = IREE_CURESULT_TO_STATUS_NEW(allocator->symbols,
			cuMemHostAlloc(&host_ptr, allocation_size, flags));
		if (iree_status_is_ok(status)) {
			status = IREE_CURESULT_TO_STATUS_NEW(allocator->symbols,
				cuMemHostGetDevicePointer(&device_ptr, host_ptr, 0));
		}
	}

	iree_hal_buffer_t *buffer = NULL;
	if (iree_status_is_ok(status)) {
		const iree_hal_buffer_placement_t placement = {
			.device = allocator->parent_device,
			.queue_affinity = params->queue_affinity
								 ? params->queue_affinity
								 : IREE_HAL_QUEUE_AFFINITY_ANY,
			.flags = IREE_HAL_BUFFER_PLACEMENT_FLAG_NONE,
		};
		iree_hal_buffer_release_callback_t callback = {
			.fn = iree_hal_cuda_new_buffer_release_callback,
			.user_data = (void *)base_allocator,
		};
		status = iree_hal_cuda_new_buffer_wrap(placement, compat_params.type,
			compat_params.access, compat_params.usage, allocation_size,
			/*byte_offset=*/0, /*byte_length=*/allocation_size, buffer_type,
			device_ptr, host_ptr, callback,
			iree_hal_allocator_host_allocator(base_allocator), &buffer);
	}

	if (iree_status_is_ok(status)) {
		IREE_TRACE_ALLOC_NAMED(IREE_HAL_CUDA_NEW_ALLOCATOR_ID,
			(void *)iree_hal_cuda_new_buffer_device_pointer(buffer),
			allocation_size);
		IREE_STATISTICS(iree_hal_allocator_statistics_record_alloc(
			&allocator->statistics, compat_params.type, allocation_size));
		*out_buffer = buffer;
	} else {
		if (!buffer && (device_ptr || host_ptr)) {
			iree_hal_cuda_new_buffer_free(allocator->symbols, buffer_type,
				device_ptr, host_ptr);
		} else {
			iree_hal_buffer_release(buffer);
		}
	}
	return status;
}

static void iree_hal_cuda_new_allocator_deallocate_buffer(
	iree_hal_allocator_t *IREE_RESTRICT base_allocator,
	iree_hal_buffer_t *IREE_RESTRICT base_buffer) {
	iree_hal_cuda_new_allocator_t *allocator =
		iree_hal_cuda_new_allocator_cast(base_allocator);

	const iree_hal_cuda_new_buffer_type_t buffer_type =
		iree_hal_cuda_new_buffer_type(base_buffer);

	iree_hal_cuda_new_buffer_free(allocator->symbols, buffer_type,
		iree_hal_cuda_new_buffer_device_pointer(base_buffer),
		iree_hal_cuda_new_buffer_host_pointer(base_buffer));

	switch (buffer_type) {
	case IREE_HAL_CUDA_NEW_BUFFER_TYPE_DEVICE:
	case IREE_HAL_CUDA_NEW_BUFFER_TYPE_HOST:
		IREE_TRACE_FREE_NAMED(IREE_HAL_CUDA_NEW_ALLOCATOR_ID,
			(void *)iree_hal_cuda_new_buffer_device_pointer(base_buffer));
		IREE_STATISTICS(iree_hal_allocator_statistics_record_free(
			&allocator->statistics, iree_hal_buffer_memory_type(base_buffer),
			iree_hal_buffer_allocation_size(base_buffer)));
		break;
	default:
		break;
	}

	iree_hal_buffer_destroy(base_buffer);
}

static void iree_hal_cuda_new_buffer_release_callback(void *user_data,
	iree_hal_buffer_t *buffer) {
	iree_hal_cuda_new_allocator_t *allocator =
		(iree_hal_cuda_new_allocator_t *)user_data;

	const iree_hal_cuda_new_buffer_type_t buffer_type =
		iree_hal_cuda_new_buffer_type(buffer);

	iree_hal_cuda_new_buffer_free(allocator->symbols, buffer_type,
		iree_hal_cuda_new_buffer_device_pointer(buffer),
		iree_hal_cuda_new_buffer_host_pointer(buffer));

	switch (buffer_type) {
	case IREE_HAL_CUDA_NEW_BUFFER_TYPE_DEVICE:
	case IREE_HAL_CUDA_NEW_BUFFER_TYPE_HOST:
		IREE_TRACE_FREE_NAMED(IREE_HAL_CUDA_NEW_ALLOCATOR_ID,
			(void *)iree_hal_cuda_new_buffer_device_pointer(buffer));
		IREE_STATISTICS(iree_hal_allocator_statistics_record_free(
			&allocator->statistics, iree_hal_buffer_memory_type(buffer),
			iree_hal_buffer_allocation_size(buffer)));
		break;
	default:
		break;
	}
}

static iree_status_t iree_hal_cuda_new_allocator_import_buffer(
	iree_hal_allocator_t *IREE_RESTRICT base_allocator,
	const iree_hal_buffer_params_t *IREE_RESTRICT params,
	iree_hal_external_buffer_t *IREE_RESTRICT external_buffer,
	iree_hal_buffer_release_callback_t release_callback,
	iree_hal_buffer_t **IREE_RESTRICT out_buffer) {
	return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
		"cuda_new buffer import not yet implemented");
}

static iree_status_t iree_hal_cuda_new_allocator_export_buffer(
	iree_hal_allocator_t *IREE_RESTRICT base_allocator,
	iree_hal_buffer_t *IREE_RESTRICT buffer,
	iree_hal_external_buffer_type_t requested_type,
	iree_hal_external_buffer_flags_t requested_flags,
	iree_hal_external_buffer_t *IREE_RESTRICT out_external_buffer) {
	return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
		"cuda_new buffer export not yet implemented");
}

static const iree_hal_allocator_vtable_t iree_hal_cuda_new_allocator_vtable = {
	.destroy = iree_hal_cuda_new_allocator_destroy,
	.host_allocator = iree_hal_cuda_new_allocator_host_allocator,
	.trim = iree_hal_cuda_new_allocator_trim,
	.query_statistics = iree_hal_cuda_new_allocator_query_statistics,
	.query_memory_heaps = iree_hal_cuda_new_allocator_query_memory_heaps,
	.query_buffer_compatibility =
		iree_hal_cuda_new_allocator_query_buffer_compatibility,
	.allocate_buffer = iree_hal_cuda_new_allocator_allocate_buffer,
	.deallocate_buffer = iree_hal_cuda_new_allocator_deallocate_buffer,
	.import_buffer = iree_hal_cuda_new_allocator_import_buffer,
	.export_buffer = iree_hal_cuda_new_allocator_export_buffer,
};
