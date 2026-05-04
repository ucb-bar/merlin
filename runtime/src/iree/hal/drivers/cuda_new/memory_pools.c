// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "memory_pools.h"

#include "buffer.h"
#include "status_util.h"

static iree_status_t iree_hal_cuda_new_create_memory_pool(
	const iree_hal_cuda_new_dynamic_symbols_t *syms, CUdevice cu_device,
	CUmemoryPool *out_pool) {
	*out_pool = NULL;

	CUmemPoolProps pool_props = {
		.allocType = CU_MEM_ALLOCATION_TYPE_PINNED,
		.handleTypes = CU_MEM_HANDLE_TYPE_NONE,
		.location =
			{
				.type = CU_MEM_LOCATION_TYPE_DEVICE,
				.id = cu_device,
			},
		.win32SecurityAttributes = NULL,
		.reserved = {0},
	};

	CUmemoryPool pool = NULL;
	IREE_CUDA_NEW_RETURN_IF_ERROR(syms,
		cuMemPoolCreate(&pool, &pool_props), "cuMemPoolCreate");

	uint64_t threshold = UINT64_MAX;
	iree_status_t status = IREE_CURESULT_TO_STATUS_NEW(syms,
		cuMemPoolSetAttribute(pool, CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
			&threshold),
		"cuMemPoolSetAttribute");

	if (iree_status_is_ok(status)) {
		*out_pool = pool;
	} else {
		IREE_CUDA_NEW_IGNORE_ERROR(syms, cuMemPoolDestroy(pool));
	}
	return status;
}

iree_status_t iree_hal_cuda_new_memory_pools_initialize(
	iree_hal_device_t *parent_device,
	const iree_hal_cuda_new_dynamic_symbols_t *syms,
	CUdevice cu_device, iree_allocator_t host_allocator,
	iree_hal_cuda_new_memory_pools_t *out_pools) {
	IREE_ASSERT_ARGUMENT(parent_device);
	IREE_ASSERT_ARGUMENT(syms);
	IREE_ASSERT_ARGUMENT(out_pools);
	IREE_TRACE_ZONE_BEGIN(z0);

	memset(out_pools, 0, sizeof(*out_pools));
	out_pools->parent_device = parent_device;
	out_pools->syms = syms;
	out_pools->host_allocator = host_allocator;

	iree_status_t status =
		iree_hal_cuda_new_create_memory_pool(syms, cu_device,
			&out_pools->device_local);
	if (iree_status_is_ok(status)) {
		status = iree_hal_cuda_new_create_memory_pool(syms, cu_device,
			&out_pools->other);
	}

	IREE_TRACE_ZONE_END(z0);
	return status;
}

void iree_hal_cuda_new_memory_pools_deinitialize(
	iree_hal_cuda_new_memory_pools_t *pools) {
	IREE_TRACE_ZONE_BEGIN(z0);
	if (pools->device_local) {
		IREE_CUDA_NEW_IGNORE_ERROR(pools->syms,
			cuMemPoolDestroy(pools->device_local));
		pools->device_local = NULL;
	}
	if (pools->other) {
		IREE_CUDA_NEW_IGNORE_ERROR(pools->syms,
			cuMemPoolDestroy(pools->other));
		pools->other = NULL;
	}
	IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_cuda_new_memory_pools_trim(
	iree_hal_cuda_new_memory_pools_t *pools) {
	IREE_CUDA_NEW_RETURN_IF_ERROR(pools->syms,
		cuMemPoolTrimTo(pools->device_local, 0), "cuMemPoolTrimTo");
	IREE_CUDA_NEW_RETURN_IF_ERROR(pools->syms,
		cuMemPoolTrimTo(pools->other, 0), "cuMemPoolTrimTo");
	return iree_ok_status();
}

static void iree_hal_cuda_new_async_buffer_release_callback(
	void *user_data, iree_hal_buffer_t *buffer) {
	iree_hal_cuda_new_memory_pools_t *pools =
		(iree_hal_cuda_new_memory_pools_t *)user_data;
	CUdeviceptr device_ptr = iree_hal_cuda_new_buffer_device_pointer(buffer);
	IREE_CUDA_NEW_IGNORE_ERROR(pools->syms, cuMemFree(device_ptr));
}

iree_status_t iree_hal_cuda_new_memory_pools_alloca(
	iree_hal_cuda_new_memory_pools_t *pools, CUstream stream,
	iree_hal_allocator_pool_t pool, iree_hal_buffer_params_t params,
	iree_device_size_t allocation_size, iree_hal_alloca_flags_t flags,
	iree_hal_buffer_t **out_buffer) {
	IREE_TRACE_ZONE_BEGIN(z0);

	iree_hal_buffer_params_canonicalize(&params);

	CUmemoryPool memory_pool =
		iree_all_bits_set(params.type, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL)
			? pools->device_local
			: pools->other;

	CUdeviceptr device_ptr = 0;
	iree_status_t status = IREE_CURESULT_TO_STATUS_NEW(pools->syms,
		cuMemAllocFromPoolAsync(&device_ptr, (size_t)allocation_size,
			memory_pool, stream),
		"cuMemAllocFromPoolAsync");

	iree_hal_buffer_t *buffer = NULL;
	if (iree_status_is_ok(status)) {
		const iree_hal_buffer_placement_t placement = {
			.device = pools->parent_device,
			.queue_affinity = params.queue_affinity
								 ? params.queue_affinity
								 : IREE_HAL_QUEUE_AFFINITY_ANY,
			.flags = IREE_HAL_BUFFER_PLACEMENT_FLAG_ASYNCHRONOUS,
		};
		iree_hal_buffer_release_callback_t release_callback = {
			.fn = iree_hal_cuda_new_async_buffer_release_callback,
			.user_data = pools,
		};
		status = iree_hal_cuda_new_buffer_wrap(placement, params.type,
			params.access, params.usage, allocation_size,
			/*byte_offset=*/0, /*byte_length=*/allocation_size,
			IREE_HAL_CUDA_NEW_BUFFER_TYPE_ASYNC, device_ptr,
			/*host_ptr=*/NULL, release_callback, pools->host_allocator,
			&buffer);
	}

	if (iree_status_is_ok(status)) {
		*out_buffer = buffer;
	} else if (buffer) {
		iree_hal_buffer_release(buffer);
	} else if (device_ptr) {
		IREE_CUDA_NEW_IGNORE_ERROR(pools->syms,
			cuMemFreeAsync(device_ptr, stream));
	}

	IREE_TRACE_ZONE_END(z0);
	return status;
}

iree_status_t iree_hal_cuda_new_memory_pools_dealloca(
	iree_hal_cuda_new_memory_pools_t *pools, CUstream stream,
	iree_hal_buffer_t *buffer, iree_hal_dealloca_flags_t flags) {
	IREE_TRACE_ZONE_BEGIN(z0);

	iree_status_t status = iree_ok_status();
	if (iree_hal_cuda_new_buffer_type(buffer) ==
		IREE_HAL_CUDA_NEW_BUFFER_TYPE_ASYNC) {
		CUdeviceptr device_ptr =
			iree_hal_cuda_new_buffer_device_pointer(buffer);
		status = IREE_CURESULT_TO_STATUS_NEW(pools->syms,
			cuMemFreeAsync(device_ptr, stream), "cuMemFreeAsync");
		if (iree_status_is_ok(status)) {
			iree_hal_cuda_new_buffer_drop_release_callback(buffer);
		}
	}

	IREE_TRACE_ZONE_END(z0);
	return status;
}
