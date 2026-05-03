// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "physical_device.h"

#include <string.h>

#include "status_util.h"

static iree_status_t iree_hal_cuda_new_query_device_attribute(
	const iree_hal_cuda_new_dynamic_symbols_t *syms, CUdevice device,
	CUdevice_attribute attribute, int *out_value) {
	IREE_CUDA_NEW_RETURN_IF_ERROR(syms,
		cuDeviceGetAttribute(out_value, attribute, device),
		"cuDeviceGetAttribute");
	return iree_ok_status();
}

static iree_status_t iree_hal_cuda_new_fill_target_caps(
	const iree_hal_cuda_new_dynamic_symbols_t *syms, CUdevice device,
	int cuda_ordinal, iree_hal_cuda_new_target_caps_t *caps) {
	iree_hal_cuda_new_target_caps_initialize_defaults(caps);
	caps->cuda_ordinal = cuda_ordinal;

	int value = 0;

	IREE_RETURN_IF_ERROR(iree_hal_cuda_new_query_device_attribute(
		syms, device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, &value));
	caps->compute_capability_major = value;

	IREE_RETURN_IF_ERROR(iree_hal_cuda_new_query_device_attribute(
		syms, device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, &value));
	caps->compute_capability_minor = value;

	IREE_RETURN_IF_ERROR(iree_hal_cuda_new_query_device_attribute(
		syms, device, CU_DEVICE_ATTRIBUTE_WARP_SIZE, &value));
	caps->warp_size = (uint32_t)value;

	IREE_RETURN_IF_ERROR(iree_hal_cuda_new_query_device_attribute(
		syms, device, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, &value));
	caps->max_threads_per_block = (uint32_t)value;

	IREE_RETURN_IF_ERROR(iree_hal_cuda_new_query_device_attribute(
		syms, device, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, &value));
	caps->max_block_dims[0] = (uint32_t)value;
	IREE_RETURN_IF_ERROR(iree_hal_cuda_new_query_device_attribute(
		syms, device, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, &value));
	caps->max_block_dims[1] = (uint32_t)value;
	IREE_RETURN_IF_ERROR(iree_hal_cuda_new_query_device_attribute(
		syms, device, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, &value));
	caps->max_block_dims[2] = (uint32_t)value;

	IREE_RETURN_IF_ERROR(iree_hal_cuda_new_query_device_attribute(
		syms, device, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, &value));
	caps->max_grid_dims[0] = (uint32_t)value;
	IREE_RETURN_IF_ERROR(iree_hal_cuda_new_query_device_attribute(
		syms, device, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, &value));
	caps->max_grid_dims[1] = (uint32_t)value;
	IREE_RETURN_IF_ERROR(iree_hal_cuda_new_query_device_attribute(
		syms, device, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, &value));
	caps->max_grid_dims[2] = (uint32_t)value;

	IREE_RETURN_IF_ERROR(iree_hal_cuda_new_query_device_attribute(syms,
		device, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, &value));
	caps->max_shared_memory_per_block = (uint32_t)value;

	caps->pointer_size_bits = 64;

	// SM 9.0+ cluster/TMA features — query but tolerate absence.
	if (caps->compute_capability_major >= 9) {
		if (iree_status_is_ok(iree_hal_cuda_new_query_device_attribute(
				syms, device,
				CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH, &value))) {
			caps->supports_clusters = (value != 0);
		}
		caps->supports_tma = true;
	}

	// Async copy (cp.async) available on SM 8.0+.
	caps->supports_async_copy = (caps->compute_capability_major >= 8);

	return iree_ok_status();
}

iree_status_t iree_hal_cuda_new_physical_device_initialize(
	const iree_hal_cuda_new_dynamic_symbols_t *syms, int cuda_ordinal,
	iree_hal_cuda_new_physical_device_t *out_device) {
	IREE_ASSERT_ARGUMENT(syms);
	IREE_ASSERT_ARGUMENT(out_device);
	IREE_TRACE_ZONE_BEGIN(z0);

	memset(out_device, 0, sizeof(*out_device));
	out_device->cuda_ordinal = cuda_ordinal;

	IREE_CUDA_NEW_RETURN_AND_END_ZONE_IF_ERROR(z0, syms,
		cuDeviceGet(&out_device->cu_device, cuda_ordinal),
		"cuDeviceGet(%d)", cuda_ordinal);

	IREE_CUDA_NEW_RETURN_AND_END_ZONE_IF_ERROR(z0, syms,
		cuDeviceGetName(out_device->device_name,
			IREE_HAL_CUDA_NEW_MAX_DEVICE_NAME_LENGTH,
			out_device->cu_device),
		"cuDeviceGetName");

	iree_status_t status = iree_hal_cuda_new_fill_target_caps(
		syms, out_device->cu_device, cuda_ordinal, &out_device->caps);

	IREE_TRACE_ZONE_END(z0);
	return status;
}

void iree_hal_cuda_new_physical_device_deinitialize(
	iree_hal_cuda_new_physical_device_t *device) {
	if (!device) return;
	memset(device, 0, sizeof(*device));
}
