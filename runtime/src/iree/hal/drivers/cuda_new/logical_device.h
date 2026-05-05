// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_CUDA_NEW_LOGICAL_DEVICE_H_
#define IREE_HAL_DRIVERS_CUDA_NEW_LOGICAL_DEVICE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "api.h"
#include "dynamic_symbols.h"
#include "nccl_dynamic_symbols.h"
#include "physical_device.h"

iree_status_t iree_hal_cuda_new_logical_device_create(
	iree_hal_driver_t *driver, iree_string_view_t identifier,
	const iree_hal_cuda_new_logical_device_options_t *options,
	const iree_hal_cuda_new_dynamic_symbols_t *syms,
	const iree_hal_cuda_new_nccl_dynamic_symbols_t *nccl_syms,
	const iree_hal_cuda_new_physical_device_t *physical_device,
	const iree_hal_device_create_params_t *create_params,
	iree_allocator_t host_allocator, iree_hal_device_t **out_device);

#endif // IREE_HAL_DRIVERS_CUDA_NEW_LOGICAL_DEVICE_H_
