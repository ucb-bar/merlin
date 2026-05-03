// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_CUDA_NEW_PHYSICAL_DEVICE_H_
#define IREE_HAL_DRIVERS_CUDA_NEW_PHYSICAL_DEVICE_H_

#include "iree/base/api.h"
#include "dynamic_symbols.h"
#include "target_caps.h"

#define IREE_HAL_CUDA_NEW_MAX_DEVICE_NAME_LENGTH 256

typedef struct iree_hal_cuda_new_physical_device_t {
	int cuda_ordinal;
	char device_name[IREE_HAL_CUDA_NEW_MAX_DEVICE_NAME_LENGTH];
	iree_hal_cuda_new_target_caps_t caps;
} iree_hal_cuda_new_physical_device_t;

iree_status_t iree_hal_cuda_new_physical_device_initialize(
	const iree_hal_cuda_new_dynamic_symbols_t *syms, int cuda_ordinal,
	iree_hal_cuda_new_physical_device_t *out_device);

void iree_hal_cuda_new_physical_device_deinitialize(
	iree_hal_cuda_new_physical_device_t *device);

#endif // IREE_HAL_DRIVERS_CUDA_NEW_PHYSICAL_DEVICE_H_
