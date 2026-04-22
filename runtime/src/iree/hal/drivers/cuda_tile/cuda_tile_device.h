// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_CUDA_TILE_CUDA_TILE_DEVICE_H_
#define IREE_HAL_DRIVERS_CUDA_TILE_CUDA_TILE_DEVICE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/cuda_tile/api.h"
#include "iree/hal/drivers/cuda_tile/cuda_tile_dynamic_symbols.h"
#include "iree/hal/drivers/cuda_tile/cuda_tile_nccl_dynamic_symbols.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a device that owns and manages its own CUcontext.
iree_status_t iree_hal_cuda_tile_device_create(
    iree_hal_driver_t* driver, iree_string_view_t identifier,
    const iree_hal_cuda_tile_device_params_t* params,
    const iree_hal_cuda_tile_dynamic_symbols_t* symbols,
    const iree_hal_cuda_tile_nccl_dynamic_symbols_t* nccl_symbols,
    CUdevice device, const iree_hal_device_create_params_t* create_params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device);

// Creates a CUDA stream-backed command buffer using resources from the given
// |base_device|.
iree_status_t iree_hal_cuda_tile_device_create_stream_command_buffer(
    iree_hal_device_t* base_device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_host_size_t binding_capacity,
    iree_hal_command_buffer_t** out_command_buffer);

// Returns the CUDA context bound to the given |device|.
//
// WARNING: this API is unsafe and unstable.
CUcontext iree_hal_cuda_tile_device_context(iree_hal_device_t* device);

// Returns the dynamic symbol table from the |device|.
//
// WARNING: the symbols are only valid for as long as the device is.
const iree_hal_cuda_tile_dynamic_symbols_t*
iree_hal_cuda_tile_device_dynamic_symbols(iree_hal_device_t* device);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_CUDA_TILE_CUDA_TILE_DEVICE_H_
