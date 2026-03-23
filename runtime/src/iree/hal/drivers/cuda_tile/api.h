// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// See iree/base/api.h for documentation on the API conventions used.

#ifndef IREE_HAL_DRIVERS_CUDA_TILE_API_H_
#define IREE_HAL_DRIVERS_CUDA_TILE_API_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_cuda_tile_device_t
//===----------------------------------------------------------------------===//

// How command buffers are recorded and executed.
typedef enum iree_hal_cuda_tile_command_buffer_mode_e {
  // Command buffers are recorded into CUDA graphs.
  IREE_HAL_CUDA_TILE_COMMAND_BUFFER_MODE_GRAPH = 0,
  // Command buffers are directly issued against a CUDA stream.
  IREE_HAL_CUDA_TILE_COMMAND_BUFFER_MODE_STREAM = 1,
} iree_hal_cuda_tile_command_buffer_mode_t;

// ncclUniqueId exposed without exporting the NCCL headers.
typedef struct {
  char data[128];
} iree_hal_cuda_tile_nccl_id_t;

// Parameters defining a CUmemoryPool.
typedef struct iree_hal_cuda_tile_memory_pool_params_t {
  uint64_t minimum_capacity;
  uint64_t release_threshold;
} iree_hal_cuda_tile_memory_pool_params_t;

// Parameters for each CUmemoryPool used for queue-ordered allocations.
typedef struct iree_hal_cuda_tile_memory_pooling_params_t {
  iree_hal_cuda_tile_memory_pool_params_t device_local;
  iree_hal_cuda_tile_memory_pool_params_t other;
} iree_hal_cuda_tile_memory_pooling_params_t;

// Parameters configuring an iree_hal_cuda_tile_device_t.
// Must be initialized with iree_hal_cuda_tile_device_params_initialize prior to
// use.
typedef struct iree_hal_cuda_tile_device_params_t {
  iree_host_size_t queue_count;
  iree_host_size_t arena_block_size;
  iree_host_size_t event_pool_capacity;

  // Specifies how command buffers are recorded and executed.
  iree_hal_cuda_tile_command_buffer_mode_t command_buffer_mode;

  // Controls the verbosity of tracing when IREE tracing is enabled.
  int32_t stream_tracing;

  // Whether to use async allocations even if reported as available by the
  // device. Defaults to true when the device supports it.
  bool async_allocations;

  // Parameters for each CUmemoryPool used for queue-ordered allocations.
  iree_hal_cuda_tile_memory_pooling_params_t memory_pools;
} iree_hal_cuda_tile_device_params_t;

// Initializes |out_params| to default values.
IREE_API_EXPORT void iree_hal_cuda_tile_device_params_initialize(
    iree_hal_cuda_tile_device_params_t* out_params);

//===----------------------------------------------------------------------===//
// iree_hal_cuda_tile_driver_t
//===----------------------------------------------------------------------===//

// CUDA HAL driver creation options.
typedef struct iree_hal_cuda_tile_driver_options_t {
  // The index of the default CUDA device to use within the list of available
  // devices.
  int default_device_index;
} iree_hal_cuda_tile_driver_options_t;

// Initializes the given |out_options| with default driver creation options.
IREE_API_EXPORT void iree_hal_cuda_tile_driver_options_initialize(
    iree_hal_cuda_tile_driver_options_t* out_options);

// Creates a CUDA HAL driver with the given |options|, from which CUDA devices
// can be enumerated and created with specific parameters.
//
// |out_driver| must be released by the caller (see iree_hal_driver_release).
IREE_API_EXPORT iree_status_t iree_hal_cuda_tile_driver_create(
    iree_string_view_t identifier,
    const iree_hal_cuda_tile_driver_options_t* options,
    const iree_hal_cuda_tile_device_params_t* default_params,
    iree_allocator_t host_allocator, iree_hal_driver_t** out_driver);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_CUDA_TILE_API_H_
