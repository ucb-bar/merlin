// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/cuda_tile/registration/driver_module.h"

#include <inttypes.h>
#include <stddef.h>

#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/drivers/cuda_tile/api.h"

IREE_FLAG(
    bool, cuda_tile_use_streams, true,
    "Use CUDA streams (instead of graphs) for executing command buffers.");

IREE_FLAG(
    bool, cuda_tile_async_allocations, true,
    "Enables CUDA asynchronous stream-ordered allocations when supported.");

IREE_FLAG(
    int32_t, cuda_tile_tracing, 2,
    "Controls the verbosity of cuda_tile tracing when Tracy is enabled.\n"
    "  0: disabled, 1: coarse, 2: fine-grained kernel level.");

IREE_FLAG(int32_t, cuda_tile_default_index, 0,
          "Specifies the index of the default CUDA device for cuda_tile");

static iree_status_t iree_hal_cuda_tile_driver_factory_enumerate(
    void* self, iree_host_size_t* out_driver_info_count,
    const iree_hal_driver_info_t** out_driver_infos) {
  IREE_ASSERT_ARGUMENT(out_driver_info_count);
  IREE_ASSERT_ARGUMENT(out_driver_infos);
  IREE_TRACE_ZONE_BEGIN(z0);

  static const iree_hal_driver_info_t driver_infos[1] = {{
      .driver_name = IREE_SVL("cuda_tile"),
      .full_name = IREE_SVL("NVIDIA CUDA Tile IR HAL driver"),
  }};
  *out_driver_info_count = IREE_ARRAYSIZE(driver_infos);
  *out_driver_infos = driver_infos;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_tile_driver_factory_try_create(
    void* self, iree_string_view_t driver_name, iree_allocator_t host_allocator,
    iree_hal_driver_t** out_driver) {
  IREE_ASSERT_ARGUMENT(out_driver);

  if (!iree_string_view_equal(driver_name, IREE_SV("cuda_tile"))) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "no driver '%.*s' is provided by this factory",
                            (int)driver_name.size, driver_name.data);
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_cuda_tile_driver_options_t driver_options;
  iree_hal_cuda_tile_driver_options_initialize(&driver_options);

  iree_hal_cuda_tile_device_params_t device_params;
  iree_hal_cuda_tile_device_params_initialize(&device_params);
  device_params.command_buffer_mode =
      FLAG_cuda_tile_use_streams
          ? IREE_HAL_CUDA_TILE_COMMAND_BUFFER_MODE_STREAM
          : IREE_HAL_CUDA_TILE_COMMAND_BUFFER_MODE_GRAPH;
  device_params.stream_tracing = FLAG_cuda_tile_tracing;
  device_params.async_allocations = FLAG_cuda_tile_async_allocations;

  driver_options.default_device_index = FLAG_cuda_tile_default_index;

  iree_status_t status = iree_hal_cuda_tile_driver_create(
      driver_name, &driver_options, &device_params, host_allocator, out_driver);

  IREE_TRACE_ZONE_END(z0);

  return status;
}

IREE_API_EXPORT iree_status_t
iree_hal_cuda_tile_driver_module_register(iree_hal_driver_registry_t* registry) {
  static const iree_hal_driver_factory_t factory = {
      .self = NULL,
      .enumerate = iree_hal_cuda_tile_driver_factory_enumerate,
      .try_create = iree_hal_cuda_tile_driver_factory_try_create,
  };
  return iree_hal_driver_registry_register_factory(registry, &factory);
}
