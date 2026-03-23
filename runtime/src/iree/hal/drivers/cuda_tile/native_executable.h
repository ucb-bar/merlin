// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Runtime loader for cuda_tile executables (CTL1 FlatBuffer format).
// Loads CUBIN binaries produced by the cuda_tile compiler backend via
// cuModuleLoadDataEx, bypassing PTX JIT compilation entirely.

#ifndef IREE_HAL_DRIVERS_CUDA_TILE_NATIVE_EXECUTABLE_H_
#define IREE_HAL_DRIVERS_CUDA_TILE_NATIVE_EXECUTABLE_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/tracing.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/cuda_tile/cuda_tile_dynamic_symbols.h"
#include "iree/hal/drivers/cuda_tile/cuda_tile_headers.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// Max per-dispatch bindings and constants (matching CUDA HAL limits).
#define IREE_HAL_CUDA_TILE_MAX_DISPATCH_BINDING_COUNT 16
#define IREE_HAL_CUDA_TILE_MAX_DISPATCH_CONSTANT_COUNT 64

typedef struct iree_hal_cuda_tile_kernel_debug_info_t {
  iree_string_view_t function_name;
  iree_string_view_t source_filename;
  uint32_t source_line;
} iree_hal_cuda_tile_kernel_debug_info_t;

typedef struct iree_hal_cuda_tile_kernel_params_t {
  CUfunction function;

  uint32_t constant_count;
  uint32_t binding_count;

  // cuda_tile kernels use block_dims = {1,1,1}.
  // grid_dims are baked at compile time from static tensor shapes.
  uint32_t grid_dims[3];

  // Optional CTA cluster dimensions for Hopper (0,0,0 = no clustering).
  uint32_t cluster_dims[3];

  IREE_TRACE(iree_hal_cuda_tile_kernel_debug_info_t debug_info;)
} iree_hal_cuda_tile_kernel_params_t;

// Infers the format of the executable and calculates its total size.
// Returns "CTL1" as the format string for cuda_tile executables.
iree_status_t iree_hal_cuda_tile_native_executable_infer_format(
    iree_const_byte_span_t executable_data,
    iree_host_size_t executable_format_capacity, char* executable_format,
    iree_host_size_t* out_inferred_size);

// Creates an IREE executable from a cuda_tile CTL1 FlatBuffer containing
// CUBIN binary data. The cubin is loaded via cuModuleLoadDataEx.
iree_status_t iree_hal_cuda_tile_native_executable_create(
    const iree_hal_cuda_tile_dynamic_symbols_t* symbols, CUdevice device,
    const iree_hal_executable_params_t* executable_params,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable);

// Returns the kernel launch parameters for the given |export_ordinal|.
iree_status_t iree_hal_cuda_tile_native_executable_lookup_kernel_params(
    iree_hal_executable_t* executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    const iree_hal_cuda_tile_kernel_params_t** out_params);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // IREE_HAL_DRIVERS_CUDA_TILE_NATIVE_EXECUTABLE_H_
