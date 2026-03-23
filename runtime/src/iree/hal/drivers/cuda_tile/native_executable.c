// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Runtime loader for cuda_tile executables (CTL1 FlatBuffer format).
// Reads CUBIN binaries from the CTL1 FlatBuffer and loads them via
// cuModuleLoadDataEx — the same CUDA driver API used by the standard
// CUDA backend, but with pre-compiled cubin instead of PTX text.

#include "iree/hal/drivers/cuda_tile/native_executable.h"

#include <stddef.h>

#include "iree/base/api.h"
#include "iree/hal/drivers/cuda_tile/cuda_tile_dynamic_symbols.h"
#include "iree/hal/drivers/cuda_tile/cuda_tile_status_util.h"
#include "iree/hal/utils/executable_debug_info.h"
#include "iree/hal/utils/executable_header.h"

// flatcc schemas:
#include "iree/base/internal/flatcc/parsing.h"
#include "iree/schemas/cuda_tile_executable_def_reader.h"
#include "iree/schemas/cuda_tile_executable_def_verifier.h"
#include "iree/schemas/executable_debug_info_reader.h"
#include "iree/schemas/executable_debug_info_verifier.h"

typedef struct iree_hal_cuda_tile_native_executable_t {
  // Abstract resource used for injecting reference counting and vtable;
  // must be at offset 0.
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;

  const iree_hal_cuda_tile_dynamic_symbols_t* symbols;

  // Loaded CUDA modules (from cubin data in the FlatBuffer).
  iree_host_size_t module_count;
  CUmodule* modules;

  // Exported kernels referencing the loaded modules.
  iree_host_size_t export_count;
  iree_hal_cuda_tile_kernel_params_t exports[];
} iree_hal_cuda_tile_native_executable_t;

static const iree_hal_executable_vtable_t
    iree_hal_cuda_tile_native_executable_vtable;

static iree_hal_cuda_tile_native_executable_t*
iree_hal_cuda_tile_native_executable_cast(
    iree_hal_executable_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value,
                       &iree_hal_cuda_tile_native_executable_vtable);
  return (iree_hal_cuda_tile_native_executable_t*)base_value;
}

//===----------------------------------------------------------------------===//
// FlatBuffer Verification
//===----------------------------------------------------------------------===//

// Verifies the structure of the CTL1 FlatBuffer so that we can avoid doing
// so during runtime. After this succeeds, we can safely walk the file.
static iree_status_t
iree_hal_cuda_tile_native_executable_flatbuffer_verify(
    iree_const_byte_span_t flatbuffer_data) {
  IREE_ASSERT(flatbuffer_data.data && flatbuffer_data.data_length >= 16);

  int verify_ret = iree_hal_cuda_tile_ExecutableDef_verify_as_root(
      flatbuffer_data.data, flatbuffer_data.data_length);
  if (verify_ret != flatcc_verify_ok) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "CTL1 flatbuffer verification failed: %s",
                            flatcc_verify_error_string(verify_ret));
  }

  iree_hal_cuda_tile_ExecutableDef_table_t executable_def =
      iree_hal_cuda_tile_ExecutableDef_as_root(flatbuffer_data.data);

  // Verify modules.
  iree_hal_cuda_tile_ModuleDef_vec_t modules_vec =
      iree_hal_cuda_tile_ExecutableDef_modules_get(executable_def);
  iree_host_size_t module_count =
      iree_hal_cuda_tile_ModuleDef_vec_len(modules_vec);
  for (iree_host_size_t i = 0; i < module_count; ++i) {
    iree_hal_cuda_tile_ModuleDef_table_t module_def =
        iree_hal_cuda_tile_ModuleDef_vec_at(modules_vec, i);
    if (!module_def) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "modules[%" PRIhsz "] is NULL", i);
    }
    flatbuffers_uint8_vec_t cubin_image =
        iree_hal_cuda_tile_ModuleDef_cubin_image_get(module_def);
    if (flatbuffers_uint8_vec_len(cubin_image) == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "modules[%" PRIhsz "] cubin_image is empty", i);
    }
  }

  // Verify exports.
  iree_hal_cuda_tile_ExportDef_vec_t exports_vec =
      iree_hal_cuda_tile_ExecutableDef_exports_get(executable_def);
  for (iree_host_size_t i = 0;
       i < iree_hal_cuda_tile_ExportDef_vec_len(exports_vec); ++i) {
    iree_hal_cuda_tile_ExportDef_table_t export_def =
        iree_hal_cuda_tile_ExportDef_vec_at(exports_vec, i);
    if (!export_def) continue;

    if (flatbuffers_string_len(
            iree_hal_cuda_tile_ExportDef_kernel_name_get(export_def)) == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "exports[%" PRIhsz "] kernel_name is empty", i);
    }

    uint32_t constant_count =
        iree_hal_cuda_tile_ExportDef_constant_count_get(export_def);
    if (constant_count > IREE_HAL_CUDA_TILE_MAX_DISPATCH_CONSTANT_COUNT) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "exports[%" PRIhsz "] constant_count %u exceeds maximum of %u", i,
          constant_count, IREE_HAL_CUDA_TILE_MAX_DISPATCH_CONSTANT_COUNT);
    }

    uint32_t binding_count =
        iree_hal_cuda_tile_ExportDef_binding_count_get(export_def);
    if (binding_count > IREE_HAL_CUDA_TILE_MAX_DISPATCH_BINDING_COUNT) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "exports[%" PRIhsz "] binding_count %u exceeds maximum of %u", i,
          binding_count, IREE_HAL_CUDA_TILE_MAX_DISPATCH_BINDING_COUNT);
    }

    IREE_RETURN_IF_ERROR(iree_hal_debug_verify_export_def(
        iree_hal_cuda_tile_ExportDef_debug_info_get(export_def)));
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Format Inference
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_cuda_tile_native_executable_infer_format(
    iree_const_byte_span_t executable_data,
    iree_host_size_t executable_format_capacity, char* executable_format,
    iree_host_size_t* out_inferred_size) {
  const bool unsafe_infer_size = (executable_data.data_length == 0);
  iree_const_byte_span_t flatbuffer_data = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_hal_read_executable_flatbuffer_header(
      executable_data, unsafe_infer_size,
      iree_hal_cuda_tile_ExecutableDef_file_identifier, &flatbuffer_data));

  if (!iree_hal_cuda_tile_ExecutableDef_verify_as_root(
          flatbuffer_data.data, flatbuffer_data.data_length)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "failed to verify CTL1 flatbuffer structure");
  }

  iree_string_view_t format = IREE_SV("CTL1");
  if (format.size >= executable_format_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "executable format buffer too small");
  }
  memcpy(executable_format, format.data, format.size + /*NUL*/ 1);

  *out_inferred_size =
      sizeof(iree_flatbuffer_file_header_t) + flatbuffer_data.data_length;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Executable Creation
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_cuda_tile_native_executable_create(
    const iree_hal_cuda_tile_dynamic_symbols_t* symbols, CUdevice device,
    const iree_hal_executable_params_t* executable_params,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(executable_params);
  IREE_ASSERT_ARGUMENT(out_executable);
  IREE_TRACE_ZONE_BEGIN(z0);

  *out_executable = NULL;

  // Read and strip the FlatBuffer header prefix.
  iree_const_byte_span_t executable_flatbuffer = iree_const_byte_span_empty();
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hal_read_executable_flatbuffer_header(
          executable_params->executable_data, /*unsafe_infer_size=*/false,
          iree_hal_cuda_tile_ExecutableDef_file_identifier,
          &executable_flatbuffer));

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_cuda_tile_native_executable_flatbuffer_verify(
              executable_flatbuffer));

  iree_hal_cuda_tile_ExecutableDef_table_t executable_def =
      iree_hal_cuda_tile_ExecutableDef_as_root(executable_flatbuffer.data);

  iree_hal_cuda_tile_ModuleDef_vec_t modules_vec =
      iree_hal_cuda_tile_ExecutableDef_modules_get(executable_def);
  iree_host_size_t module_count =
      iree_hal_cuda_tile_ModuleDef_vec_len(modules_vec);
  iree_hal_cuda_tile_ExportDef_vec_t exports_vec =
      iree_hal_cuda_tile_ExecutableDef_exports_get(executable_def);
  iree_host_size_t export_count =
      iree_hal_cuda_tile_ExportDef_vec_len(exports_vec);

  // Calculate total export info length for tracing.
  iree_host_size_t total_export_info_length = 0;
  IREE_TRACE({
    for (iree_host_size_t i = 0; i < export_count; ++i) {
      iree_hal_cuda_tile_ExportDef_table_t export_def =
          iree_hal_cuda_tile_ExportDef_vec_at(exports_vec, i);
      total_export_info_length += iree_hal_debug_calculate_export_info_size(
          iree_hal_cuda_tile_ExportDef_debug_info_get(export_def));
    }
  });

  // Allocate storage for the executable and its associated data structures.
  iree_hal_cuda_tile_native_executable_t* executable = NULL;
  const iree_host_size_t total_size =
      sizeof(*executable) + module_count * sizeof(executable->modules[0]) +
      export_count * sizeof(executable->exports[0]) + total_export_info_length;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, total_size, (void**)&executable));

  iree_hal_resource_initialize(&iree_hal_cuda_tile_native_executable_vtable,
                               &executable->resource);
  executable->host_allocator = host_allocator;
  executable->symbols = symbols;
  executable->module_count = module_count;
  executable->modules =
      (CUmodule*)((uint8_t*)executable + sizeof(*executable) +
                  export_count * sizeof(executable->exports[0]));
  executable->export_count = export_count;
  IREE_TRACE(uint8_t* export_info_ptr =
                 ((uint8_t*)executable->modules +
                  module_count * sizeof(executable->modules[0])));

  // Publish any embedded source files to tracing infrastructure.
  iree_hal_debug_publish_source_files(
      iree_hal_cuda_tile_ExecutableDef_source_files_get(executable_def));

  // Load CUBIN modules via cuModuleLoadDataEx.
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < module_count; ++i) {
    iree_hal_cuda_tile_ModuleDef_table_t module_def =
        iree_hal_cuda_tile_ModuleDef_vec_at(modules_vec, i);

    flatbuffers_uint8_vec_t cubin_image =
        iree_hal_cuda_tile_ModuleDef_cubin_image_get(module_def);

    char error_log[8192] = {0};
    CUjit_option jit_options[] = {
        CU_JIT_ERROR_LOG_BUFFER,
        CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
    };
    void* jit_option_values[] = {
        (void*)error_log,
        (void*)(uint32_t)sizeof(error_log),
    };
    CUmodule module = NULL;
    status = IREE_CURESULT_TO_STATUS(
        symbols,
        cuModuleLoadDataEx(&module, cubin_image, IREE_ARRAYSIZE(jit_options),
                           jit_options, jit_option_values),
        "cuModuleLoadDataEx");
    if (!iree_status_is_ok(status)) {
      status = iree_status_annotate(
          status,
          IREE_SV("cuda_tile: mismatched sm_arch? cubin load failed"));
      if (strlen(error_log) > 0) {
        status =
            iree_status_annotate(status, iree_make_cstring_view(error_log));
      }
      break;
    }

    executable->modules[i] = module;
  }

  // Look up exported kernel functions from the loaded modules.
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < export_count; ++i) {
      iree_hal_cuda_tile_ExportDef_table_t export_def =
          iree_hal_cuda_tile_ExportDef_vec_at(exports_vec, i);

      // All cuda_tile kernels are in module 0 today.
      CUmodule module = executable->modules[0];
      flatbuffers_string_t kernel_name =
          iree_hal_cuda_tile_ExportDef_kernel_name_get(export_def);
      CUfunction function = NULL;
      status = IREE_CURESULT_TO_STATUS(
          symbols, cuModuleGetFunction(&function, module, kernel_name),
          "cuModuleGetFunction");
      if (!iree_status_is_ok(status)) break;
      if (!function) {
        status = iree_make_status(IREE_STATUS_NOT_FOUND,
                                  "exports[%" PRIhsz
                                  "] kernel `%s` not found in module",
                                  i, kernel_name);
        break;
      }

      // Package kernel parameters.
      iree_hal_cuda_tile_kernel_params_t* kernel_info =
          &executable->exports[i];
      kernel_info->function = function;
      kernel_info->binding_count =
          iree_hal_cuda_tile_ExportDef_binding_count_get(export_def);
      kernel_info->constant_count =
          iree_hal_cuda_tile_ExportDef_constant_count_get(export_def);

      // Grid dims from the FlatBuffer export definition.
      const iree_hal_cuda_tile_GridDims_t* grid_dims =
          iree_hal_cuda_tile_ExportDef_grid_dims_get(export_def);
      if (grid_dims) {
        kernel_info->grid_dims[0] = grid_dims->x;
        kernel_info->grid_dims[1] = grid_dims->y;
        kernel_info->grid_dims[2] = grid_dims->z;
      } else {
        kernel_info->grid_dims[0] = 1;
        kernel_info->grid_dims[1] = 1;
        kernel_info->grid_dims[2] = 1;
      }

      // Cluster dims for Hopper CTA clustering (future schema extension).
      // Default to {0,0,0} = no clustering.
      kernel_info->cluster_dims[0] = 0;
      kernel_info->cluster_dims[1] = 0;
      kernel_info->cluster_dims[2] = 0;

      // Copy debug info for tracing.
      IREE_TRACE({
        iree_hal_debug_export_info_t* export_info =
            (iree_hal_debug_export_info_t*)export_info_ptr;
        export_info_ptr += iree_hal_debug_copy_export_info(
            iree_hal_cuda_tile_ExportDef_debug_info_get(export_def),
            export_info);
        kernel_info->debug_info.function_name = export_info->function_name;
        kernel_info->debug_info.source_filename = export_info->source_filename;
        kernel_info->debug_info.source_line = export_info->source_line;
      });
    }
  }

  if (iree_status_is_ok(status)) {
    *out_executable = (iree_hal_executable_t*)executable;
  } else {
    // Cleanup on failure.
    for (iree_host_size_t i = 0; i < module_count; ++i) {
      if (executable->modules[i]) {
        symbols->cuModuleUnload(executable->modules[i]);
      }
    }
    iree_allocator_free(host_allocator, executable);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Kernel Parameter Lookup
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_cuda_tile_native_executable_lookup_kernel_params(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    const iree_hal_cuda_tile_kernel_params_t** out_params) {
  iree_hal_cuda_tile_native_executable_t* executable =
      iree_hal_cuda_tile_native_executable_cast(base_executable);
  if (export_ordinal >= executable->export_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "export ordinal %u out of range; count=%" PRIhsz,
                            export_ordinal, executable->export_count);
  }
  *out_params = &executable->exports[export_ordinal];
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Executable Lifecycle
//===----------------------------------------------------------------------===//

static void iree_hal_cuda_tile_native_executable_destroy(
    iree_hal_executable_t* base_executable) {
  iree_hal_cuda_tile_native_executable_t* executable =
      iree_hal_cuda_tile_native_executable_cast(base_executable);
  iree_allocator_t host_allocator = executable->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  for (iree_host_size_t i = 0; i < executable->module_count; ++i) {
    if (executable->modules[i]) {
      executable->symbols->cuModuleUnload(executable->modules[i]);
    }
  }

  iree_allocator_free(host_allocator, executable);
  IREE_TRACE_ZONE_END(z0);
}

static const iree_hal_executable_vtable_t
    iree_hal_cuda_tile_native_executable_vtable = {
        .destroy = iree_hal_cuda_tile_native_executable_destroy,
};
