// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Simplified stream command buffer for cuda_tile HAL driver.
// Compared to the upstream CUDA HAL:
//   - No NCCL collective operations
//   - Block dims always {1,1,1}
//   - Grid dims from IREE dispatch workgroup_count

#include "iree/hal/drivers/cuda_tile/cuda_tile_stream_command_buffer.h"

#include "iree/hal/drivers/cuda_tile/cuda_tile_buffer.h"
#include "iree/hal/drivers/cuda_tile/cuda_tile_status_util.h"
#include "iree/hal/drivers/cuda_tile/native_executable.h"
#include "iree/hal/utils/resource_set.h"

typedef struct iree_hal_cuda_tile_stream_command_buffer_t {
  iree_hal_command_buffer_t base;
  iree_allocator_t host_allocator;

  const iree_hal_cuda_tile_dynamic_symbols_t* cuda_symbols;

  // Per-stream CUDA tracing context.
  iree_hal_stream_tracing_context_t* tracing_context;
  iree_hal_stream_tracing_context_event_list_t tracing_event_list;

  CUstream cu_stream;

  // A resource set to maintain references to all resources used within the
  // command buffer. Reset on each begin.
  iree_hal_resource_set_t* resource_set;

  // Staging arena used for host->device transfers.
  iree_arena_allocator_t arena;
} iree_hal_cuda_tile_stream_command_buffer_t;

static const iree_hal_command_buffer_vtable_t
    iree_hal_cuda_tile_stream_command_buffer_vtable;

static iree_hal_cuda_tile_stream_command_buffer_t*
iree_hal_cuda_tile_stream_command_buffer_cast(
    iree_hal_command_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value,
                        &iree_hal_cuda_tile_stream_command_buffer_vtable);
  return (iree_hal_cuda_tile_stream_command_buffer_t*)base_value;
}

iree_status_t iree_hal_cuda_tile_stream_command_buffer_create(
    iree_hal_allocator_t* device_allocator,
    const iree_hal_cuda_tile_dynamic_symbols_t* cuda_symbols,
    iree_hal_stream_tracing_context_t* tracing_context,
    iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_host_size_t binding_capacity, CUstream stream,
    iree_arena_block_pool_t* block_pool, iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_command_buffer) {
  IREE_ASSERT_ARGUMENT(device_allocator);
  IREE_ASSERT_ARGUMENT(cuda_symbols);
  IREE_ASSERT_ARGUMENT(out_command_buffer);
  *out_command_buffer = NULL;

  if (binding_capacity > 0) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "indirect command buffers not yet implemented");
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_cuda_tile_stream_command_buffer_t* command_buffer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator,
                            sizeof(*command_buffer) +
                                iree_hal_command_buffer_validation_state_size(
                                    mode, binding_capacity),
                            (void**)&command_buffer));

  iree_hal_command_buffer_initialize(
      device_allocator, mode, command_categories, IREE_HAL_QUEUE_AFFINITY_ANY,
      binding_capacity, (uint8_t*)command_buffer + sizeof(*command_buffer),
      &iree_hal_cuda_tile_stream_command_buffer_vtable, &command_buffer->base);
  command_buffer->host_allocator = host_allocator;
  command_buffer->cuda_symbols = cuda_symbols;
  command_buffer->tracing_context = tracing_context;
  command_buffer->tracing_event_list.head = NULL;
  command_buffer->tracing_event_list.tail = NULL;
  command_buffer->cu_stream = stream;
  iree_arena_initialize(block_pool, &command_buffer->arena);

  iree_status_t status = iree_ok_status();
  if (!iree_all_bits_set(mode, IREE_HAL_COMMAND_BUFFER_MODE_UNRETAINED)) {
    status = iree_hal_resource_set_allocate(block_pool,
                                            &command_buffer->resource_set);
  }

  *out_command_buffer = &command_buffer->base;
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_cuda_tile_stream_command_buffer_destroy(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_cuda_tile_stream_command_buffer_t* command_buffer =
      iree_hal_cuda_tile_stream_command_buffer_cast(base_command_buffer);
  iree_allocator_t host_allocator = command_buffer->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_stream_tracing_free(command_buffer->tracing_context,
                               &command_buffer->tracing_event_list);

  iree_hal_resource_set_free(command_buffer->resource_set);
  iree_arena_deinitialize(&command_buffer->arena);
  iree_allocator_free(host_allocator, command_buffer);

  IREE_TRACE_ZONE_END(z0);
}

bool iree_hal_cuda_tile_stream_command_buffer_isa(
    iree_hal_command_buffer_t* command_buffer) {
  return iree_hal_resource_is(
      &command_buffer->resource,
      &iree_hal_cuda_tile_stream_command_buffer_vtable);
}

void iree_hal_cuda_tile_stream_notify_submitted_commands(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_cuda_tile_stream_command_buffer_t* command_buffer =
      iree_hal_cuda_tile_stream_command_buffer_cast(base_command_buffer);
  if (!command_buffer->tracing_context) {
    return;
  }
  iree_hal_stream_tracing_notify_submitted(command_buffer->tracing_context,
                                           &command_buffer->tracing_event_list);
}

static iree_status_t iree_hal_cuda_tile_stream_command_buffer_begin(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_cuda_tile_stream_command_buffer_t* command_buffer =
      iree_hal_cuda_tile_stream_command_buffer_cast(base_command_buffer);
  (void)command_buffer;

  IREE_HAL_STREAM_TRACE_ZONE_BEGIN_EXTERNAL(
      command_buffer->tracing_context, &command_buffer->tracing_event_list,
      IREE_HAL_STREAM_TRACING_VERBOSITY_COARSE,
      /*file_name=*/NULL, 0, /*line=*/0,
      "iree_hal_cuda_tile_stream_command_buffer",
      strlen("iree_hal_cuda_tile_stream_command_buffer"), /*name=*/NULL, 0);

  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_tile_stream_command_buffer_end(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_cuda_tile_stream_command_buffer_t* command_buffer =
      iree_hal_cuda_tile_stream_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Reset the arena as there should be nothing using it now.
  iree_arena_reset(&command_buffer->arena);
  iree_hal_resource_set_free(command_buffer->resource_set);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_resource_set_allocate(command_buffer->arena.block_pool,
                                         &command_buffer->resource_set));

  IREE_HAL_STREAM_TRACE_ZONE_END(command_buffer->tracing_context,
                                 &command_buffer->tracing_event_list,
                                 IREE_HAL_STREAM_TRACING_VERBOSITY_COARSE);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t
iree_hal_cuda_tile_stream_command_buffer_begin_debug_group(
    iree_hal_command_buffer_t* base_command_buffer, iree_string_view_t label,
    iree_hal_label_color_t label_color,
    const iree_hal_label_location_t* location) {
  iree_hal_cuda_tile_stream_command_buffer_t* command_buffer =
      iree_hal_cuda_tile_stream_command_buffer_cast(base_command_buffer);
  (void)command_buffer;

  IREE_HAL_STREAM_TRACE_ZONE_BEGIN_EXTERNAL(
      command_buffer->tracing_context, &command_buffer->tracing_event_list,
      IREE_HAL_STREAM_TRACING_VERBOSITY_COARSE,
      location ? location->file.data : NULL,
      location ? location->file.size : 0, location ? location->line : 0,
      /*func_name=*/NULL, 0, label.data, label.size);

  return iree_ok_status();
}

static iree_status_t
iree_hal_cuda_tile_stream_command_buffer_end_debug_group(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_cuda_tile_stream_command_buffer_t* command_buffer =
      iree_hal_cuda_tile_stream_command_buffer_cast(base_command_buffer);
  (void)command_buffer;

  IREE_HAL_STREAM_TRACE_ZONE_END(command_buffer->tracing_context,
                                 &command_buffer->tracing_event_list,
                                 IREE_HAL_STREAM_TRACING_VERBOSITY_COARSE);

  return iree_ok_status();
}

static iree_status_t
iree_hal_cuda_tile_stream_command_buffer_execution_barrier(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_execution_barrier_flags_t flags,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  if (iree_any_bit_set(source_stage_mask, IREE_HAL_EXECUTION_STAGE_HOST) ||
      iree_any_bit_set(target_stage_mask, IREE_HAL_EXECUTION_STAGE_HOST)) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "barrier involving host not yet supported");
  }

  if (flags != IREE_HAL_EXECUTION_BARRIER_FLAG_NONE) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "non-zero barrier flag not yet supported");
  }

  // Nothing to do — CUDA stream semantics guarantees execution and memory
  // visibility in program order.
  return iree_ok_status();
}

static iree_status_t
iree_hal_cuda_tile_stream_command_buffer_signal_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "event not yet supported");
}

static iree_status_t iree_hal_cuda_tile_stream_command_buffer_reset_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "event not yet supported");
}

static iree_status_t iree_hal_cuda_tile_stream_command_buffer_wait_events(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_host_size_t event_count, const iree_hal_event_t** events,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "event not yet supported");
}

static iree_status_t
iree_hal_cuda_tile_stream_command_buffer_advise_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t buffer_ref, iree_hal_memory_advise_flags_t flags,
    uint64_t arg0, uint64_t arg1) {
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_tile_stream_command_buffer_fill_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t target_ref, const void* pattern,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags) {
  iree_hal_cuda_tile_stream_command_buffer_t* command_buffer =
      iree_hal_cuda_tile_stream_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  CUdeviceptr target_device_buffer = iree_hal_cuda_tile_buffer_device_pointer(
      iree_hal_buffer_allocated_buffer(target_ref.buffer));
  iree_device_size_t target_offset =
      iree_hal_buffer_byte_offset(target_ref.buffer) + target_ref.offset;
  CUdeviceptr dst = target_device_buffer + target_offset;
  size_t num_elements = target_ref.length / pattern_length;
  switch (pattern_length) {
    case 4: {
      IREE_CUDA_TILE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, command_buffer->cuda_symbols,
          cuMemsetD32Async(dst, *(const uint32_t*)(pattern), num_elements,
                           command_buffer->cu_stream),
          "cuMemsetD32Async");
      break;
    }
    case 2: {
      IREE_CUDA_TILE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, command_buffer->cuda_symbols,
          cuMemsetD16Async(dst, *(const uint16_t*)(pattern), num_elements,
                           command_buffer->cu_stream),
          "cuMemsetD16Async");
      break;
    }
    case 1: {
      IREE_CUDA_TILE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, command_buffer->cuda_symbols,
          cuMemsetD8Async(dst, *(const uint8_t*)(pattern), num_elements,
                          command_buffer->cu_stream),
          "cuMemsetD8Async");
      break;
    }
    default:
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "unsupported fill pattern length");
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t
iree_hal_cuda_tile_stream_command_buffer_update_buffer(
    iree_hal_command_buffer_t* base_command_buffer, const void* source_buffer,
    iree_host_size_t source_offset, iree_hal_buffer_ref_t target_ref,
    iree_hal_update_flags_t flags) {
  iree_hal_cuda_tile_stream_command_buffer_t* command_buffer =
      iree_hal_cuda_tile_stream_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Copy source to arena to ensure async safety.
  const uint8_t* src = (const uint8_t*)source_buffer + source_offset;
  if (command_buffer->arena.block_pool) {
    uint8_t* storage = NULL;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_arena_allocate(&command_buffer->arena, target_ref.length,
                                (void**)&storage));
    memcpy(storage, src, target_ref.length);
    src = storage;
  }

  CUdeviceptr target_device_buffer = iree_hal_cuda_tile_buffer_device_pointer(
      iree_hal_buffer_allocated_buffer(target_ref.buffer));
  CUdeviceptr dst = target_device_buffer +
                    iree_hal_buffer_byte_offset(target_ref.buffer) +
                    target_ref.offset;
  IREE_CUDA_TILE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, command_buffer->cuda_symbols,
      cuMemcpyHtoDAsync(dst, src, target_ref.length,
                        command_buffer->cu_stream),
      "cuMemcpyHtoDAsync");

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_tile_stream_command_buffer_copy_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t source_ref, iree_hal_buffer_ref_t target_ref,
    iree_hal_copy_flags_t flags) {
  iree_hal_cuda_tile_stream_command_buffer_t* command_buffer =
      iree_hal_cuda_tile_stream_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  CUdeviceptr source_device_buffer = iree_hal_cuda_tile_buffer_device_pointer(
      iree_hal_buffer_allocated_buffer(source_ref.buffer));
  iree_device_size_t source_offset =
      iree_hal_buffer_byte_offset(source_ref.buffer) + source_ref.offset;
  CUdeviceptr target_device_buffer = iree_hal_cuda_tile_buffer_device_pointer(
      iree_hal_buffer_allocated_buffer(target_ref.buffer));
  iree_device_size_t target_offset =
      iree_hal_buffer_byte_offset(target_ref.buffer) + target_ref.offset;
  CUdeviceptr src_ptr = source_device_buffer + source_offset;
  CUdeviceptr dst_ptr = target_device_buffer + target_offset;

  IREE_CUDA_TILE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, command_buffer->cuda_symbols,
      cuMemcpyAsync(dst_ptr, src_ptr, target_ref.length,
                    command_buffer->cu_stream),
      "cuMemcpyAsync");

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_tile_stream_command_buffer_collective(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_channel_t* channel,
    iree_hal_collective_op_t op, uint32_t param, iree_hal_buffer_ref_t send_ref,
    iree_hal_buffer_ref_t recv_ref, iree_device_size_t element_count) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "NCCL collectives not available in cuda_tile driver");
}

//===----------------------------------------------------------------------===//
// cuda_tile dispatch — block_dims={1,1,1}, grid from workgroup_count
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_cuda_tile_stream_command_buffer_dispatch(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    iree_hal_buffer_ref_list_t bindings, iree_hal_dispatch_flags_t flags) {
  iree_hal_cuda_tile_stream_command_buffer_t* command_buffer =
      iree_hal_cuda_tile_stream_command_buffer_cast(base_command_buffer);

  if (iree_hal_dispatch_uses_custom_arguments(flags)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "direct/indirect arguments are not supported in CUDA streams");
  } else if (iree_hal_dispatch_uses_indirect_parameters(flags)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "indirect parameters are not supported in CUDA streams");
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  // Lookup kernel parameters from the CTL1 executable.
  const iree_hal_cuda_tile_kernel_params_t* kernel_params = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_cuda_tile_native_executable_lookup_kernel_params(
              executable, export_ordinal, &kernel_params));

  IREE_HAL_STREAM_TRACE_ZONE_BEGIN_EXTERNAL(
      command_buffer->tracing_context, &command_buffer->tracing_event_list,
      IREE_HAL_STREAM_TRACING_VERBOSITY_FINE,
      kernel_params->debug_info.source_filename.data,
      kernel_params->debug_info.source_filename.size,
      kernel_params->debug_info.source_line,
      kernel_params->debug_info.function_name.data,
      kernel_params->debug_info.function_name.size,
      /*name=*/NULL, 0);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_resource_set_insert(command_buffer->resource_set, 1,
                                       &executable));

  // Build the two-level parameter indirection required by CUDA.
  iree_host_size_t kernel_params_count =
      kernel_params->binding_count + kernel_params->constant_count;
  iree_host_size_t kernel_params_length = kernel_params_count * sizeof(void*);
  iree_host_size_t total_size = kernel_params_length * 2;
  uint8_t* storage_base = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_arena_allocate(&command_buffer->arena, total_size,
                              (void**)&storage_base));
  void** params_ptr = (void**)storage_base;
  CUdeviceptr* payload_ptr =
      (CUdeviceptr*)((uint8_t*)params_ptr + kernel_params_length);
  for (size_t i = 0; i < kernel_params_count; i++) {
    params_ptr[i] = &payload_ptr[i];
  }
  for (iree_host_size_t i = 0; i < bindings.count; i++) {
    const iree_hal_buffer_ref_t* binding = &bindings.values[i];
    CUdeviceptr device_ptr = 0;
    if (binding->buffer) {
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_hal_resource_set_insert(command_buffer->resource_set, 1,
                                           &binding->buffer));
      CUdeviceptr device_buffer = iree_hal_cuda_tile_buffer_device_pointer(
          iree_hal_buffer_allocated_buffer(binding->buffer));
      iree_device_size_t offset =
          iree_hal_buffer_byte_offset(binding->buffer);
      device_ptr = device_buffer + offset + binding->offset;
    }
    payload_ptr[i] = device_ptr;
  }

  // Push constants stored after bindings.
  for (iree_host_size_t i = 0; i < kernel_params->constant_count; i++) {
    *((uint32_t*)params_ptr[kernel_params->binding_count + i]) =
        ((const uint32_t*)constants.data)[i];
  }

  // cuda_tile dispatch: block_dims = {1,1,1}, grid from workgroup_count.
  // Shared memory is managed by the cubin itself.
  IREE_CUDA_TILE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, command_buffer->cuda_symbols,
      cuLaunchKernel(kernel_params->function,
                     config.workgroup_count[0],  // grid X
                     config.workgroup_count[1],  // grid Y
                     config.workgroup_count[2],  // grid Z
                     1, 1, 1,                    // block dims always {1,1,1}
                     0,  // shared memory managed by cubin
                     command_buffer->cu_stream, params_ptr, NULL),
      "cuLaunchKernel");

  IREE_HAL_STREAM_TRACE_ZONE_END(command_buffer->tracing_context,
                                 &command_buffer->tracing_event_list,
                                 IREE_HAL_STREAM_TRACING_VERBOSITY_FINE);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static const iree_hal_command_buffer_vtable_t
    iree_hal_cuda_tile_stream_command_buffer_vtable = {
        .destroy = iree_hal_cuda_tile_stream_command_buffer_destroy,
        .begin = iree_hal_cuda_tile_stream_command_buffer_begin,
        .end = iree_hal_cuda_tile_stream_command_buffer_end,
        .begin_debug_group =
            iree_hal_cuda_tile_stream_command_buffer_begin_debug_group,
        .end_debug_group =
            iree_hal_cuda_tile_stream_command_buffer_end_debug_group,
        .execution_barrier =
            iree_hal_cuda_tile_stream_command_buffer_execution_barrier,
        .signal_event =
            iree_hal_cuda_tile_stream_command_buffer_signal_event,
        .reset_event =
            iree_hal_cuda_tile_stream_command_buffer_reset_event,
        .wait_events =
            iree_hal_cuda_tile_stream_command_buffer_wait_events,
        .advise_buffer =
            iree_hal_cuda_tile_stream_command_buffer_advise_buffer,
        .fill_buffer =
            iree_hal_cuda_tile_stream_command_buffer_fill_buffer,
        .update_buffer =
            iree_hal_cuda_tile_stream_command_buffer_update_buffer,
        .copy_buffer =
            iree_hal_cuda_tile_stream_command_buffer_copy_buffer,
        .collective =
            iree_hal_cuda_tile_stream_command_buffer_collective,
        .dispatch = iree_hal_cuda_tile_stream_command_buffer_dispatch,
};
