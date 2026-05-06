// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "command_buffer.h"

#include <string.h>

#include "buffer.h"
#include "executable.h"
#include "nccl_channel.h"
#include "status_util.h"
#include "iree/hal/utils/collective_batch.h"
#include "iree/hal/utils/resource_set.h"

typedef struct iree_hal_cuda_new_stream_command_buffer_t {
	iree_hal_command_buffer_t base;
	iree_allocator_t host_allocator;

	const iree_hal_cuda_new_dynamic_symbols_t *syms;
	const iree_hal_cuda_new_nccl_dynamic_symbols_t *nccl_syms;
	CUcontext cu_context;
	CUstream cu_stream;

	iree_hal_resource_set_t *resource_set;
	iree_arena_allocator_t arena;
	iree_hal_collective_batch_t collective_batch;
} iree_hal_cuda_new_stream_command_buffer_t;

static const iree_hal_command_buffer_vtable_t
	iree_hal_cuda_new_stream_command_buffer_vtable;

static iree_status_t
iree_hal_cuda_new_stream_command_buffer_flush_collectives(
	iree_hal_cuda_new_stream_command_buffer_t *command_buffer);

static iree_hal_cuda_new_stream_command_buffer_t *
iree_hal_cuda_new_stream_command_buffer_cast(
	iree_hal_command_buffer_t *base_value) {
	IREE_HAL_ASSERT_TYPE(base_value,
		&iree_hal_cuda_new_stream_command_buffer_vtable);
	return (iree_hal_cuda_new_stream_command_buffer_t *)base_value;
}

iree_status_t iree_hal_cuda_new_stream_command_buffer_create(
	iree_hal_allocator_t *device_allocator,
	const iree_hal_cuda_new_dynamic_symbols_t *syms,
	const iree_hal_cuda_new_nccl_dynamic_symbols_t *nccl_syms,
	CUcontext cu_context,
	iree_hal_command_buffer_mode_t mode,
	iree_hal_command_category_t command_categories,
	iree_host_size_t binding_capacity, CUstream stream,
	iree_arena_block_pool_t *block_pool, iree_allocator_t host_allocator,
	iree_hal_command_buffer_t **out_command_buffer) {
	IREE_ASSERT_ARGUMENT(device_allocator);
	IREE_ASSERT_ARGUMENT(syms);
	IREE_ASSERT_ARGUMENT(out_command_buffer);
	*out_command_buffer = NULL;

	if (binding_capacity > 0) {
		return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
			"indirect command buffers not yet implemented");
	}

	IREE_TRACE_ZONE_BEGIN(z0);

	iree_hal_cuda_new_stream_command_buffer_t *command_buffer = NULL;
	IREE_RETURN_AND_END_ZONE_IF_ERROR(z0,
		iree_allocator_malloc(host_allocator,
			sizeof(*command_buffer) +
				iree_hal_command_buffer_validation_state_size(
					mode, binding_capacity),
			(void **)&command_buffer));

	iree_hal_command_buffer_initialize(device_allocator, mode,
		command_categories, IREE_HAL_QUEUE_AFFINITY_ANY, binding_capacity,
		(uint8_t *)command_buffer + sizeof(*command_buffer),
		&iree_hal_cuda_new_stream_command_buffer_vtable,
		&command_buffer->base);
	command_buffer->host_allocator = host_allocator;
	command_buffer->syms = syms;
	command_buffer->nccl_syms = nccl_syms;
	command_buffer->cu_context = cu_context;
	command_buffer->cu_stream = stream;
	iree_arena_initialize(block_pool, &command_buffer->arena);

	iree_status_t status = iree_ok_status();
	if (!iree_all_bits_set(mode, IREE_HAL_COMMAND_BUFFER_MODE_UNRETAINED)) {
		status = iree_hal_resource_set_allocate(block_pool,
			&command_buffer->resource_set);
	}

	if (iree_status_is_ok(status)) {
		iree_hal_collective_batch_initialize(&command_buffer->arena,
			command_buffer->resource_set,
			&command_buffer->collective_batch);
	}

	*out_command_buffer = &command_buffer->base;
	IREE_TRACE_ZONE_END(z0);
	return status;
}

bool iree_hal_cuda_new_stream_command_buffer_isa(
	iree_hal_command_buffer_t *command_buffer) {
	return iree_hal_resource_is(&command_buffer->resource,
		&iree_hal_cuda_new_stream_command_buffer_vtable);
}

static void iree_hal_cuda_new_stream_command_buffer_destroy(
	iree_hal_command_buffer_t *base_command_buffer) {
	iree_hal_cuda_new_stream_command_buffer_t *command_buffer =
		iree_hal_cuda_new_stream_command_buffer_cast(base_command_buffer);
	iree_allocator_t host_allocator = command_buffer->host_allocator;
	IREE_TRACE_ZONE_BEGIN(z0);

	iree_hal_collective_batch_deinitialize(
		&command_buffer->collective_batch);
	iree_hal_resource_set_free(command_buffer->resource_set);
	iree_arena_deinitialize(&command_buffer->arena);
	iree_allocator_free(host_allocator, command_buffer);

	IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_cuda_new_stream_command_buffer_begin(
	iree_hal_command_buffer_t *base_command_buffer) {
	iree_hal_cuda_new_stream_command_buffer_t *command_buffer =
		iree_hal_cuda_new_stream_command_buffer_cast(base_command_buffer);
	return IREE_CURESULT_TO_STATUS_NEW(command_buffer->syms,
		cuCtxSetCurrent(command_buffer->cu_context), "cuCtxSetCurrent");
}

static iree_status_t iree_hal_cuda_new_stream_command_buffer_end(
	iree_hal_command_buffer_t *base_command_buffer) {
	iree_hal_cuda_new_stream_command_buffer_t *command_buffer =
		iree_hal_cuda_new_stream_command_buffer_cast(base_command_buffer);
	return iree_hal_cuda_new_stream_command_buffer_flush_collectives(
		command_buffer);
}

static iree_status_t
iree_hal_cuda_new_stream_command_buffer_begin_debug_group(
	iree_hal_command_buffer_t *base_command_buffer, iree_string_view_t label,
	iree_hal_label_color_t label_color,
	const iree_hal_label_location_t *location) {
	return iree_ok_status();
}

static iree_status_t
iree_hal_cuda_new_stream_command_buffer_end_debug_group(
	iree_hal_command_buffer_t *base_command_buffer) {
	return iree_ok_status();
}

static iree_status_t
iree_hal_cuda_new_stream_command_buffer_execution_barrier(
	iree_hal_command_buffer_t *base_command_buffer,
	iree_hal_execution_stage_t source_stage_mask,
	iree_hal_execution_stage_t target_stage_mask,
	iree_hal_execution_barrier_flags_t flags,
	iree_host_size_t memory_barrier_count,
	const iree_hal_memory_barrier_t *memory_barriers,
	iree_host_size_t buffer_barrier_count,
	const iree_hal_buffer_barrier_t *buffer_barriers) {
	// On a single stream, barriers are implicit.
	return iree_ok_status();
}

static iree_status_t
iree_hal_cuda_new_stream_command_buffer_signal_event(
	iree_hal_command_buffer_t *base_command_buffer, iree_hal_event_t *event,
	iree_hal_execution_stage_t source_stage_mask) {
	return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
		"cuda_new stream command buffer events are not yet implemented");
}

static iree_status_t
iree_hal_cuda_new_stream_command_buffer_reset_event(
	iree_hal_command_buffer_t *base_command_buffer, iree_hal_event_t *event,
	iree_hal_execution_stage_t source_stage_mask) {
	return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
		"cuda_new stream command buffer events are not yet implemented");
}

static iree_status_t
iree_hal_cuda_new_stream_command_buffer_wait_events(
	iree_hal_command_buffer_t *base_command_buffer,
	iree_host_size_t event_count, const iree_hal_event_t **events,
	iree_hal_execution_stage_t source_stage_mask,
	iree_hal_execution_stage_t target_stage_mask,
	iree_host_size_t memory_barrier_count,
	const iree_hal_memory_barrier_t *memory_barriers,
	iree_host_size_t buffer_barrier_count,
	const iree_hal_buffer_barrier_t *buffer_barriers) {
	return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
		"cuda_new stream command buffer events are not yet implemented");
}

static iree_status_t
iree_hal_cuda_new_stream_command_buffer_advise_buffer(
	iree_hal_command_buffer_t *base_command_buffer,
	iree_hal_buffer_ref_t buffer_ref, iree_hal_memory_advise_flags_t flags,
	uint64_t arg0, uint64_t arg1) {
	return iree_ok_status();
}

static iree_status_t iree_hal_cuda_new_stream_command_buffer_fill_buffer(
	iree_hal_command_buffer_t *base_command_buffer,
	iree_hal_buffer_ref_t target_ref, const void *pattern,
	iree_host_size_t pattern_length, iree_hal_fill_flags_t flags) {
	iree_hal_cuda_new_stream_command_buffer_t *command_buffer =
		iree_hal_cuda_new_stream_command_buffer_cast(base_command_buffer);
	IREE_TRACE_ZONE_BEGIN(z0);

	CUdeviceptr target_device_buffer =
		iree_hal_cuda_new_buffer_device_pointer(
			iree_hal_buffer_allocated_buffer(target_ref.buffer));
	iree_device_size_t target_offset =
		iree_hal_buffer_byte_offset(target_ref.buffer) + target_ref.offset;
	CUdeviceptr dst = target_device_buffer + target_offset;
	size_t num_elements = target_ref.length / pattern_length;

	switch (pattern_length) {
	case 4: {
		IREE_CUDA_NEW_RETURN_AND_END_ZONE_IF_ERROR(z0,
			command_buffer->syms,
			cuMemsetD32Async(dst, *(const uint32_t *)(pattern),
				num_elements, command_buffer->cu_stream),
			"cuMemsetD32Async");
		break;
	}
	case 2: {
		IREE_CUDA_NEW_RETURN_AND_END_ZONE_IF_ERROR(z0,
			command_buffer->syms,
			cuMemsetD16Async(dst, *(const uint16_t *)(pattern),
				num_elements, command_buffer->cu_stream),
			"cuMemsetD16Async");
		break;
	}
	case 1: {
		IREE_CUDA_NEW_RETURN_AND_END_ZONE_IF_ERROR(z0,
			command_buffer->syms,
			cuMemsetD8Async(dst, *(const uint8_t *)(pattern),
				num_elements, command_buffer->cu_stream),
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
iree_hal_cuda_new_stream_command_buffer_update_buffer(
	iree_hal_command_buffer_t *base_command_buffer,
	const void *source_buffer, iree_host_size_t source_offset,
	iree_hal_buffer_ref_t target_ref, iree_hal_update_flags_t flags) {
	iree_hal_cuda_new_stream_command_buffer_t *command_buffer =
		iree_hal_cuda_new_stream_command_buffer_cast(base_command_buffer);
	IREE_TRACE_ZONE_BEGIN(z0);

	const uint8_t *src = (const uint8_t *)source_buffer + source_offset;
	if (command_buffer->arena.block_pool) {
		uint8_t *storage = NULL;
		IREE_RETURN_AND_END_ZONE_IF_ERROR(z0,
			iree_arena_allocate(&command_buffer->arena,
				target_ref.length, (void **)&storage));
		memcpy(storage, src, target_ref.length);
		src = storage;
	}

	CUdeviceptr target_device_buffer =
		iree_hal_cuda_new_buffer_device_pointer(
			iree_hal_buffer_allocated_buffer(target_ref.buffer));
	CUdeviceptr dst = target_device_buffer +
					  iree_hal_buffer_byte_offset(target_ref.buffer) +
					  target_ref.offset;
	IREE_CUDA_NEW_RETURN_AND_END_ZONE_IF_ERROR(z0,
		command_buffer->syms,
		cuMemcpyHtoDAsync(dst, src, target_ref.length,
			command_buffer->cu_stream),
		"cuMemcpyHtoDAsync");

	IREE_TRACE_ZONE_END(z0);
	return iree_ok_status();
}

static iree_status_t iree_hal_cuda_new_stream_command_buffer_copy_buffer(
	iree_hal_command_buffer_t *base_command_buffer,
	iree_hal_buffer_ref_t source_ref, iree_hal_buffer_ref_t target_ref,
	iree_hal_copy_flags_t flags) {
	iree_hal_cuda_new_stream_command_buffer_t *command_buffer =
		iree_hal_cuda_new_stream_command_buffer_cast(base_command_buffer);
	IREE_TRACE_ZONE_BEGIN(z0);

	CUdeviceptr source_device_buffer =
		iree_hal_cuda_new_buffer_device_pointer(
			iree_hal_buffer_allocated_buffer(source_ref.buffer));
	iree_device_size_t source_offset =
		iree_hal_buffer_byte_offset(source_ref.buffer) + source_ref.offset;
	CUdeviceptr target_device_buffer =
		iree_hal_cuda_new_buffer_device_pointer(
			iree_hal_buffer_allocated_buffer(target_ref.buffer));
	iree_device_size_t target_offset =
		iree_hal_buffer_byte_offset(target_ref.buffer) + target_ref.offset;

	IREE_CUDA_NEW_RETURN_AND_END_ZONE_IF_ERROR(z0,
		command_buffer->syms,
		cuMemcpyAsync(target_device_buffer + target_offset,
			source_device_buffer + source_offset, target_ref.length,
			command_buffer->cu_stream),
		"cuMemcpyAsync");

	IREE_TRACE_ZONE_END(z0);
	return iree_ok_status();
}

static iree_status_t
iree_hal_cuda_new_stream_command_buffer_collective(
	iree_hal_command_buffer_t *base_command_buffer,
	iree_hal_channel_t *channel, iree_hal_collective_op_t op,
	uint32_t param, iree_hal_buffer_ref_t send_ref,
	iree_hal_buffer_ref_t recv_ref, iree_device_size_t element_count) {
	iree_hal_cuda_new_stream_command_buffer_t *command_buffer =
		iree_hal_cuda_new_stream_command_buffer_cast(base_command_buffer);
	// Graph capture cannot capture NCCL collectives; nccl_syms is NULL
	// during graph mode to make this explicit. Collectives are unsupported
	// in graph command buffers — use stream fallback for collective work.
	if (!command_buffer->nccl_syms || !command_buffer->nccl_syms->dylib) {
		return iree_make_status(IREE_STATUS_UNAVAILABLE,
			"NCCL runtime library not available for collective ops");
	}
	IREE_RETURN_IF_ERROR(iree_hal_resource_set_insert(
		command_buffer->resource_set, 1, &channel));
	iree_hal_buffer_binding_t send_binding = {
		.buffer = send_ref.buffer,
		.offset = send_ref.offset,
		.length = send_ref.length,
	};
	iree_hal_buffer_binding_t recv_binding = {
		.buffer = recv_ref.buffer,
		.offset = recv_ref.offset,
		.length = recv_ref.length,
	};
	return iree_hal_collective_batch_append(
		&command_buffer->collective_batch, channel, op, param,
		send_binding, recv_binding, element_count);
}

static iree_status_t
iree_hal_cuda_new_stream_command_buffer_flush_collectives(
	iree_hal_cuda_new_stream_command_buffer_t *command_buffer) {
	if (iree_hal_collective_batch_is_empty(
			&command_buffer->collective_batch)) {
		return iree_ok_status();
	}
	iree_status_t status = iree_hal_cuda_new_nccl_submit_batch(
		command_buffer->nccl_syms, &command_buffer->collective_batch,
		command_buffer->cu_stream);
	iree_hal_collective_batch_clear(&command_buffer->collective_batch);
	return status;
}

//===----------------------------------------------------------------------===//
// Dispatch
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_cuda_new_stream_command_buffer_dispatch(
	iree_hal_command_buffer_t *base_command_buffer,
	iree_hal_executable_t *executable,
	iree_hal_executable_export_ordinal_t export_ordinal,
	const iree_hal_dispatch_config_t config,
	iree_const_byte_span_t constants,
	iree_hal_buffer_ref_list_t bindings,
	iree_hal_dispatch_flags_t flags) {
	iree_hal_cuda_new_stream_command_buffer_t *command_buffer =
		iree_hal_cuda_new_stream_command_buffer_cast(base_command_buffer);

	if (iree_hal_dispatch_uses_custom_arguments(flags)) {
		return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
			"custom arguments not supported");
	} else if (iree_hal_dispatch_uses_indirect_parameters(flags)) {
		return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
			"indirect parameters not supported");
	}

	IREE_TRACE_ZONE_BEGIN(z0);

	IREE_RETURN_AND_END_ZONE_IF_ERROR(z0,
		iree_hal_cuda_new_stream_command_buffer_flush_collectives(
			command_buffer));

	const iree_hal_cuda_new_kernel_params_t *kernel_params = NULL;
	IREE_RETURN_AND_END_ZONE_IF_ERROR(z0,
		iree_hal_cuda_new_executable_lookup_kernel_params(
			executable, export_ordinal, &kernel_params));

	IREE_RETURN_AND_END_ZONE_IF_ERROR(z0,
		iree_hal_resource_set_insert(command_buffer->resource_set, 1,
			&executable));

	// Build two-level parameter indirection for CUDA.
	iree_host_size_t kernel_params_count =
		kernel_params->binding_count + kernel_params->constant_count;
	iree_host_size_t kernel_params_length =
		kernel_params_count * sizeof(void *);
	iree_host_size_t total_size = kernel_params_length * 2;
	uint8_t *storage_base = NULL;
	IREE_RETURN_AND_END_ZONE_IF_ERROR(z0,
		iree_arena_allocate(&command_buffer->arena, total_size,
			(void **)&storage_base));
	void **params_ptr = (void **)storage_base;
	CUdeviceptr *payload_ptr =
		(CUdeviceptr *)((uint8_t *)params_ptr + kernel_params_length);
	for (size_t i = 0; i < kernel_params_count; i++) {
		params_ptr[i] = &payload_ptr[i];
	}

	// Resolve bindings: device_ptr = base + byte_offset + subspan_offset.
	for (iree_host_size_t i = 0; i < bindings.count; i++) {
		const iree_hal_buffer_ref_t *binding = &bindings.values[i];
		CUdeviceptr device_ptr = 0;
		if (binding->buffer) {
			IREE_RETURN_AND_END_ZONE_IF_ERROR(z0,
				iree_hal_resource_set_insert(
					command_buffer->resource_set, 1,
					&binding->buffer));
			CUdeviceptr device_buffer =
				iree_hal_cuda_new_buffer_device_pointer(
					iree_hal_buffer_allocated_buffer(binding->buffer));
			iree_device_size_t offset =
				iree_hal_buffer_byte_offset(binding->buffer);
			device_ptr = device_buffer + offset + binding->offset;
		}
		payload_ptr[i] = device_ptr;
	}

	// Push constants after bindings.
	for (iree_host_size_t i = 0; i < kernel_params->constant_count; i++) {
		*((uint32_t *)params_ptr[kernel_params->binding_count + i]) =
			((const uint32_t *)constants.data)[i];
	}

	IREE_CUDA_NEW_RETURN_AND_END_ZONE_IF_ERROR(z0,
		command_buffer->syms,
		cuLaunchKernel(kernel_params->function,
			config.workgroup_count[0], config.workgroup_count[1],
			config.workgroup_count[2],
			kernel_params->block_dims[0], kernel_params->block_dims[1],
			kernel_params->block_dims[2],
			0, command_buffer->cu_stream, params_ptr, NULL),
		"cuLaunchKernel");

	IREE_TRACE_ZONE_END(z0);
	return iree_ok_status();
}

static const iree_hal_command_buffer_vtable_t
	iree_hal_cuda_new_stream_command_buffer_vtable = {
		.destroy = iree_hal_cuda_new_stream_command_buffer_destroy,
		.begin = iree_hal_cuda_new_stream_command_buffer_begin,
		.end = iree_hal_cuda_new_stream_command_buffer_end,
		.begin_debug_group =
			iree_hal_cuda_new_stream_command_buffer_begin_debug_group,
		.end_debug_group =
			iree_hal_cuda_new_stream_command_buffer_end_debug_group,
		.execution_barrier =
			iree_hal_cuda_new_stream_command_buffer_execution_barrier,
		.signal_event =
			iree_hal_cuda_new_stream_command_buffer_signal_event,
		.reset_event =
			iree_hal_cuda_new_stream_command_buffer_reset_event,
		.wait_events =
			iree_hal_cuda_new_stream_command_buffer_wait_events,
		.advise_buffer =
			iree_hal_cuda_new_stream_command_buffer_advise_buffer,
		.fill_buffer =
			iree_hal_cuda_new_stream_command_buffer_fill_buffer,
		.update_buffer =
			iree_hal_cuda_new_stream_command_buffer_update_buffer,
		.copy_buffer =
			iree_hal_cuda_new_stream_command_buffer_copy_buffer,
		.collective =
			iree_hal_cuda_new_stream_command_buffer_collective,
		.dispatch = iree_hal_cuda_new_stream_command_buffer_dispatch,
};
