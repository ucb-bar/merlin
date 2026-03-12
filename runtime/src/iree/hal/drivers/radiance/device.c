// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "device.h"

#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "api.h"
#include "iree/base/internal/arena.h"
#include "iree/hal/local/inline_command_buffer.h"
#include "iree/hal/utils/deferred_command_buffer.h"
#include "iree/hal/utils/file_registry.h"
#include "iree/hal/utils/file_transfer.h"
#include "iree/hal/utils/queue_emulation.h"
#include "semaphore.h"
#include "target_caps.h"
#include "transport/transport.h"

#define IREE_HAL_RADIANCE_COMMAND_ARENA_BLOCK_SIZE (32 * 1024)

typedef struct iree_hal_radiance_device_t {
	iree_hal_resource_t resource;
	iree_string_view_t identifier;

	iree_allocator_t host_allocator;
	iree_hal_allocator_t *device_allocator;
	iree_hal_channel_provider_t *channel_provider;

	iree_hal_radiance_device_options_t options;
	iree_hal_radiance_target_caps_t caps;
	iree_hal_radiance_transport_t *transport;

	// Arena block pool used by deferred command buffers.
	iree_arena_block_pool_t command_block_pool;

	// Serializes queue execution and transport synchronization.
	iree_slim_mutex_t queue_mutex;

	// + trailing identifier storage.
} iree_hal_radiance_device_t;

static const iree_hal_device_vtable_t iree_hal_radiance_device_vtable;

static iree_hal_radiance_device_t *iree_hal_radiance_device_cast(
	iree_hal_device_t *base_value) {
	IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_radiance_device_vtable);
	return (iree_hal_radiance_device_t *)base_value;
}

IREE_API_EXPORT void iree_hal_radiance_device_options_initialize(
	iree_hal_radiance_device_options_t *out_options) {
	memset(out_options, 0, sizeof(*out_options));
	out_options->backend = IREE_HAL_RADIANCE_TRANSPORT_BACKEND_AUTO;
	out_options->rpc_socket_path = IREE_SV("/tmp/gluon_rpc.sock");
	out_options->direct_socket_path = IREE_SV("/tmp/gluon_direct.sock");
	out_options->stream_id = 0;
	out_options->regs_per_thread = 32;
	out_options->shmem_per_block = 0;
}

static iree_status_t iree_hal_radiance_device_options_verify(
	const iree_hal_radiance_device_options_t *options) {
	if (options->regs_per_thread == 0) {
		return iree_make_status(
			IREE_STATUS_INVALID_ARGUMENT, "regs_per_thread must be non-zero");
	}
	return iree_ok_status();
}

static void iree_hal_radiance_device_destroy(iree_hal_device_t *base_device) {
	iree_hal_radiance_device_t *device =
		iree_hal_radiance_device_cast(base_device);
	IREE_TRACE_ZONE_BEGIN(z0);

	iree_hal_allocator_release(device->device_allocator);
	iree_hal_channel_provider_release(device->channel_provider);
	iree_hal_radiance_transport_destroy(device->transport);

	iree_slim_mutex_deinitialize(&device->queue_mutex);
	iree_arena_block_pool_deinitialize(&device->command_block_pool);

	iree_allocator_free(device->host_allocator, device);
	IREE_TRACE_ZONE_END(z0);
}

static iree_string_view_t iree_hal_radiance_device_id(
	iree_hal_device_t *base_device) {
	iree_hal_radiance_device_t *device =
		iree_hal_radiance_device_cast(base_device);
	return device->identifier;
}

static iree_allocator_t iree_hal_radiance_device_host_allocator(
	iree_hal_device_t *base_device) {
	iree_hal_radiance_device_t *device =
		iree_hal_radiance_device_cast(base_device);
	return device->host_allocator;
}

static iree_hal_allocator_t *iree_hal_radiance_device_allocator(
	iree_hal_device_t *base_device) {
	iree_hal_radiance_device_t *device =
		iree_hal_radiance_device_cast(base_device);
	return device->device_allocator;
}

static void iree_hal_radiance_device_replace_device_allocator(
	iree_hal_device_t *base_device, iree_hal_allocator_t *new_allocator) {
	iree_hal_radiance_device_t *device =
		iree_hal_radiance_device_cast(base_device);
	iree_hal_allocator_retain(new_allocator);
	iree_hal_allocator_release(device->device_allocator);
	device->device_allocator = new_allocator;
}

static void iree_hal_radiance_device_replace_channel_provider(
	iree_hal_device_t *base_device, iree_hal_channel_provider_t *new_provider) {
	iree_hal_radiance_device_t *device =
		iree_hal_radiance_device_cast(base_device);
	iree_hal_channel_provider_retain(new_provider);
	iree_hal_channel_provider_release(device->channel_provider);
	device->channel_provider = new_provider;
}

static iree_status_t iree_hal_radiance_device_trim(
	iree_hal_device_t *base_device) {
	iree_hal_radiance_device_t *device =
		iree_hal_radiance_device_cast(base_device);
	return iree_hal_allocator_trim(device->device_allocator);
}

static iree_status_t iree_hal_radiance_device_query_i64(
	iree_hal_device_t *base_device, iree_string_view_t category,
	iree_string_view_t key, int64_t *out_value) {
	iree_hal_radiance_device_t *device =
		iree_hal_radiance_device_cast(base_device);
	*out_value = 0;

	if (iree_string_view_equal(category, IREE_SV("hal.device.id"))) {
		*out_value =
			iree_string_view_match_pattern(device->identifier, key) ? 1 : 0;
		return iree_ok_status();
	}

	if (iree_string_view_equal(category, IREE_SV("hal.executable.format"))) {
		const bool supports_muon =
			iree_string_view_match_pattern(IREE_SV("radiance-muon-elf"), key) ||
			iree_string_view_match_pattern(IREE_SV("radiance-elf"), key);
		*out_value = supports_muon ? 1 : 0;
		return iree_ok_status();
	}

	if (iree_string_view_equal(category, IREE_SV("hal.device")) &&
		iree_string_view_equal(key, IREE_SV("concurrency"))) {
		*out_value = (int64_t)device->caps.warp_slots_per_core;
		return iree_ok_status();
	}
	if (iree_string_view_equal(category, IREE_SV("hal.dispatch")) &&
		iree_string_view_equal(key, IREE_SV("concurrency"))) {
		*out_value = (int64_t)device->caps.warp_slots_per_core;
		return iree_ok_status();
	}

	return iree_make_status(IREE_STATUS_NOT_FOUND,
		"unknown device query key '%.*s :: %.*s'", (int)category.size,
		category.data, (int)key.size, key.data);
}

static iree_status_t iree_hal_radiance_device_create_channel(
	iree_hal_device_t *base_device, iree_hal_queue_affinity_t queue_affinity,
	iree_hal_channel_params_t params, iree_hal_channel_t **out_channel) {
	(void)base_device;
	(void)queue_affinity;
	(void)params;
	(void)out_channel;
	return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
		"collective channels are not yet implemented");
}

static iree_status_t iree_hal_radiance_device_create_command_buffer(
	iree_hal_device_t *base_device, iree_hal_command_buffer_mode_t mode,
	iree_hal_command_category_t command_categories,
	iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
	iree_hal_command_buffer_t **out_command_buffer) {
	iree_hal_radiance_device_t *device =
		iree_hal_radiance_device_cast(base_device);
	if (iree_all_bits_set(
			mode, IREE_HAL_COMMAND_BUFFER_MODE_ALLOW_INLINE_EXECUTION)) {
		return iree_hal_inline_command_buffer_create(
			iree_hal_device_allocator(base_device), mode, command_categories,
			queue_affinity, binding_capacity, device->host_allocator,
			out_command_buffer);
	}
	return iree_hal_deferred_command_buffer_create(
		iree_hal_device_allocator(base_device), mode, command_categories,
		queue_affinity, binding_capacity, &device->command_block_pool,
		device->host_allocator, out_command_buffer);
}

static iree_status_t iree_hal_radiance_device_create_event(
	iree_hal_device_t *base_device, iree_hal_queue_affinity_t queue_affinity,
	iree_hal_event_flags_t flags, iree_hal_event_t **out_event) {
	(void)base_device;
	(void)queue_affinity;
	(void)flags;
	(void)out_event;
	return iree_make_status(
		IREE_STATUS_UNIMPLEMENTED, "events are not yet implemented");
}

static iree_status_t iree_hal_radiance_device_create_executable_cache(
	iree_hal_device_t *base_device, iree_string_view_t identifier,
	iree_loop_t loop, iree_hal_executable_cache_t **out_executable_cache) {
	(void)base_device;
	(void)identifier;
	(void)loop;
	(void)out_executable_cache;
	return iree_make_status(
		IREE_STATUS_UNIMPLEMENTED, "executable cache is not yet implemented");
}

static iree_status_t iree_hal_radiance_device_import_file(
	iree_hal_device_t *base_device, iree_hal_queue_affinity_t queue_affinity,
	iree_hal_memory_access_t access, iree_io_file_handle_t *handle,
	iree_hal_external_file_flags_t flags, iree_hal_file_t **out_file) {
	(void)flags;
	return iree_hal_file_from_handle(iree_hal_device_allocator(base_device),
		queue_affinity, access, handle,
		iree_hal_device_host_allocator(base_device), out_file);
}

static iree_status_t iree_hal_radiance_device_create_semaphore(
	iree_hal_device_t *base_device, iree_hal_queue_affinity_t queue_affinity,
	uint64_t initial_value, iree_hal_semaphore_flags_t flags,
	iree_hal_semaphore_t **out_semaphore) {
	iree_hal_radiance_device_t *device =
		iree_hal_radiance_device_cast(base_device);
	return iree_hal_radiance_semaphore_create(queue_affinity, initial_value,
		flags, device->host_allocator, out_semaphore);
}

static iree_hal_semaphore_compatibility_t
iree_hal_radiance_device_query_semaphore_compatibility(
	iree_hal_device_t *base_device, iree_hal_semaphore_t *semaphore) {
	(void)base_device;
	(void)semaphore;
	return IREE_HAL_SEMAPHORE_COMPATIBILITY_HOST_ONLY;
}

static iree_status_t iree_hal_radiance_device_queue_alloca(
	iree_hal_device_t *base_device, iree_hal_queue_affinity_t queue_affinity,
	const iree_hal_semaphore_list_t wait_semaphore_list,
	const iree_hal_semaphore_list_t signal_semaphore_list,
	iree_hal_allocator_pool_t pool, iree_hal_buffer_params_t params,
	iree_device_size_t allocation_size, iree_hal_alloca_flags_t flags,
	iree_hal_buffer_t **IREE_RESTRICT out_buffer) {
	(void)queue_affinity;
	(void)pool;
	(void)flags;
	IREE_RETURN_IF_ERROR(iree_hal_semaphore_list_wait(wait_semaphore_list,
		iree_infinite_timeout(), IREE_HAL_WAIT_FLAG_DEFAULT));
	IREE_RETURN_IF_ERROR(iree_hal_allocator_allocate_buffer(
		iree_hal_device_allocator(base_device), params, allocation_size,
		out_buffer));
	IREE_RETURN_IF_ERROR(iree_hal_semaphore_list_signal(signal_semaphore_list));
	return iree_ok_status();
}

static iree_status_t iree_hal_radiance_device_queue_dealloca(
	iree_hal_device_t *base_device, iree_hal_queue_affinity_t queue_affinity,
	const iree_hal_semaphore_list_t wait_semaphore_list,
	const iree_hal_semaphore_list_t signal_semaphore_list,
	iree_hal_buffer_t *buffer, iree_hal_dealloca_flags_t flags) {
	(void)buffer;
	(void)flags;
	// Preserve timeline behavior with a queue barrier for now.
	return iree_hal_device_queue_barrier(base_device, queue_affinity,
		wait_semaphore_list, signal_semaphore_list, IREE_HAL_EXECUTE_FLAG_NONE);
}

static iree_status_t iree_hal_radiance_device_queue_read(
	iree_hal_device_t *base_device, iree_hal_queue_affinity_t queue_affinity,
	const iree_hal_semaphore_list_t wait_semaphore_list,
	const iree_hal_semaphore_list_t signal_semaphore_list,
	iree_hal_file_t *source_file, uint64_t source_offset,
	iree_hal_buffer_t *target_buffer, iree_device_size_t target_offset,
	iree_device_size_t length, iree_hal_read_flags_t flags) {
	iree_status_t loop_status = iree_ok_status();
	iree_hal_file_transfer_options_t options = {
		.loop = iree_loop_inline(&loop_status),
		.chunk_count = IREE_HAL_FILE_TRANSFER_CHUNK_COUNT_DEFAULT,
		.chunk_size = IREE_HAL_FILE_TRANSFER_CHUNK_SIZE_DEFAULT,
	};
	IREE_RETURN_IF_ERROR(iree_hal_device_queue_read_streaming(base_device,
		queue_affinity, wait_semaphore_list, signal_semaphore_list, source_file,
		source_offset, target_buffer, target_offset, length, flags, options));
	return loop_status;
}

static iree_status_t iree_hal_radiance_device_queue_write(
	iree_hal_device_t *base_device, iree_hal_queue_affinity_t queue_affinity,
	const iree_hal_semaphore_list_t wait_semaphore_list,
	const iree_hal_semaphore_list_t signal_semaphore_list,
	iree_hal_buffer_t *source_buffer, iree_device_size_t source_offset,
	iree_hal_file_t *target_file, uint64_t target_offset,
	iree_device_size_t length, iree_hal_write_flags_t flags) {
	iree_status_t loop_status = iree_ok_status();
	iree_hal_file_transfer_options_t options = {
		.loop = iree_loop_inline(&loop_status),
		.chunk_count = IREE_HAL_FILE_TRANSFER_CHUNK_COUNT_DEFAULT,
		.chunk_size = IREE_HAL_FILE_TRANSFER_CHUNK_SIZE_DEFAULT,
	};
	IREE_RETURN_IF_ERROR(
		iree_hal_device_queue_write_streaming(base_device, queue_affinity,
			wait_semaphore_list, signal_semaphore_list, source_buffer,
			source_offset, target_file, target_offset, length, flags, options));
	return loop_status;
}

static iree_status_t iree_hal_radiance_device_queue_host_call(
	iree_hal_device_t *base_device, iree_hal_queue_affinity_t queue_affinity,
	const iree_hal_semaphore_list_t wait_semaphore_list,
	const iree_hal_semaphore_list_t signal_semaphore_list,
	iree_hal_host_call_t call, const uint64_t args[4],
	iree_hal_host_call_flags_t flags) {
	IREE_RETURN_IF_ERROR(iree_hal_semaphore_list_wait(wait_semaphore_list,
		iree_infinite_timeout(), IREE_HAL_WAIT_FLAG_DEFAULT));

	const bool is_nonblocking =
		iree_any_bit_set(flags, IREE_HAL_HOST_CALL_FLAG_NON_BLOCKING);
	if (is_nonblocking) {
		IREE_RETURN_IF_ERROR(
			iree_hal_semaphore_list_signal(signal_semaphore_list));
	}

	iree_hal_host_call_context_t context = {
		.device = base_device,
		.queue_affinity = queue_affinity,
		.signal_semaphore_list = is_nonblocking
			? iree_hal_semaphore_list_empty()
			: signal_semaphore_list,
	};
	iree_status_t call_status = call.fn(call.user_data, args, &context);
	if (is_nonblocking || iree_status_is_deferred(call_status)) {
		return iree_ok_status();
	}
	if (iree_status_is_ok(call_status)) {
		return iree_hal_semaphore_list_signal(signal_semaphore_list);
	}
	if (!is_nonblocking) {
		iree_hal_semaphore_list_fail(signal_semaphore_list, call_status);
	} else {
		iree_status_ignore(call_status);
	}
	return iree_ok_status();
}

static iree_status_t iree_hal_radiance_device_queue_dispatch(
	iree_hal_device_t *base_device, iree_hal_queue_affinity_t queue_affinity,
	const iree_hal_semaphore_list_t wait_semaphore_list,
	const iree_hal_semaphore_list_t signal_semaphore_list,
	iree_hal_executable_t *executable,
	iree_hal_executable_export_ordinal_t export_ordinal,
	const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
	const iree_hal_buffer_ref_list_t bindings,
	iree_hal_dispatch_flags_t flags) {
	(void)base_device;
	(void)queue_affinity;
	(void)wait_semaphore_list;
	(void)signal_semaphore_list;
	(void)executable;
	(void)export_ordinal;
	(void)config;
	(void)constants;
	(void)bindings;
	(void)flags;
	return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
		"dispatch is not yet implemented for radiance");
}

static iree_status_t iree_hal_radiance_device_apply_deferred_command_buffer(
	iree_hal_radiance_device_t *device,
	iree_hal_command_buffer_t *command_buffer,
	iree_hal_buffer_binding_table_t binding_table) {
	if (!command_buffer || iree_hal_inline_command_buffer_isa(command_buffer)) {
		return iree_ok_status();
	}
	if (!iree_hal_deferred_command_buffer_isa(command_buffer)) {
		return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
			"unsupported command buffer implementation");
	}

	iree_host_size_t storage_size = iree_hal_inline_command_buffer_size(
		iree_hal_command_buffer_mode(command_buffer) |
			IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT |
			IREE_HAL_COMMAND_BUFFER_MODE_ALLOW_INLINE_EXECUTION |
			(iree_hal_buffer_binding_table_is_empty(binding_table)
					? IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED
					: 0),
		/*binding_capacity=*/0);
	iree_byte_span_t storage =
		iree_make_byte_span(iree_alloca(storage_size), storage_size);

	iree_hal_command_buffer_t *inline_command_buffer = NULL;
	IREE_RETURN_IF_ERROR(
		iree_hal_inline_command_buffer_initialize(device->device_allocator,
			iree_hal_command_buffer_mode(command_buffer) |
				IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT |
				IREE_HAL_COMMAND_BUFFER_MODE_ALLOW_INLINE_EXECUTION |
				(iree_hal_buffer_binding_table_is_empty(binding_table)
						? IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED
						: 0),
			iree_hal_command_buffer_allowed_categories(command_buffer),
			IREE_HAL_QUEUE_AFFINITY_ANY, /*binding_capacity=*/0,
			device->host_allocator, storage, &inline_command_buffer));

	iree_status_t status = iree_hal_deferred_command_buffer_apply(
		command_buffer, inline_command_buffer, binding_table);
	iree_hal_inline_command_buffer_deinitialize(inline_command_buffer);
	return status;
}

static iree_status_t iree_hal_radiance_device_queue_execute(
	iree_hal_device_t *base_device, iree_hal_queue_affinity_t queue_affinity,
	const iree_hal_semaphore_list_t wait_semaphore_list,
	const iree_hal_semaphore_list_t signal_semaphore_list,
	iree_hal_command_buffer_t *command_buffer,
	iree_hal_buffer_binding_table_t binding_table,
	iree_hal_execute_flags_t flags) {
	(void)queue_affinity;
	(void)flags;
	iree_hal_radiance_device_t *device =
		iree_hal_radiance_device_cast(base_device);

	if (command_buffer &&
		iree_any_bit_set(
			iree_hal_command_buffer_allowed_categories(command_buffer),
			IREE_HAL_COMMAND_CATEGORY_DISPATCH)) {
		return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
			"dispatch command buffers are not yet supported");
	}

	IREE_RETURN_IF_ERROR(iree_hal_semaphore_list_wait(wait_semaphore_list,
		iree_infinite_timeout(), IREE_HAL_WAIT_FLAG_DEFAULT));

	iree_slim_mutex_lock(&device->queue_mutex);
	iree_status_t status =
		iree_hal_radiance_device_apply_deferred_command_buffer(
			device, command_buffer, binding_table);
	if (iree_status_is_ok(status)) {
		status = iree_hal_radiance_transport_synchronize(
			device->transport, device->options.stream_id);
	}
	if (iree_status_is_ok(status)) {
		status = iree_hal_semaphore_list_signal(signal_semaphore_list);
	} else {
		iree_hal_semaphore_list_fail(
			signal_semaphore_list, iree_status_clone(status));
	}
	iree_slim_mutex_unlock(&device->queue_mutex);
	return status;
}

static iree_status_t iree_hal_radiance_device_queue_flush(
	iree_hal_device_t *base_device, iree_hal_queue_affinity_t queue_affinity) {
	(void)queue_affinity;
	iree_hal_radiance_device_t *device =
		iree_hal_radiance_device_cast(base_device);
	return iree_hal_radiance_transport_synchronize(
		device->transport, device->options.stream_id);
}

static iree_status_t iree_hal_radiance_device_wait_any_semaphores(
	const iree_hal_semaphore_list_t semaphore_list, iree_timeout_t timeout) {
	iree_convert_timeout_to_absolute(&timeout);
	const iree_time_t deadline_ns = iree_timeout_as_deadline_ns(timeout);
	for (;;) {
		bool any_signaled = false;
		for (iree_host_size_t i = 0; i < semaphore_list.count; ++i) {
			uint64_t current_value = 0;
			iree_status_t status = iree_hal_semaphore_query(
				semaphore_list.semaphores[i], &current_value);
			if (!iree_status_is_ok(status)) {
				iree_status_ignore(status);
				return iree_status_from_code(IREE_STATUS_ABORTED);
			}
			if (current_value >= semaphore_list.payload_values[i]) {
				any_signaled = true;
				break;
			}
		}
		if (any_signaled) {
			return iree_ok_status();
		}

		if (iree_timeout_is_immediate(timeout) ||
			iree_time_now() >= deadline_ns) {
			return iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
		}

		iree_time_t sleep_deadline_ns = iree_time_now() + 1000000; // 1ms
		if (sleep_deadline_ns > deadline_ns) {
			sleep_deadline_ns = deadline_ns;
		}
		iree_wait_until(sleep_deadline_ns);
	}
}

static iree_status_t iree_hal_radiance_device_wait_semaphores(
	iree_hal_device_t *base_device, iree_hal_wait_mode_t wait_mode,
	const iree_hal_semaphore_list_t semaphore_list, iree_timeout_t timeout,
	iree_hal_wait_flags_t flags) {
	(void)base_device;
	if (wait_mode == IREE_HAL_WAIT_MODE_ALL) {
		return iree_hal_semaphore_list_wait(semaphore_list, timeout, flags);
	}
	return iree_hal_radiance_device_wait_any_semaphores(
		semaphore_list, timeout);
}

static iree_status_t iree_hal_radiance_device_profiling_begin(
	iree_hal_device_t *base_device,
	const iree_hal_device_profiling_options_t *options) {
	(void)base_device;
	(void)options;
	return iree_ok_status();
}

static iree_status_t iree_hal_radiance_device_profiling_flush(
	iree_hal_device_t *base_device) {
	(void)base_device;
	return iree_ok_status();
}

static iree_status_t iree_hal_radiance_device_profiling_end(
	iree_hal_device_t *base_device) {
	(void)base_device;
	return iree_ok_status();
}

static const iree_hal_device_vtable_t iree_hal_radiance_device_vtable = {
	.destroy = iree_hal_radiance_device_destroy,
	.id = iree_hal_radiance_device_id,
	.host_allocator = iree_hal_radiance_device_host_allocator,
	.device_allocator = iree_hal_radiance_device_allocator,
	.replace_device_allocator =
		iree_hal_radiance_device_replace_device_allocator,
	.replace_channel_provider =
		iree_hal_radiance_device_replace_channel_provider,
	.trim = iree_hal_radiance_device_trim,
	.query_i64 = iree_hal_radiance_device_query_i64,
	.create_channel = iree_hal_radiance_device_create_channel,
	.create_command_buffer = iree_hal_radiance_device_create_command_buffer,
	.create_event = iree_hal_radiance_device_create_event,
	.create_executable_cache = iree_hal_radiance_device_create_executable_cache,
	.import_file = iree_hal_radiance_device_import_file,
	.create_semaphore = iree_hal_radiance_device_create_semaphore,
	.query_semaphore_compatibility =
		iree_hal_radiance_device_query_semaphore_compatibility,
	.queue_alloca = iree_hal_radiance_device_queue_alloca,
	.queue_dealloca = iree_hal_radiance_device_queue_dealloca,
	.queue_fill = iree_hal_device_queue_emulated_fill,
	.queue_update = iree_hal_device_queue_emulated_update,
	.queue_copy = iree_hal_device_queue_emulated_copy,
	.queue_read = iree_hal_radiance_device_queue_read,
	.queue_write = iree_hal_radiance_device_queue_write,
	.queue_host_call = iree_hal_radiance_device_queue_host_call,
	.queue_dispatch = iree_hal_radiance_device_queue_dispatch,
	.queue_execute = iree_hal_radiance_device_queue_execute,
	.queue_flush = iree_hal_radiance_device_queue_flush,
	.wait_semaphores = iree_hal_radiance_device_wait_semaphores,
	.profiling_begin = iree_hal_radiance_device_profiling_begin,
	.profiling_flush = iree_hal_radiance_device_profiling_flush,
	.profiling_end = iree_hal_radiance_device_profiling_end,
};

IREE_API_EXPORT iree_status_t iree_hal_radiance_device_create(
	iree_string_view_t identifier,
	const iree_hal_radiance_device_options_t *options,
	iree_allocator_t host_allocator, iree_hal_device_t **out_device) {
	IREE_ASSERT_ARGUMENT(options);
	IREE_ASSERT_ARGUMENT(out_device);
	IREE_TRACE_ZONE_BEGIN(z0);
	*out_device = NULL;

	iree_hal_radiance_target_caps_t caps;
	iree_hal_radiance_target_caps_initialize_defaults(&caps);
	if (options->regs_per_thread > caps.max_registers_per_thread) {
		IREE_TRACE_ZONE_END(z0);
		return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
			"requested regs_per_thread=%u exceeds target max=%u",
			options->regs_per_thread, caps.max_registers_per_thread);
	}

	IREE_RETURN_AND_END_ZONE_IF_ERROR(
		z0, iree_hal_radiance_device_options_verify(options));

	iree_hal_radiance_transport_t *transport = NULL;
	IREE_RETURN_AND_END_ZONE_IF_ERROR(z0,
		iree_hal_radiance_transport_create(
			options, host_allocator, &transport));

	iree_hal_radiance_device_t *device = NULL;
	const iree_host_size_t total_size = sizeof(*device) + identifier.size;
	IREE_RETURN_AND_END_ZONE_IF_ERROR(z0,
		iree_allocator_malloc(host_allocator, total_size, (void **)&device));
	memset(device, 0, total_size);

	iree_hal_resource_initialize(
		&iree_hal_radiance_device_vtable, &device->resource);
	iree_string_view_append_to_buffer(identifier, &device->identifier,
		(char *)device + total_size - identifier.size);
	device->host_allocator = host_allocator;
	device->options = *options;
	device->caps = caps;
	device->transport = transport;
	iree_arena_block_pool_initialize(IREE_HAL_RADIANCE_COMMAND_ARENA_BLOCK_SIZE,
		host_allocator, &device->command_block_pool);
	iree_slim_mutex_initialize(&device->queue_mutex);

	iree_status_t status = iree_hal_allocator_create_heap(IREE_SV("radiance"),
		host_allocator, host_allocator, &device->device_allocator);
	if (!iree_status_is_ok(status)) {
		iree_hal_device_release((iree_hal_device_t *)device);
		IREE_TRACE_ZONE_END(z0);
		return status;
	}

	*out_device = (iree_hal_device_t *)device;
	IREE_TRACE_ZONE_END(z0);
	return iree_ok_status();
}
