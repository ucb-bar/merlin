// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// QNN dispatch command buffer. This is the "target" buffer that
// iree_hal_deferred_command_buffer_apply replays against during
// queue_execute. Only the dispatch handler does real work — it translates
// IREE HAL dispatch into QnnGraph_executeAsync (with a graphExecute fall-
// back when the backend's QnnInterface doesn't expose the async fn).
//
// Architecture:
//   - The async path uses Qnn_NotifyFn_t to track completion. Each
//     dispatch allocates a small "pending op" descriptor on the device's
//     queue arena, hands it to QNN as notifyParam, and signals a
//     condition variable when the kernel finishes.
//   - The current dispatch handler still waits for completion before
//     returning (i.e. one dispatch finishes before the next is recorded
//     into the same command buffer). The next graduation step is to
//     drop that wait so back-to-back dispatches in different command
//     buffers can pipeline on the QNN backend's internal queue.
//
// All other vtable methods are no-ops because:
//   - QNN context binaries are pre-compiled and don't need fill/copy/update
//     ops at the HAL level (the graph itself contains all the compute).
//   - Synchronisation is a no-op at the HAL command-buffer layer because
//     dispatch waits for its own kernel; events/barriers are unnecessary.
//   - Buffer advise/collective ops aren't applicable to a single-graph
//     execute path.

#include "qnn_command_buffer.h"

#include <pthread.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"

// QNN SDK.
#include "QnnCommon.h"
#include "QnnGraph.h"
#include "QnnInterface.h"
#include "QnnTypes.h"

#include "qnn_executable.h"

// Maximum bindings we accept per dispatch. QNN graphs typically have <16
// inputs+outputs combined (a conv has 3-4, a matmul 2-3, etc).
#define IREE_HAL_QNN_MAX_DISPATCH_BINDINGS 32

static bool iree_hal_qnn_trace_enabled(void) {
	const char *v = getenv("IREE_HAL_QNN_TRACE");
	return v && v[0] && strcmp(v, "0") != 0;
}

static uint32_t iree_hal_qnn_dtype_size_bytes(Qnn_DataType_t data_type) {
	switch (data_type) {
		case QNN_DATATYPE_INT_8:
		case QNN_DATATYPE_UINT_8:
		case QNN_DATATYPE_SFIXED_POINT_8:
		case QNN_DATATYPE_UFIXED_POINT_8:
		case QNN_DATATYPE_BOOL_8:
			return 1;
		case QNN_DATATYPE_INT_16:
		case QNN_DATATYPE_UINT_16:
		case QNN_DATATYPE_FLOAT_16:
		case QNN_DATATYPE_SFIXED_POINT_16:
		case QNN_DATATYPE_UFIXED_POINT_16:
			return 2;
		case QNN_DATATYPE_INT_32:
		case QNN_DATATYPE_UINT_32:
		case QNN_DATATYPE_FLOAT_32:
		case QNN_DATATYPE_SFIXED_POINT_32:
		case QNN_DATATYPE_UFIXED_POINT_32:
			return 4;
		case QNN_DATATYPE_INT_64:
		case QNN_DATATYPE_UINT_64:
		case QNN_DATATYPE_FLOAT_64:
			return 8;
		default:
			return 0;
	}
}

static bool iree_hal_qnn_tensor_byte_size(
	const Qnn_Tensor_t *tensor, uint32_t *out_size) {
	if (!tensor || tensor->version != QNN_TENSOR_VERSION_1)
		return false;
	uint32_t elem_size = iree_hal_qnn_dtype_size_bytes(tensor->v1.dataType);
	if (elem_size == 0)
		return false;
	uint64_t count = 1;
	for (uint32_t i = 0; i < tensor->v1.rank; ++i) {
		count *= tensor->v1.dimensions[i];
		if (count > UINT32_MAX / elem_size)
			return false;
	}
	*out_size = (uint32_t)(count * elem_size);
	return true;
}

// -----------------------------------------------------------------------------
// Async dispatch completion plumbing
// -----------------------------------------------------------------------------
//
// Per-dispatch completion record passed to QNN as `notifyParam`. QNN calls
// our notify_fn from a backend-internal worker thread when the kernel
// finishes; we set `done` and broadcast the cond_var so the original
// dispatch caller can wake up.
//
// Allocated on the dispatch caller's stack (lives only for the duration
// of one QnnGraph_executeAsync call). The pthread primitives are
// initialized + destroyed per call — cheap (~µs) compared to the kernel
// runtime (ms).

typedef struct iree_hal_qnn_async_completion_t {
	pthread_mutex_t mu;
	pthread_cond_t cv;
	bool done;
	Qnn_NotifyStatus_t status;
} iree_hal_qnn_async_completion_t;

static void iree_hal_qnn_async_completion_init(
	iree_hal_qnn_async_completion_t *c) {
	pthread_mutex_init(&c->mu, NULL);
	pthread_cond_init(&c->cv, NULL);
	c->done = false;
	Qnn_NotifyStatus_t init = QNN_NOTIFY_STATUS_INIT;
	c->status = init;
}

static void iree_hal_qnn_async_completion_deinit(
	iree_hal_qnn_async_completion_t *c) {
	pthread_cond_destroy(&c->cv);
	pthread_mutex_destroy(&c->mu);
}

// Called by QNN from its internal worker thread when execution finishes.
// Just records status + signals; the dispatch thread is waiting on the
// cond var.
static void iree_hal_qnn_async_notify_fn(
	void *notifyParam, Qnn_NotifyStatus_t status) {
	iree_hal_qnn_async_completion_t *c =
		(iree_hal_qnn_async_completion_t *)notifyParam;
	pthread_mutex_lock(&c->mu);
	c->status = status;
	c->done = true;
	pthread_cond_broadcast(&c->cv);
	pthread_mutex_unlock(&c->mu);
}

static Qnn_NotifyStatus_t iree_hal_qnn_async_completion_wait(
	iree_hal_qnn_async_completion_t *c) {
	pthread_mutex_lock(&c->mu);
	while (!c->done) {
		pthread_cond_wait(&c->cv, &c->mu);
	}
	Qnn_NotifyStatus_t s = c->status;
	pthread_mutex_unlock(&c->mu);
	return s;
}

// One pending dispatch's bookkeeping: completion + per-binding host
// mappings. The mappings must outlive the QnnGraph_executeAsync call
// (the QNN backend reads clientBuf pointers at hardware-execute time,
// not at submit time). We keep them alive in a per-command-buffer pool
// and unmap at end_command_buffer time, after fencing the chain.
//
// Per-buffer fencing: each pending record records the underlying buffer
// pointers it writes (output bindings). When a new dispatch reads any
// buffer that a prior pending writes, we wait that pending record before
// recording, ensuring the read sees committed data. Pendings whose
// outputs aren't read by anyone in this CB just flush at end-of-CB.
typedef struct iree_hal_qnn_pending_t {
	iree_hal_qnn_async_completion_t completion;
	iree_hal_buffer_mapping_t mappings[IREE_HAL_QNN_MAX_DISPATCH_BINDINGS];
	iree_host_size_t mapping_count;
	// Snapshot of output buffers written by this dispatch — opaque
	// pointer-equality with future dispatches' input buffers is the
	// dependency check. We compare device-allocator-level buffer
	// identity, not bytes; same buffer used by N readers fences exactly
	// once because we mark this pending "fenced" after the first wait.
	iree_hal_buffer_t *output_buffers[IREE_HAL_QNN_MAX_DISPATCH_BINDINGS];
	iree_host_size_t output_buffer_count;
	// Snapshot of which path this dispatch took. If async submit failed we
	// already finished synchronously; the end-of-CB fence skips waiting.
	bool used_async;
	// Set true once we've waited on this pending (either by per-buffer
	// fence triggered from a downstream dispatch, or end-of-CB fence).
	// Idempotent: repeated waits collapse to one cond_wait, repeated
	// unmaps would double-free so we gate on this flag.
	bool fenced;
} iree_hal_qnn_pending_t;

// Soft cap on chained async dispatches per command buffer. Beyond this
// we synchronously fence + reset, trading a bit of per-batch latency
// for bounded memory. 64 covers dronet's 20 dispatches with headroom;
// most heterogeneous chunks are smaller. Bumping later is one-line.
#define IREE_HAL_QNN_MAX_PENDING 64

typedef struct iree_hal_qnn_command_buffer_t {
	iree_hal_command_buffer_t base;
	iree_allocator_t host_allocator;
	const QnnInterface_t *qnn_interface;

	// Pool of pending-dispatch records. completion mutex/cond initialised
	// once per record at command-buffer create — one pthread-init pair per
	// record, amortised across however many dispatches replay through
	// this CB. The done flag is reset at each new dispatch; the
	// mu/cv stay alive across resets.
	iree_hal_qnn_pending_t pending[IREE_HAL_QNN_MAX_PENDING];
	iree_host_size_t pending_count;
	bool pending_inited;
} iree_hal_qnn_command_buffer_t;

static const iree_hal_command_buffer_vtable_t
	iree_hal_qnn_command_buffer_vtable;

static iree_hal_qnn_command_buffer_t *iree_hal_qnn_command_buffer_cast(
	iree_hal_command_buffer_t *base_value) {
	IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_qnn_command_buffer_vtable);
	return (iree_hal_qnn_command_buffer_t *)base_value;
}

iree_status_t iree_hal_qnn_command_buffer_create(
	iree_hal_allocator_t *device_allocator, iree_hal_command_buffer_mode_t mode,
	iree_hal_command_category_t command_categories,
	iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
	const void *qnn_interface, iree_allocator_t host_allocator,
	iree_hal_command_buffer_t **out_command_buffer) {
	IREE_ASSERT_ARGUMENT(out_command_buffer);
	IREE_ASSERT_ARGUMENT(qnn_interface);
	*out_command_buffer = NULL;

	iree_host_size_t total_size = sizeof(iree_hal_qnn_command_buffer_t) +
		iree_hal_command_buffer_validation_state_size(mode, binding_capacity);

	iree_hal_qnn_command_buffer_t *command_buffer = NULL;
	IREE_RETURN_IF_ERROR(iree_allocator_malloc(
		host_allocator, total_size, (void **)&command_buffer));
	iree_hal_command_buffer_initialize(device_allocator, mode,
		command_categories, queue_affinity, binding_capacity,
		(uint8_t *)command_buffer + sizeof(*command_buffer),
		&iree_hal_qnn_command_buffer_vtable, &command_buffer->base);
	command_buffer->host_allocator = host_allocator;
	command_buffer->qnn_interface = (const QnnInterface_t *)qnn_interface;
	for (iree_host_size_t i = 0; i < IREE_HAL_QNN_MAX_PENDING; ++i) {
		iree_hal_qnn_async_completion_init(
			&command_buffer->pending[i].completion);
		command_buffer->pending[i].mapping_count = 0;
		command_buffer->pending[i].output_buffer_count = 0;
		command_buffer->pending[i].used_async = false;
		command_buffer->pending[i].fenced = false;
	}
	command_buffer->pending_count = 0;
	command_buffer->pending_inited = true;

	*out_command_buffer = &command_buffer->base;
	return iree_ok_status();
}

static void iree_hal_qnn_command_buffer_destroy(
	iree_hal_command_buffer_t *base_command_buffer) {
	iree_hal_qnn_command_buffer_t *command_buffer =
		iree_hal_qnn_command_buffer_cast(base_command_buffer);
	if (command_buffer->pending_inited) {
		for (iree_host_size_t i = 0; i < IREE_HAL_QNN_MAX_PENDING; ++i) {
			iree_hal_qnn_async_completion_deinit(
				&command_buffer->pending[i].completion);
		}
	}
	iree_allocator_free(command_buffer->host_allocator, command_buffer);
}

bool iree_hal_qnn_command_buffer_isa(
	iree_hal_command_buffer_t *command_buffer) {
	return iree_hal_resource_is(
		&command_buffer->resource, &iree_hal_qnn_command_buffer_vtable);
}

// ---------------------------------------------------------------------------
// No-op vtable methods.
// ---------------------------------------------------------------------------

// Fence all pending async dispatches recorded into this command buffer
// and release their host mappings. Called from end_command_buffer
// (the normal path) and from dispatch() if the pending pool overflows.
//
// Each pending record either holds an outstanding graphExecuteAsync
// (used_async=true) or already finished synchronously inline
// (used_async=false). For async records we wait the cond_var; for sync
// records the wait is a no-op. After the wait we unmap the host
// mappings — this is the point where it's safe to release them because
// the QNN backend has finished reading clientBuf pointers.
// Fence one pending record (idempotent). Used both by per-buffer
// fencing (downstream dispatch reads upstream's output) and end-of-CB.
static iree_status_t iree_hal_qnn_fence_one(iree_hal_qnn_pending_t *pending) {
	if (pending->fenced)
		return iree_ok_status();
	iree_status_t status = iree_ok_status();
	if (pending->used_async) {
		Qnn_NotifyStatus_t notify_status =
			iree_hal_qnn_async_completion_wait(&pending->completion);
		if (notify_status.error != QNN_SUCCESS) {
			status = iree_make_status(IREE_STATUS_INTERNAL,
				"QnnGraph_executeAsync notify reported rc=%lld",
				(long long)notify_status.error);
		}
	}
	for (iree_host_size_t i = 0; i < pending->mapping_count; ++i) {
		iree_status_t unmap_status =
			iree_hal_buffer_unmap_range(&pending->mappings[i]);
		if (iree_status_is_ok(status) && !iree_status_is_ok(unmap_status)) {
			status = unmap_status;
		} else {
			iree_status_ignore(unmap_status);
		}
	}
	pending->mapping_count = 0;
	pending->used_async = false;
	pending->output_buffer_count = 0;
	pending->fenced = true;
	return status;
}

static iree_status_t iree_hal_qnn_command_buffer_fence_pending(
	iree_hal_qnn_command_buffer_t *command_buffer) {
	iree_status_t status = iree_ok_status();
	for (iree_host_size_t p = 0; p < command_buffer->pending_count; ++p) {
		iree_status_t one = iree_hal_qnn_fence_one(&command_buffer->pending[p]);
		if (iree_status_is_ok(status) && !iree_status_is_ok(one)) {
			status = one;
		} else if (!iree_status_is_ok(one)) {
			iree_status_ignore(one);
		}
		// Reset fenced flag for next CB recording phase.
		command_buffer->pending[p].fenced = false;
	}
	command_buffer->pending_count = 0;
	return status;
}

// Walk the pending list and fence any record that wrote a buffer the
// new dispatch's input bindings will read. Returns when all such
// upstream dependencies are resolved. Pendings whose outputs aren't
// read remain in flight (and flush at end-of-CB).
//
// Buffer identity is the iree_hal_buffer_t pointer — same buffer used
// by N readers fences exactly once because fence_one sets `fenced=true`
// (subsequent reader hits the early-return).
static iree_status_t iree_hal_qnn_fence_dependencies(
	iree_hal_qnn_command_buffer_t *command_buffer,
	iree_host_size_t num_input_bindings, iree_hal_buffer_ref_list_t bindings) {
	for (iree_host_size_t i = 0; i < num_input_bindings; ++i) {
		iree_hal_buffer_t *in_buffer = bindings.values[i].buffer;
		if (!in_buffer)
			continue;
		for (iree_host_size_t p = 0; p < command_buffer->pending_count; ++p) {
			iree_hal_qnn_pending_t *pending = &command_buffer->pending[p];
			if (pending->fenced)
				continue;
			bool overlaps = false;
			for (iree_host_size_t b = 0; b < pending->output_buffer_count;
				 ++b) {
				if (pending->output_buffers[b] == in_buffer) {
					overlaps = true;
					break;
				}
			}
			if (overlaps) {
				IREE_RETURN_IF_ERROR(iree_hal_qnn_fence_one(pending));
			}
		}
	}
	return iree_ok_status();
}

static iree_status_t iree_hal_qnn_command_buffer_begin(
	iree_hal_command_buffer_t *base_command_buffer) {
	// Defensive: a reused command buffer (mode=allow-inline-execution) may
	// call begin again with stale pending entries from a prior replay. The
	// fence here is a no-op when pending_count is already 0.
	iree_hal_qnn_command_buffer_t *command_buffer =
		iree_hal_qnn_command_buffer_cast(base_command_buffer);
	return iree_hal_qnn_command_buffer_fence_pending(command_buffer);
}

static iree_status_t iree_hal_qnn_command_buffer_end(
	iree_hal_command_buffer_t *base_command_buffer) {
	// The chain-fence: every dispatch recorded into this command buffer
	// submitted graphExecuteAsync without waiting; this is where we
	// collect their completions in submission order, releasing host
	// mappings after each. Replay code in
	// iree_hal_deferred_command_buffer_apply calls end after the last dispatch,
	// so the user-visible queue_execute returns only when all async dispatches
	// have flushed.
	iree_hal_qnn_command_buffer_t *command_buffer =
		iree_hal_qnn_command_buffer_cast(base_command_buffer);
	return iree_hal_qnn_command_buffer_fence_pending(command_buffer);
}

static iree_status_t iree_hal_qnn_command_buffer_begin_debug_group(
	iree_hal_command_buffer_t *base_command_buffer, iree_string_view_t label,
	iree_hal_label_color_t label_color,
	const iree_hal_label_location_t *location) {
	return iree_ok_status();
}

static iree_status_t iree_hal_qnn_command_buffer_end_debug_group(
	iree_hal_command_buffer_t *base_command_buffer) {
	return iree_ok_status();
}

static iree_status_t iree_hal_qnn_command_buffer_execution_barrier(
	iree_hal_command_buffer_t *base_command_buffer,
	iree_hal_execution_stage_t source_stage_mask,
	iree_hal_execution_stage_t target_stage_mask,
	iree_hal_execution_barrier_flags_t flags,
	iree_host_size_t memory_barrier_count,
	const iree_hal_memory_barrier_t *memory_barriers,
	iree_host_size_t buffer_barrier_count,
	const iree_hal_buffer_barrier_t *buffer_barriers) {
	return iree_ok_status();
}

static iree_status_t iree_hal_qnn_command_buffer_signal_event(
	iree_hal_command_buffer_t *base_command_buffer, iree_hal_event_t *event,
	iree_hal_execution_stage_t source_stage_mask) {
	return iree_ok_status();
}

static iree_status_t iree_hal_qnn_command_buffer_reset_event(
	iree_hal_command_buffer_t *base_command_buffer, iree_hal_event_t *event,
	iree_hal_execution_stage_t source_stage_mask) {
	return iree_ok_status();
}

static iree_status_t iree_hal_qnn_command_buffer_wait_events(
	iree_hal_command_buffer_t *base_command_buffer,
	iree_host_size_t event_count, const iree_hal_event_t **events,
	iree_hal_execution_stage_t source_stage_mask,
	iree_hal_execution_stage_t target_stage_mask,
	iree_host_size_t memory_barrier_count,
	const iree_hal_memory_barrier_t *memory_barriers,
	iree_host_size_t buffer_barrier_count,
	const iree_hal_buffer_barrier_t *buffer_barriers) {
	return iree_ok_status();
}

static iree_status_t iree_hal_qnn_command_buffer_advise_buffer(
	iree_hal_command_buffer_t *base_command_buffer,
	iree_hal_buffer_ref_t buffer_ref, iree_hal_memory_advise_flags_t flags,
	uint64_t arg0, uint64_t arg1) {
	return iree_ok_status();
}

// HAL transfer ops are eagerly evaluated against host-mapped buffers — the
// IREE runtime emits these to move tensor IO around the QNN graph dispatch.
// The QNN backend's compute is in the precompiled context binary; what
// fill / copy / update need to do at the HAL layer is just shuffle host
// memory. This matches the inline-CPU command-buffer semantics: dispatch
// is the only async op, everything else is a host-thread memcpy.
//
// We rely on `iree_hal_buffer_map_*` because the QNN HAL device hands out
// buffers from the standard host-task allocator (see qnn_device.c —
// `device_allocator` is the local heap), which is always host-mappable.
//
// Each transfer fences any pending async dispatch that wrote the source
// (for copy) or target (for fill/update) — without this, an inline
// memcpy can race a still-running graphExecuteAsync. Fence is a no-op
// when no upstream pending writes the involved buffer; the per-buffer
// dependency tracker only blocks on actual writers.
//
// fence_buffer_writers walks the pending list and fences any record
// whose output_buffers includes `target` — same logic as
// iree_hal_qnn_fence_dependencies but for a single buffer.
static iree_status_t iree_hal_qnn_fence_buffer_writers(
	iree_hal_qnn_command_buffer_t *command_buffer, iree_hal_buffer_t *target) {
	if (!target)
		return iree_ok_status();
	for (iree_host_size_t p = 0; p < command_buffer->pending_count; ++p) {
		iree_hal_qnn_pending_t *pending = &command_buffer->pending[p];
		if (pending->fenced)
			continue;
		for (iree_host_size_t b = 0; b < pending->output_buffer_count; ++b) {
			if (pending->output_buffers[b] == target) {
				IREE_RETURN_IF_ERROR(iree_hal_qnn_fence_one(pending));
				break;
			}
		}
	}
	return iree_ok_status();
}

static iree_status_t iree_hal_qnn_command_buffer_fill_buffer(
	iree_hal_command_buffer_t *base_command_buffer,
	iree_hal_buffer_ref_t target_ref, const void *pattern,
	iree_host_size_t pattern_length, iree_hal_fill_flags_t flags) {
	iree_hal_qnn_command_buffer_t *command_buffer =
		iree_hal_qnn_command_buffer_cast(base_command_buffer);
	IREE_RETURN_IF_ERROR(
		iree_hal_qnn_fence_buffer_writers(command_buffer, target_ref.buffer));
	return iree_hal_buffer_map_fill(target_ref.buffer, target_ref.offset,
		target_ref.length, pattern, pattern_length);
}

static iree_status_t iree_hal_qnn_command_buffer_update_buffer(
	iree_hal_command_buffer_t *base_command_buffer, const void *source_buffer,
	iree_host_size_t source_offset, iree_hal_buffer_ref_t target_ref,
	iree_hal_update_flags_t flags) {
	iree_hal_qnn_command_buffer_t *command_buffer =
		iree_hal_qnn_command_buffer_cast(base_command_buffer);
	IREE_RETURN_IF_ERROR(
		iree_hal_qnn_fence_buffer_writers(command_buffer, target_ref.buffer));
	return iree_hal_buffer_map_write(target_ref.buffer, target_ref.offset,
		(const uint8_t *)source_buffer + source_offset, target_ref.length);
}

static iree_status_t iree_hal_qnn_command_buffer_copy_buffer(
	iree_hal_command_buffer_t *base_command_buffer,
	iree_hal_buffer_ref_t source_ref, iree_hal_buffer_ref_t target_ref,
	iree_hal_copy_flags_t flags) {
	iree_hal_qnn_command_buffer_t *command_buffer =
		iree_hal_qnn_command_buffer_cast(base_command_buffer);
	// Fence both ends: source must be fully written, target must be ready
	// to receive (in case it's also a pending writer's target via aliasing).
	IREE_RETURN_IF_ERROR(
		iree_hal_qnn_fence_buffer_writers(command_buffer, source_ref.buffer));
	IREE_RETURN_IF_ERROR(
		iree_hal_qnn_fence_buffer_writers(command_buffer, target_ref.buffer));
	return iree_hal_buffer_map_copy(source_ref.buffer, source_ref.offset,
		target_ref.buffer, target_ref.offset, target_ref.length);
}

static iree_status_t iree_hal_qnn_command_buffer_collective(
	iree_hal_command_buffer_t *base_command_buffer, iree_hal_channel_t *channel,
	iree_hal_collective_op_t op, uint32_t param, iree_hal_buffer_ref_t send_ref,
	iree_hal_buffer_ref_t recv_ref, iree_device_size_t element_count) {
	return iree_make_status(
		IREE_STATUS_UNIMPLEMENTED, "collective ops not supported on QNN");
}

// ---------------------------------------------------------------------------
// dispatch — translates IREE HAL dispatch into QnnGraph_execute.
// ---------------------------------------------------------------------------

static iree_status_t iree_hal_qnn_command_buffer_dispatch(
	iree_hal_command_buffer_t *base_command_buffer,
	iree_hal_executable_t *executable,
	iree_hal_executable_export_ordinal_t export_ordinal,
	const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
	iree_hal_buffer_ref_list_t bindings, iree_hal_dispatch_flags_t flags) {
	iree_hal_qnn_command_buffer_t *command_buffer =
		iree_hal_qnn_command_buffer_cast(base_command_buffer);

	if (!iree_hal_qnn_executable_isa(executable)) {
		return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
			"QNN command buffer received non-QNN executable in dispatch");
	}

	// Resolve the dispatch's export_ordinal to a Qnn_GraphHandle_t. The
	// executable's create-time enumeration via QnnSystemContext populated
	// the cache in graph order — ordinal=0 is "the first graph in the
	// .qnn-ctx blob", which for single-graph chunks (the common case) is
	// exactly what we want.
	iree_hal_qnn_graph_handle_t graph_handle =
		iree_hal_qnn_executable_lookup_graph_by_ordinal(
			executable, (iree_host_size_t)export_ordinal);
	if (graph_handle == NULL) {
		return iree_make_status(IREE_STATUS_NOT_FOUND,
			"QNN executable has no graph at ordinal %u (graphs were not "
			"enumerated; check that libQnnSystem.so loaded at "
			"executable_create and the .qnn-ctx is well-formed)",
			(unsigned)export_ordinal);
	}

	if (bindings.count > IREE_HAL_QNN_MAX_DISPATCH_BINDINGS) {
		return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
			"QNN dispatch binding count %" PRIhsz " exceeds maximum %d",
			bindings.count, IREE_HAL_QNN_MAX_DISPATCH_BINDINGS);
	}

	// Fetch this graph's expected I/O prototypes (name, dataType, rank,
	// dimensions) from the executable's QnnSystemContext-derived cache, then
	// build local Qnn_Tensor_t arrays whose clientBuf points at the IREE
	// buffer mapping. IREE's HAL pipeline layout convention is:
	//   bindings[0..num_inputs)    = inputs (READ-only)
	//   bindings[num_inputs..end)  = outputs (WRITE-able)
	// which matches the order in #hal.pipeline.binding<storage_buffer, ...>.
	iree_hal_qnn_tensor_proto_t *protos_in = NULL;
	iree_hal_qnn_tensor_proto_t *protos_out = NULL;
	iree_host_size_t num_inputs = 0;
	iree_host_size_t num_outputs = 0;
	IREE_RETURN_IF_ERROR(iree_hal_qnn_executable_get_graph_io(executable,
		(iree_host_size_t)export_ordinal, &protos_in, &num_inputs, &protos_out,
		&num_outputs));
	const Qnn_Tensor_t *proto_inputs = (const Qnn_Tensor_t *)protos_in;
	const Qnn_Tensor_t *proto_outputs = (const Qnn_Tensor_t *)protos_out;

	if (iree_hal_qnn_trace_enabled()) {
		fprintf(stderr,
			"[qnn-cmd] dispatch ordinal=%u bindings=%zu inputs=%zu "
			"outputs=%zu\n",
			(unsigned)export_ordinal, (size_t)bindings.count,
			(size_t)num_inputs, (size_t)num_outputs);
	}

	if (num_inputs + num_outputs > bindings.count) {
		return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
			"QNN graph expects %zu inputs + %zu outputs (=%zu bindings) but "
			"dispatch supplied %zu",
			(size_t)num_inputs, (size_t)num_outputs,
			(size_t)(num_inputs + num_outputs), (size_t)bindings.count);
	}

	// Per-buffer fence: any pending async dispatch whose output buffer
	// matches one of THIS dispatch's input buffers must complete before
	// we can safely read from that buffer. This is what makes async +
	// chained dispatches correct: pendings without dep stay in flight,
	// pendings with dep block this dispatch but not unrelated ones.
	IREE_RETURN_IF_ERROR(
		iree_hal_qnn_fence_dependencies(command_buffer, num_inputs, bindings));

	// Patch the executable's cached prototype Qnn_Tensor_t arrays IN PLACE
	// with the per-call clientBuf pointers. Safe because:
	//   1. The device's queue_mutex serialises all dispatches on this device
	//      (graphExecuteAsync + the deferred CB replay both run under it),
	//   2. The fields we mutate (memType + clientBuf) are clientBuf only —
	//      name, dataType, rank, dimensions remain untouched and shared.
	// Eliminates ~200B/binding stack-copy per dispatch (no perf win on
	// current 6 ms kernels; pure code-size cleanup).
	Qnn_Tensor_t *qnn_inputs = (Qnn_Tensor_t *)proto_inputs;
	Qnn_Tensor_t *qnn_outputs = (Qnn_Tensor_t *)proto_outputs;

	// Allocate a pending-dispatch slot. If the pool is full, fence the
	// chain so far and reset, so we can keep recording. This bounds the
	// peak host-mapping count without forcing the user to pre-size.
	if (command_buffer->pending_count >= IREE_HAL_QNN_MAX_PENDING) {
		iree_status_t fence_status =
			iree_hal_qnn_command_buffer_fence_pending(command_buffer);
		if (!iree_status_is_ok(fence_status))
			return fence_status;
	}
	iree_hal_qnn_pending_t *pending =
		&command_buffer->pending[command_buffer->pending_count];
	pending->mapping_count = 0;
	pending->output_buffer_count = 0;
	pending->used_async = false;
	pending->fenced = false;

	iree_status_t status = iree_ok_status();

	for (iree_host_size_t i = 0;
		 i < num_inputs + num_outputs && iree_status_is_ok(status); ++i) {
		const bool is_output = i >= num_inputs;
		Qnn_Tensor_t *dst =
			is_output ? &qnn_outputs[i - num_inputs] : &qnn_inputs[i];

		const iree_hal_buffer_ref_t *ref = &bindings.values[i];
		if (ref->buffer == NULL) {
			status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
				"QNN dispatch binding %zu has NULL buffer", (size_t)i);
			break;
		}

		uint32_t qnn_tensor_bytes = 0;
		bool has_tensor_bytes =
			iree_hal_qnn_tensor_byte_size(dst, &qnn_tensor_bytes);
		if (has_tensor_bytes && ref->length < qnn_tensor_bytes) {
			status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
				"QNN dispatch binding %zu length %" PRIhsz
				" is smaller than tensor byte size %u",
				(size_t)i, ref->length, qnn_tensor_bytes);
			break;
		}

		status = iree_hal_buffer_map_range(ref->buffer,
			IREE_HAL_MAPPING_MODE_PERSISTENT,
			is_output ? IREE_HAL_MEMORY_ACCESS_WRITE
					  : IREE_HAL_MEMORY_ACCESS_READ,
			ref->offset, ref->length,
			&pending->mappings[pending->mapping_count]);
		if (!iree_status_is_ok(status))
			break;
		iree_hal_buffer_mapping_t *m =
			&pending->mappings[pending->mapping_count++];

		// Track output buffers for the per-buffer fence: future dispatches
		// that read this same buffer will wait on this pending's completion.
		if (is_output &&
			pending->output_buffer_count < IREE_HAL_QNN_MAX_DISPATCH_BINDINGS) {
			pending->output_buffers[pending->output_buffer_count++] =
				ref->buffer;
		}

		Qnn_TensorV1_t *v1 = (Qnn_TensorV1_t *)&dst->v1;
		v1->memType = QNN_TENSORMEMTYPE_RAW;
		v1->clientBuf.data = m->contents.data;
		// IREE dispatch bindings often refer to a larger backing buffer than
		// the QNN graph tensor (for example a dispatch.tensor.load slice from a
		// model-wide activation arena). QNN validates clientBuf.dataSize
		// against the tensor shape, not the backing allocation size.
		v1->clientBuf.dataSize = has_tensor_bytes
			? qnn_tensor_bytes
			: (uint32_t)m->contents.data_length;
		if (iree_hal_qnn_trace_enabled()) {
			fprintf(stderr,
				"[qnn-cmd] binding %zu %s name=%s dtype=%d rank=%u "
				"tensor_bytes=%u ref_length=%zu data=%p\n",
				(size_t)i, is_output ? "output" : "input",
				v1->name ? v1->name : "<null>", (int)v1->dataType, v1->rank,
				has_tensor_bytes ? qnn_tensor_bytes : 0u, (size_t)ref->length,
				v1->clientBuf.data);
		}
	}

	if (iree_status_is_ok(status)) {
		const QnnInterface_t *iface = command_buffer->qnn_interface;
		Qnn_ErrorHandle_t qnn_rc = QNN_SUCCESS;

		// Sync graphExecute. We have full async + per-buffer-fence
		// machinery ready (see fence_dependencies, fence_buffer_writers
		// above), but on QAIRT 2.45 GPU/HTA the graphExecuteAsync path adds
		// ~0.9 ms per call vs the sync path — likely the notify_fn
		// pthread_cond signal traversing the QNN backend worker thread
		// boundary. For our typical workloads (data-dependent chains, where
		// per-buffer fence would force serialization anyway) the async path
		// has no upside and a real downside on small/medium dispatches.
		//
		// The async + fence infrastructure stays in place because it's
		// correct and cheap, ready to flip back when:
		//  - we have non-data-dependent independent dispatches in one CB
		//    (rare today; IREE generally serializes data deps)
		//  - a future QAIRT release shrinks the async submit overhead
		//  - we want to overlap dispatches on different graphs (e.g.,
		//    qnn://gpu and qnn://hta running concurrently — would need
		//    cross-device fencing layered on top)
		if (iree_hal_qnn_trace_enabled()) {
			fprintf(stderr, "[qnn-cmd] before graphExecute\n");
		}
		qnn_rc = iface->QNN_INTERFACE_VER_NAME.graphExecute(
			(Qnn_GraphHandle_t)graph_handle, qnn_inputs, (uint32_t)num_inputs,
			qnn_outputs, (uint32_t)num_outputs, /*profileHandle=*/NULL,
			/*signalHandle=*/NULL);
		if (iree_hal_qnn_trace_enabled()) {
			fprintf(stderr, "[qnn-cmd] after graphExecute rc=%lld\n",
				(long long)qnn_rc);
		}

		if (qnn_rc != QNN_SUCCESS) {
			status = iree_make_status(IREE_STATUS_INTERNAL,
				"QnnGraph_execute failed rc=%lld for ordinal=%u "
				"(num_inputs=%zu num_outputs=%zu)",
				(long long)qnn_rc, (unsigned)export_ordinal, (size_t)num_inputs,
				(size_t)num_outputs);
		}
	}

	if (!iree_status_is_ok(status)) {
		// Failure path: roll back the pending entry's mappings synchronously
		// and don't enqueue. The async submission, if it happened, has
		// already populated completion via notify_fn; wait for it before
		// releasing mappings to avoid a use-after-unmap on the QNN side.
		if (pending->used_async) {
			iree_hal_qnn_async_completion_wait(&pending->completion);
			pending->used_async = false;
		}
		for (iree_host_size_t i = 0; i < pending->mapping_count; ++i) {
			iree_status_t unmap_status =
				iree_hal_buffer_unmap_range(&pending->mappings[i]);
			iree_status_ignore(unmap_status);
		}
		pending->mapping_count = 0;
		return status;
	}

	// Success: commit the pending record. Mappings stay alive; the
	// end-of-CB fence walks the queue and unmaps after waiting on each
	// completion. For sync-fallback dispatches, the wait is a no-op
	// (used_async=false) and the unmap happens at end as well, keeping
	// the cleanup path uniform.
	++command_buffer->pending_count;
	return iree_ok_status();
}

static const iree_hal_command_buffer_vtable_t
	iree_hal_qnn_command_buffer_vtable = {
		.destroy = iree_hal_qnn_command_buffer_destroy,
		.begin = iree_hal_qnn_command_buffer_begin,
		.end = iree_hal_qnn_command_buffer_end,
		.begin_debug_group = iree_hal_qnn_command_buffer_begin_debug_group,
		.end_debug_group = iree_hal_qnn_command_buffer_end_debug_group,
		.execution_barrier = iree_hal_qnn_command_buffer_execution_barrier,
		.signal_event = iree_hal_qnn_command_buffer_signal_event,
		.reset_event = iree_hal_qnn_command_buffer_reset_event,
		.wait_events = iree_hal_qnn_command_buffer_wait_events,
		.advise_buffer = iree_hal_qnn_command_buffer_advise_buffer,
		.fill_buffer = iree_hal_qnn_command_buffer_fill_buffer,
		.update_buffer = iree_hal_qnn_command_buffer_update_buffer,
		.copy_buffer = iree_hal_qnn_command_buffer_copy_buffer,
		.collective = iree_hal_qnn_command_buffer_collective,
		.dispatch = iree_hal_qnn_command_buffer_dispatch,
};
