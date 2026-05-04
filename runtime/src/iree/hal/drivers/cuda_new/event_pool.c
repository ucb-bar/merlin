// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "event_pool.h"

#include <stddef.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"
#include "iree/base/threading/mutex.h"
#include "status_util.h"

struct iree_hal_cuda_new_event_t {
	// 1 while pooled/owned, >1 while externally retained, 0 while destroying.
	iree_atomic_ref_count_t ref_count;
	iree_allocator_t host_allocator;
	const iree_hal_cuda_new_dynamic_symbols_t *syms;
	iree_hal_cuda_new_event_pool_t *pool;
	CUevent cu_event;
};

struct iree_hal_cuda_new_event_pool_t {
	iree_atomic_ref_count_t ref_count;
	iree_allocator_t host_allocator;
	const iree_hal_cuda_new_dynamic_symbols_t *syms;

	iree_slim_mutex_t event_mutex;
	iree_host_size_t available_capacity IREE_GUARDED_BY(event_mutex);
	iree_host_size_t available_count IREE_GUARDED_BY(event_mutex);
	iree_hal_cuda_new_event_t *available_list[] IREE_GUARDED_BY(event_mutex);
};

CUevent iree_hal_cuda_new_event_handle(
	const iree_hal_cuda_new_event_t *event) {
	return event->cu_event;
}

static void iree_hal_cuda_new_event_destroy(
	iree_hal_cuda_new_event_t *event) {
	iree_allocator_t host_allocator = event->host_allocator;
	const iree_hal_cuda_new_dynamic_symbols_t *syms = event->syms;
	IREE_TRACE_ZONE_BEGIN(z0);

	IREE_ASSERT_REF_COUNT_ZERO(&event->ref_count);
	IREE_CUDA_NEW_IGNORE_ERROR(syms, cuEventDestroy(event->cu_event));
	iree_allocator_free(host_allocator, event);

	IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_cuda_new_event_create(
	const iree_hal_cuda_new_dynamic_symbols_t *syms,
	iree_hal_cuda_new_event_pool_t *pool, iree_allocator_t host_allocator,
	iree_hal_cuda_new_event_t **out_event) {
	IREE_ASSERT_ARGUMENT(syms);
	IREE_ASSERT_ARGUMENT(pool);
	IREE_ASSERT_ARGUMENT(out_event);
	*out_event = NULL;
	IREE_TRACE_ZONE_BEGIN(z0);

	iree_hal_cuda_new_event_t *event = NULL;
	IREE_RETURN_AND_END_ZONE_IF_ERROR(z0,
		iree_allocator_malloc(host_allocator, sizeof(*event),
			(void **)&event));
	iree_atomic_ref_count_init(&event->ref_count);
	event->host_allocator = host_allocator;
	event->syms = syms;
	event->pool = pool;
	event->cu_event = NULL;

	iree_status_t status = IREE_CURESULT_TO_STATUS_NEW(syms,
		cuEventCreate(&event->cu_event, CU_EVENT_DISABLE_TIMING),
		"cuEventCreate");
	if (iree_status_is_ok(status)) {
		*out_event = event;
	} else {
		iree_atomic_ref_count_dec(&event->ref_count);
		iree_hal_cuda_new_event_destroy(event);
	}

	IREE_TRACE_ZONE_END(z0);
	return status;
}

void iree_hal_cuda_new_event_retain(iree_hal_cuda_new_event_t *event) {
	iree_atomic_ref_count_inc(&event->ref_count);
}

static void iree_hal_cuda_new_event_pool_release_events(
	iree_hal_cuda_new_event_pool_t *event_pool,
	iree_host_size_t event_count,
	iree_hal_cuda_new_event_t **events);

void iree_hal_cuda_new_event_release(iree_hal_cuda_new_event_t *event) {
	if (iree_atomic_ref_count_dec(&event->ref_count) == 1) {
		iree_hal_cuda_new_event_pool_t *pool = event->pool;
		iree_hal_cuda_new_event_pool_release_events(pool, 1, &event);
		iree_hal_cuda_new_event_pool_release(pool);
	}
}

static void iree_hal_cuda_new_event_pool_free(
	iree_hal_cuda_new_event_pool_t *event_pool) {
	iree_allocator_t host_allocator = event_pool->host_allocator;
	IREE_TRACE_ZONE_BEGIN(z0);

	for (iree_host_size_t i = 0; i < event_pool->available_count; ++i) {
		iree_hal_cuda_new_event_t *event = event_pool->available_list[i];
		iree_atomic_ref_count_dec(&event->ref_count);
		iree_hal_cuda_new_event_destroy(event);
	}
	IREE_ASSERT_REF_COUNT_ZERO(&event_pool->ref_count);

	iree_slim_mutex_deinitialize(&event_pool->event_mutex);
	iree_allocator_free(host_allocator, event_pool);

	IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_cuda_new_event_pool_allocate(
	const iree_hal_cuda_new_dynamic_symbols_t *syms,
	iree_host_size_t available_capacity, iree_allocator_t host_allocator,
	iree_hal_cuda_new_event_pool_t **out_event_pool) {
	IREE_ASSERT_ARGUMENT(syms);
	IREE_ASSERT_ARGUMENT(out_event_pool);
	*out_event_pool = NULL;
	IREE_TRACE_ZONE_BEGIN(z0);

	iree_hal_cuda_new_event_pool_t *event_pool = NULL;
	const iree_host_size_t total_size =
		sizeof(*event_pool) +
		available_capacity * sizeof(*event_pool->available_list);
	IREE_RETURN_AND_END_ZONE_IF_ERROR(z0,
		iree_allocator_malloc(host_allocator, total_size,
			(void **)&event_pool));
	iree_atomic_ref_count_init(&event_pool->ref_count);
	event_pool->host_allocator = host_allocator;
	event_pool->syms = syms;
	iree_slim_mutex_initialize(&event_pool->event_mutex);
	event_pool->available_capacity = available_capacity;
	event_pool->available_count = 0;

	iree_status_t status = iree_ok_status();
	for (iree_host_size_t i = 0; i < available_capacity; ++i) {
		iree_hal_cuda_new_event_t *event = NULL;
		status = iree_hal_cuda_new_event_create(syms, event_pool,
			host_allocator, &event);
		if (!iree_status_is_ok(status)) break;
		event_pool->available_list[event_pool->available_count++] = event;
	}

	if (iree_status_is_ok(status)) {
		*out_event_pool = event_pool;
	} else {
		iree_hal_cuda_new_event_pool_free(event_pool);
	}
	IREE_TRACE_ZONE_END(z0);
	return status;
}

void iree_hal_cuda_new_event_pool_retain(
	iree_hal_cuda_new_event_pool_t *event_pool) {
	iree_atomic_ref_count_inc(&event_pool->ref_count);
}

void iree_hal_cuda_new_event_pool_release(
	iree_hal_cuda_new_event_pool_t *event_pool) {
	if (iree_atomic_ref_count_dec(&event_pool->ref_count) == 1) {
		iree_hal_cuda_new_event_pool_free(event_pool);
	}
}

iree_status_t iree_hal_cuda_new_event_pool_acquire(
	iree_hal_cuda_new_event_pool_t *event_pool,
	iree_host_size_t event_count,
	iree_hal_cuda_new_event_t **out_events) {
	IREE_ASSERT_ARGUMENT(event_pool);
	if (!event_count) return iree_ok_status();
	IREE_ASSERT_ARGUMENT(out_events);
	IREE_TRACE_ZONE_BEGIN(z0);

	iree_host_size_t remaining_count = event_count;

	iree_slim_mutex_lock(&event_pool->event_mutex);
	const iree_host_size_t from_pool_count =
		iree_min(event_pool->available_count, event_count);
	if (from_pool_count > 0) {
		const iree_host_size_t pool_base_index =
			event_pool->available_count - from_pool_count;
		memcpy(out_events, &event_pool->available_list[pool_base_index],
			from_pool_count * sizeof(*event_pool->available_list));
		event_pool->available_count -= from_pool_count;
		remaining_count -= from_pool_count;
	}
	iree_slim_mutex_unlock(&event_pool->event_mutex);

	if (remaining_count > 0) {
		IREE_TRACE_ZONE_BEGIN_NAMED(z1, "event-pool-unpooled-acquire");
		iree_status_t status = iree_ok_status();
		for (iree_host_size_t i = 0; i < remaining_count; ++i) {
			status = iree_hal_cuda_new_event_create(event_pool->syms,
				event_pool, event_pool->host_allocator,
				&out_events[from_pool_count + i]);
			if (!iree_status_is_ok(status)) {
				iree_hal_cuda_new_event_pool_release_events(event_pool,
					from_pool_count + i, out_events);
				IREE_TRACE_ZONE_END(z1);
				IREE_TRACE_ZONE_END(z0);
				return status;
			}
		}
		IREE_TRACE_ZONE_END(z1);
	}

	for (iree_host_size_t i = 0; i < event_count; ++i) {
		iree_hal_cuda_new_event_pool_retain(out_events[i]->pool);
	}

	IREE_TRACE_ZONE_END(z0);
	return iree_ok_status();
}

static void iree_hal_cuda_new_event_pool_release_events(
	iree_hal_cuda_new_event_pool_t *event_pool,
	iree_host_size_t event_count,
	iree_hal_cuda_new_event_t **events) {
	IREE_ASSERT_ARGUMENT(event_pool);
	if (!event_count) return;
	IREE_ASSERT_ARGUMENT(events);
	IREE_TRACE_ZONE_BEGIN(z0);

	iree_host_size_t remaining_count = event_count;

	iree_slim_mutex_lock(&event_pool->event_mutex);
	const iree_host_size_t to_pool_count = iree_min(
		event_pool->available_capacity - event_pool->available_count,
		event_count);
	if (to_pool_count > 0) {
		for (iree_host_size_t i = 0; i < to_pool_count; ++i) {
			IREE_ASSERT_REF_COUNT_ZERO(&events[i]->ref_count);
			iree_hal_cuda_new_event_retain(events[i]);
		}
		const iree_host_size_t pool_base_index = event_pool->available_count;
		memcpy(&event_pool->available_list[pool_base_index], events,
			to_pool_count * sizeof(*event_pool->available_list));
		event_pool->available_count += to_pool_count;
		remaining_count -= to_pool_count;
	}
	iree_slim_mutex_unlock(&event_pool->event_mutex);

	if (remaining_count > 0) {
		IREE_TRACE_ZONE_BEGIN_NAMED(z1, "event-pool-unpooled-release");
		for (iree_host_size_t i = 0; i < remaining_count; ++i) {
			iree_hal_cuda_new_event_destroy(events[to_pool_count + i]);
		}
		IREE_TRACE_ZONE_END(z1);
	}

	IREE_TRACE_ZONE_END(z0);
}
