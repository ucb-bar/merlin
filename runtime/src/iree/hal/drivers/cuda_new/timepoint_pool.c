// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "timepoint_pool.h"

#include <stddef.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/threading/mutex.h"

struct iree_hal_cuda_new_timepoint_pool_t {
	iree_allocator_t host_allocator;
	iree_hal_cuda_new_event_pool_t *event_pool;

	iree_slim_mutex_t mutex;
	iree_host_size_t available_capacity;
	iree_host_size_t available_count;
	iree_hal_cuda_new_timepoint_t *available_list[];
};

static iree_status_t iree_hal_cuda_new_timepoint_create(
	iree_hal_cuda_new_timepoint_pool_t *pool,
	iree_allocator_t host_allocator,
	iree_hal_cuda_new_timepoint_t **out_timepoint) {
	*out_timepoint = NULL;
	iree_hal_cuda_new_timepoint_t *timepoint = NULL;
	IREE_RETURN_IF_ERROR(iree_allocator_malloc(
		host_allocator, sizeof(*timepoint), (void **)&timepoint));
	memset(timepoint, 0, sizeof(*timepoint));
	timepoint->host_allocator = host_allocator;
	timepoint->pool = pool;
	*out_timepoint = timepoint;
	return iree_ok_status();
}

static void iree_hal_cuda_new_timepoint_clear(
	iree_hal_cuda_new_timepoint_t *timepoint) {
	if (timepoint->event) {
		iree_hal_cuda_new_event_release(timepoint->event);
		timepoint->event = NULL;
	}
	timepoint->kind = IREE_HAL_CUDA_NEW_TIMEPOINT_KIND_NONE;
	memset(&timepoint->base, 0, sizeof(timepoint->base));
}

static void iree_hal_cuda_new_timepoint_destroy(
	iree_hal_cuda_new_timepoint_t *timepoint) {
	iree_hal_cuda_new_timepoint_clear(timepoint);
	iree_allocator_free(timepoint->host_allocator, timepoint);
}

iree_status_t iree_hal_cuda_new_timepoint_pool_allocate(
	iree_hal_cuda_new_event_pool_t *event_pool,
	iree_host_size_t available_capacity, iree_allocator_t host_allocator,
	iree_hal_cuda_new_timepoint_pool_t **out_timepoint_pool) {
	IREE_ASSERT_ARGUMENT(event_pool);
	IREE_ASSERT_ARGUMENT(out_timepoint_pool);
	*out_timepoint_pool = NULL;

	iree_hal_cuda_new_timepoint_pool_t *pool = NULL;
	const iree_host_size_t total_size =
		sizeof(*pool) + available_capacity * sizeof(*pool->available_list);
	IREE_RETURN_IF_ERROR(
		iree_allocator_malloc(host_allocator, total_size, (void **)&pool));
	pool->host_allocator = host_allocator;
	pool->event_pool = event_pool;
	iree_slim_mutex_initialize(&pool->mutex);
	pool->available_capacity = available_capacity;
	pool->available_count = 0;

	iree_status_t status = iree_ok_status();
	for (iree_host_size_t i = 0; i < available_capacity; ++i) {
		iree_hal_cuda_new_timepoint_t *tp = NULL;
		status = iree_hal_cuda_new_timepoint_create(pool, host_allocator,
			&tp);
		if (!iree_status_is_ok(status)) break;
		pool->available_list[pool->available_count++] = tp;
	}

	if (iree_status_is_ok(status)) {
		*out_timepoint_pool = pool;
	} else {
		iree_hal_cuda_new_timepoint_pool_free(pool);
	}
	return status;
}

void iree_hal_cuda_new_timepoint_pool_free(
	iree_hal_cuda_new_timepoint_pool_t *pool) {
	if (!pool) return;
	for (iree_host_size_t i = 0; i < pool->available_count; ++i) {
		iree_hal_cuda_new_timepoint_destroy(pool->available_list[i]);
	}
	iree_slim_mutex_deinitialize(&pool->mutex);
	iree_allocator_free(pool->host_allocator, pool);
}

iree_status_t iree_hal_cuda_new_timepoint_pool_acquire(
	iree_hal_cuda_new_timepoint_pool_t *pool,
	iree_hal_cuda_new_timepoint_kind_t kind,
	iree_hal_cuda_new_timepoint_t **out_timepoint) {
	IREE_ASSERT_ARGUMENT(pool);
	IREE_ASSERT_ARGUMENT(out_timepoint);
	*out_timepoint = NULL;

	iree_hal_cuda_new_timepoint_t *timepoint = NULL;
	iree_slim_mutex_lock(&pool->mutex);
	if (pool->available_count > 0) {
		timepoint = pool->available_list[--pool->available_count];
	}
	iree_slim_mutex_unlock(&pool->mutex);

	if (!timepoint) {
		IREE_RETURN_IF_ERROR(iree_hal_cuda_new_timepoint_create(
			pool, pool->host_allocator, &timepoint));
	}

	// Acquire a CUDA event for this timepoint.
	iree_hal_cuda_new_event_t *event = NULL;
	iree_status_t status =
		iree_hal_cuda_new_event_pool_acquire(pool->event_pool, 1, &event);
	if (!iree_status_is_ok(status)) {
		iree_hal_cuda_new_timepoint_pool_release(pool, timepoint);
		return status;
	}

	timepoint->kind = kind;
	timepoint->event = event;
	*out_timepoint = timepoint;
	return iree_ok_status();
}

void iree_hal_cuda_new_timepoint_pool_release(
	iree_hal_cuda_new_timepoint_pool_t *pool,
	iree_hal_cuda_new_timepoint_t *timepoint) {
	if (!timepoint) return;
	iree_hal_cuda_new_timepoint_clear(timepoint);

	iree_slim_mutex_lock(&pool->mutex);
	if (pool->available_count < pool->available_capacity) {
		pool->available_list[pool->available_count++] = timepoint;
		timepoint = NULL;
	}
	iree_slim_mutex_unlock(&pool->mutex);

	if (timepoint) {
		iree_hal_cuda_new_timepoint_destroy(timepoint);
	}
}
