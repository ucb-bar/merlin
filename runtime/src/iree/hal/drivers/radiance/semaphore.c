// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "semaphore.h"

#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>

#include "iree/base/internal/synchronization.h"
#include "iree/hal/utils/semaphore_base.h"

typedef struct iree_hal_radiance_semaphore_t {
	iree_hal_semaphore_t base;
	iree_allocator_t host_allocator;
	iree_hal_queue_affinity_t queue_affinity;
	iree_slim_mutex_t mutex;
	iree_notification_t notification;
	uint64_t current_value;
	iree_status_t failure_status;
} iree_hal_radiance_semaphore_t;

static const iree_hal_semaphore_vtable_t iree_hal_radiance_semaphore_vtable;

static iree_hal_radiance_semaphore_t *iree_hal_radiance_semaphore_cast(
	iree_hal_semaphore_t *base_value) {
	IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_radiance_semaphore_vtable);
	return (iree_hal_radiance_semaphore_t *)base_value;
}

iree_status_t iree_hal_radiance_semaphore_create(
	iree_hal_queue_affinity_t queue_affinity, uint64_t initial_value,
	iree_hal_semaphore_flags_t flags, iree_allocator_t host_allocator,
	iree_hal_semaphore_t **out_semaphore) {
	(void)flags;
	IREE_ASSERT_ARGUMENT(out_semaphore);
	IREE_TRACE_ZONE_BEGIN(z0);
	*out_semaphore = NULL;

	iree_hal_radiance_semaphore_t *semaphore = NULL;
	IREE_RETURN_AND_END_ZONE_IF_ERROR(z0,
		iree_allocator_malloc(
			host_allocator, sizeof(*semaphore), (void **)&semaphore));
	iree_hal_semaphore_initialize(
		&iree_hal_radiance_semaphore_vtable, &semaphore->base);
	semaphore->host_allocator = host_allocator;
	semaphore->queue_affinity = queue_affinity;
	iree_slim_mutex_initialize(&semaphore->mutex);
	iree_notification_initialize(&semaphore->notification);
	semaphore->current_value = initial_value;
	semaphore->failure_status = iree_ok_status();

	*out_semaphore = &semaphore->base;
	IREE_TRACE_ZONE_END(z0);
	return iree_ok_status();
}

static void iree_hal_radiance_semaphore_destroy(
	iree_hal_semaphore_t *base_semaphore) {
	iree_hal_radiance_semaphore_t *semaphore =
		iree_hal_radiance_semaphore_cast(base_semaphore);
	IREE_TRACE_ZONE_BEGIN(z0);

	iree_notification_deinitialize(&semaphore->notification);
	iree_slim_mutex_deinitialize(&semaphore->mutex);
	iree_status_ignore(semaphore->failure_status);

	iree_hal_semaphore_deinitialize(&semaphore->base);
	iree_allocator_free(semaphore->host_allocator, semaphore);

	IREE_TRACE_ZONE_END(z0);
}

bool iree_hal_radiance_semaphore_isa(iree_hal_semaphore_t *semaphore) {
	return iree_hal_resource_is(
		&semaphore->resource, &iree_hal_radiance_semaphore_vtable);
}

static iree_status_t iree_hal_radiance_semaphore_query(
	iree_hal_semaphore_t *base_semaphore, uint64_t *out_value) {
	iree_hal_radiance_semaphore_t *semaphore =
		iree_hal_radiance_semaphore_cast(base_semaphore);
	iree_slim_mutex_lock(&semaphore->mutex);
	*out_value = semaphore->current_value;
	iree_status_t status = iree_ok_status();
	if (*out_value >= IREE_HAL_SEMAPHORE_FAILURE_VALUE) {
		status = iree_status_clone(semaphore->failure_status);
	}
	iree_slim_mutex_unlock(&semaphore->mutex);
	return status;
}

static iree_status_t iree_hal_radiance_semaphore_signal(
	iree_hal_semaphore_t *base_semaphore, uint64_t new_value) {
	iree_hal_radiance_semaphore_t *semaphore =
		iree_hal_radiance_semaphore_cast(base_semaphore);
	iree_slim_mutex_lock(&semaphore->mutex);
	if (new_value <= semaphore->current_value) {
		const uint64_t current_value = semaphore->current_value;
		iree_slim_mutex_unlock(&semaphore->mutex);
		return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
			"semaphore values must be monotonically "
			"increasing; current_value=%" PRIu64 ", new_value=%" PRIu64,
			current_value, new_value);
	}
	semaphore->current_value = new_value;
	iree_slim_mutex_unlock(&semaphore->mutex);

	iree_hal_semaphore_notify(&semaphore->base, new_value, IREE_STATUS_OK);
	iree_notification_post(&semaphore->notification, IREE_ALL_WAITERS);
	return iree_ok_status();
}

static void iree_hal_radiance_semaphore_fail(
	iree_hal_semaphore_t *base_semaphore, iree_status_t status) {
	iree_hal_radiance_semaphore_t *semaphore =
		iree_hal_radiance_semaphore_cast(base_semaphore);
	const iree_status_code_t status_code = iree_status_code(status);

	iree_slim_mutex_lock(&semaphore->mutex);
	if (!iree_status_is_ok(semaphore->failure_status)) {
		IREE_IGNORE_ERROR(status);
		iree_slim_mutex_unlock(&semaphore->mutex);
		return;
	}
	semaphore->current_value = IREE_HAL_SEMAPHORE_FAILURE_VALUE;
	semaphore->failure_status = status;
	iree_slim_mutex_unlock(&semaphore->mutex);

	iree_hal_semaphore_notify(
		&semaphore->base, IREE_HAL_SEMAPHORE_FAILURE_VALUE, status_code);
	iree_notification_post(&semaphore->notification, IREE_ALL_WAITERS);
}

typedef struct iree_hal_radiance_semaphore_wait_state_t {
	iree_hal_radiance_semaphore_t *semaphore;
	uint64_t value;
} iree_hal_radiance_semaphore_wait_state_t;

static bool iree_hal_radiance_semaphore_has_reached(
	iree_hal_radiance_semaphore_wait_state_t *wait_state) {
	iree_hal_radiance_semaphore_t *semaphore = wait_state->semaphore;
	iree_slim_mutex_lock(&semaphore->mutex);
	const bool reached = semaphore->current_value >= wait_state->value ||
		!iree_status_is_ok(semaphore->failure_status);
	iree_slim_mutex_unlock(&semaphore->mutex);
	return reached;
}

static iree_status_t iree_hal_radiance_semaphore_wait(
	iree_hal_semaphore_t *base_semaphore, uint64_t value,
	iree_timeout_t timeout, iree_hal_wait_flags_t flags) {
	(void)flags;
	iree_hal_radiance_semaphore_t *semaphore =
		iree_hal_radiance_semaphore_cast(base_semaphore);
	iree_slim_mutex_lock(&semaphore->mutex);
	if (!iree_status_is_ok(semaphore->failure_status)) {
		iree_slim_mutex_unlock(&semaphore->mutex);
		return iree_status_from_code(IREE_STATUS_ABORTED);
	}
	if (semaphore->current_value >= value) {
		iree_slim_mutex_unlock(&semaphore->mutex);
		return iree_ok_status();
	}
	if (iree_timeout_is_immediate(timeout)) {
		iree_slim_mutex_unlock(&semaphore->mutex);
		return iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
	}
	iree_slim_mutex_unlock(&semaphore->mutex);

	iree_hal_radiance_semaphore_wait_state_t wait_state = {
		.semaphore = semaphore,
		.value = value,
	};
	iree_notification_await(&semaphore->notification,
		(iree_condition_fn_t)iree_hal_radiance_semaphore_has_reached,
		&wait_state, timeout);

	iree_slim_mutex_lock(&semaphore->mutex);
	iree_status_t status = iree_ok_status();
	if (!iree_status_is_ok(semaphore->failure_status)) {
		status = iree_status_from_code(IREE_STATUS_ABORTED);
	} else if (semaphore->current_value < value) {
		status = iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
	}
	iree_slim_mutex_unlock(&semaphore->mutex);
	return status;
}

static iree_status_t iree_hal_radiance_semaphore_import_timepoint(
	iree_hal_semaphore_t *base_semaphore, uint64_t value,
	iree_hal_queue_affinity_t queue_affinity,
	iree_hal_external_timepoint_t external_timepoint) {
	(void)base_semaphore;
	(void)value;
	(void)queue_affinity;
	(void)external_timepoint;
	return iree_make_status(
		IREE_STATUS_UNIMPLEMENTED, "timepoint import is not yet implemented");
}

static iree_status_t iree_hal_radiance_semaphore_export_timepoint(
	iree_hal_semaphore_t *base_semaphore, uint64_t value,
	iree_hal_queue_affinity_t queue_affinity,
	iree_hal_external_timepoint_type_t requested_type,
	iree_hal_external_timepoint_flags_t requested_flags,
	iree_hal_external_timepoint_t *IREE_RESTRICT out_external_timepoint) {
	(void)base_semaphore;
	(void)value;
	(void)queue_affinity;
	(void)requested_type;
	(void)requested_flags;
	(void)out_external_timepoint;
	return iree_make_status(
		IREE_STATUS_UNIMPLEMENTED, "timepoint export is not yet implemented");
}

static const iree_hal_semaphore_vtable_t iree_hal_radiance_semaphore_vtable = {
	.destroy = iree_hal_radiance_semaphore_destroy,
	.query = iree_hal_radiance_semaphore_query,
	.signal = iree_hal_radiance_semaphore_signal,
	.fail = iree_hal_radiance_semaphore_fail,
	.wait = iree_hal_radiance_semaphore_wait,
	.import_timepoint = iree_hal_radiance_semaphore_import_timepoint,
	.export_timepoint = iree_hal_radiance_semaphore_export_timepoint,
};
