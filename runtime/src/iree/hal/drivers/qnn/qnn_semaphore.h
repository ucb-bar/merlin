// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_QNN_QNN_SEMAPHORE_H_
#define IREE_HAL_DRIVERS_QNN_QNN_SEMAPHORE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef struct iree_async_proactor_t iree_async_proactor_t;

// Host-side timeline semaphore for the QNN HAL driver, mirroring the
// task_semaphore.c pattern (current canonical IREE async-semaphore API).
// QNN's QnnGraph_execute is synchronous, so semaphores only carry timeline
// ordering between submitted command buffers; signals dispatch timepoints
// directly when the executor advances the timeline.
iree_status_t iree_hal_qnn_semaphore_create(iree_async_proactor_t *proactor,
	uint64_t initial_value, iree_allocator_t host_allocator,
	iree_hal_semaphore_t **out_semaphore);

bool iree_hal_qnn_semaphore_isa(iree_hal_semaphore_t *semaphore);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // IREE_HAL_DRIVERS_QNN_QNN_SEMAPHORE_H_
