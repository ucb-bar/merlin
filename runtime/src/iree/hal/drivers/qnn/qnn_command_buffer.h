// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_QNN_QNN_COMMAND_BUFFER_H_
#define IREE_HAL_DRIVERS_QNN_QNN_COMMAND_BUFFER_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// Creates a target command buffer that the queue_execute path calls
// iree_hal_deferred_command_buffer_apply against. Most vtable methods are
// no-ops; only `dispatch` does real work — it translates the IREE HAL
// dispatch into a QnnGraph_execute call against the executable's resolved
// Qnn_GraphHandle_t.
//
// |qnn_interface| is the const QnnInterface_t* from the device's loaded
// libQnn<Backend>.so, passed as void* so this header stays QNN-SDK-free.
// Borrowed from the device — must outlive the buffer.
iree_status_t iree_hal_qnn_command_buffer_create(
	iree_hal_allocator_t *device_allocator, iree_hal_command_buffer_mode_t mode,
	iree_hal_command_category_t command_categories,
	iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
	const void *qnn_interface, iree_allocator_t host_allocator,
	iree_hal_command_buffer_t **out_command_buffer);

bool iree_hal_qnn_command_buffer_isa(iree_hal_command_buffer_t *command_buffer);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // IREE_HAL_DRIVERS_QNN_QNN_COMMAND_BUFFER_H_
