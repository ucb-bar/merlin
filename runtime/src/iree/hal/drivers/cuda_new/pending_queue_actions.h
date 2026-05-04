// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_CUDA_NEW_PENDING_QUEUE_ACTIONS_H_
#define IREE_HAL_DRIVERS_CUDA_NEW_PENDING_QUEUE_ACTIONS_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "dynamic_symbols.h"
#include "headers.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// A completion payload enqueued on the CUDA stream via cuLaunchHostFunc.
// When the stream reaches this point, the host callback fires, signals
// semaphores, and releases all retained resources.
typedef struct iree_hal_cuda_new_completion_t {
	iree_allocator_t host_allocator;

	iree_host_size_t signal_count;
	iree_hal_semaphore_t **signal_semaphores;
	uint64_t *signal_values;

	iree_hal_command_buffer_t *retained_command_buffer;
	CUgraphExec retained_graph_exec;
	const iree_hal_cuda_new_dynamic_symbols_t *syms;
} iree_hal_cuda_new_completion_t;

// Allocates a completion payload and copies the signal semaphore list.
iree_status_t iree_hal_cuda_new_completion_create(
	const iree_hal_cuda_new_dynamic_symbols_t *syms,
	const iree_hal_semaphore_list_t signal_semaphore_list,
	iree_hal_command_buffer_t *retained_command_buffer,
	CUgraphExec retained_graph_exec, iree_allocator_t host_allocator,
	iree_hal_cuda_new_completion_t **out_completion);

// The CUhostFn callback signature. Enqueue via cuLaunchHostFunc.
void CUDA_CB iree_hal_cuda_new_completion_host_callback(void *user_data);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // IREE_HAL_DRIVERS_CUDA_NEW_PENDING_QUEUE_ACTIONS_H_
