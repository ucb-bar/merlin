// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "pending_queue_actions.h"

#include <string.h>

iree_status_t iree_hal_cuda_new_completion_create(
	const iree_hal_cuda_new_dynamic_symbols_t *syms,
	const iree_hal_semaphore_list_t signal_semaphore_list,
	iree_hal_command_buffer_t *retained_command_buffer,
	CUgraphExec retained_graph_exec, iree_allocator_t host_allocator,
	iree_hal_cuda_new_completion_t **out_completion) {
	*out_completion = NULL;

	iree_host_size_t total_size = sizeof(iree_hal_cuda_new_completion_t) +
		signal_semaphore_list.count * sizeof(iree_hal_semaphore_t *) +
		signal_semaphore_list.count * sizeof(uint64_t);

	iree_hal_cuda_new_completion_t *completion = NULL;
	IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, total_size,
		(void **)&completion));

	completion->host_allocator = host_allocator;
	completion->syms = syms;
	completion->signal_count = signal_semaphore_list.count;
	completion->signal_semaphores =
		(iree_hal_semaphore_t **)((uint8_t *)completion +
								  sizeof(*completion));
	completion->signal_values =
		(uint64_t *)((uint8_t *)completion->signal_semaphores +
					 signal_semaphore_list.count *
						 sizeof(iree_hal_semaphore_t *));

	for (iree_host_size_t i = 0; i < signal_semaphore_list.count; ++i) {
		completion->signal_semaphores[i] =
			signal_semaphore_list.semaphores[i];
		iree_hal_semaphore_retain(signal_semaphore_list.semaphores[i]);
		completion->signal_values[i] =
			signal_semaphore_list.payload_values[i];
	}

	completion->retained_command_buffer = retained_command_buffer;
	if (retained_command_buffer) {
		iree_hal_command_buffer_retain(retained_command_buffer);
	}
	completion->retained_graph_exec = retained_graph_exec;

	*out_completion = completion;
	return iree_ok_status();
}

static void iree_hal_cuda_new_completion_destroy(
	iree_hal_cuda_new_completion_t *completion) {
	for (iree_host_size_t i = 0; i < completion->signal_count; ++i) {
		iree_hal_semaphore_release(completion->signal_semaphores[i]);
	}
	if (completion->retained_command_buffer) {
		iree_hal_command_buffer_release(
			completion->retained_command_buffer);
	}
	if (completion->retained_graph_exec) {
		completion->syms->cuGraphExecDestroy(
			completion->retained_graph_exec);
	}
	iree_allocator_free(completion->host_allocator, completion);
}

void iree_hal_cuda_new_completion_abort(
	iree_hal_cuda_new_completion_t *completion) {
	iree_hal_cuda_new_completion_destroy(completion);
}

void CUDA_CB iree_hal_cuda_new_completion_host_callback(void *user_data) {
	iree_hal_cuda_new_completion_t *completion =
		(iree_hal_cuda_new_completion_t *)user_data;

	for (iree_host_size_t i = 0; i < completion->signal_count; ++i) {
		iree_hal_semaphore_signal(completion->signal_semaphores[i],
			completion->signal_values[i], /*frontier=*/NULL);
	}

	iree_hal_cuda_new_completion_destroy(completion);
}
