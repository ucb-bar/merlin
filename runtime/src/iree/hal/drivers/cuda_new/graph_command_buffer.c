// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "graph_command_buffer.h"

#include "command_buffer.h"
#include "status_util.h"
#include "iree/hal/utils/deferred_command_buffer.h"

iree_status_t iree_hal_cuda_new_graph_command_buffer_create(
	iree_hal_allocator_t *device_allocator,
	const iree_hal_cuda_new_dynamic_symbols_t *syms,
	CUcontext cu_context, CUstream capture_stream,
	iree_arena_block_pool_t *block_pool, iree_allocator_t host_allocator,
	iree_hal_command_buffer_t *source_command_buffer,
	iree_hal_buffer_binding_table_t binding_table,
	CUgraphExec *out_graph_exec) {
	IREE_ASSERT_ARGUMENT(syms);
	IREE_ASSERT_ARGUMENT(source_command_buffer);
	IREE_ASSERT_ARGUMENT(out_graph_exec);
	*out_graph_exec = NULL;
	IREE_TRACE_ZONE_BEGIN(z0);

	// Set CUDA context.
	IREE_RETURN_AND_END_ZONE_IF_ERROR(z0,
		IREE_CURESULT_TO_STATUS_NEW(syms,
			cuCtxSetCurrent(cu_context), "cuCtxSetCurrent"));

	// Begin graph capture on the stream.
	IREE_RETURN_AND_END_ZONE_IF_ERROR(z0,
		IREE_CURESULT_TO_STATUS_NEW(syms,
			cuStreamBeginCapture(capture_stream,
				CU_STREAM_CAPTURE_MODE_GLOBAL),
			"cuStreamBeginCapture"));

	// Create a stream command buffer targeting the capture stream and replay.
	iree_hal_command_buffer_t *stream_cb = NULL;
	iree_status_t status = iree_hal_cuda_new_stream_command_buffer_create(
		device_allocator, syms, cu_context,
		IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
		IREE_HAL_COMMAND_CATEGORY_ANY, /*binding_capacity=*/0,
		capture_stream, block_pool, host_allocator, &stream_cb);

	if (iree_status_is_ok(status)) {
		status = iree_hal_deferred_command_buffer_apply(
			source_command_buffer, stream_cb, binding_table);
	}
	if (stream_cb) {
		iree_hal_command_buffer_release(stream_cb);
	}

	// End capture — get the graph.
	CUgraph graph = NULL;
	if (iree_status_is_ok(status)) {
		status = IREE_CURESULT_TO_STATUS_NEW(syms,
			cuStreamEndCapture(capture_stream, &graph),
			"cuStreamEndCapture");
	} else {
		// Must end capture even on failure to leave stream in valid state.
		CUgraph discard = NULL;
		IREE_CUDA_NEW_IGNORE_ERROR(syms,
			cuStreamEndCapture(capture_stream, &discard));
		if (discard) {
			IREE_CUDA_NEW_IGNORE_ERROR(syms, cuGraphDestroy(discard));
		}
		IREE_TRACE_ZONE_END(z0);
		return status;
	}

	// Instantiate the graph into an executable.
	CUgraphExec graph_exec = NULL;
	if (iree_status_is_ok(status)) {
		status = IREE_CURESULT_TO_STATUS_NEW(syms,
			cuGraphInstantiate(&graph_exec, graph, 0),
			"cuGraphInstantiate");
	}

	// Destroy the intermediate graph object.
	if (graph) {
		IREE_CUDA_NEW_IGNORE_ERROR(syms, cuGraphDestroy(graph));
	}

	if (iree_status_is_ok(status)) {
		*out_graph_exec = graph_exec;
	}

	IREE_TRACE_ZONE_END(z0);
	return status;
}
