// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// QNN context-binary executable. Loads .qnn-ctx blobs (output of
// `qnn-context-binary-generator`) via QnnContext_createFromBinary on the
// borrowed device's QnnBackend + QnnDevice handles. Caches the resulting
// Qnn_GraphHandle_t per entry-symbol so dispatch path can look up the graph
// in O(1) at submission time.

#include "qnn_executable.h"
#include "qnn_graph_builder.h"

#include <stdbool.h>
#include <stddef.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/internal/dynamic_library.h"

// QNN SDK — only included from this .c so the public header stays SDK-free.
#include "QnnCommon.h"
#include "QnnContext.h"
#include "QnnGraph.h"
#include "QnnInterface.h"
#include "System/QnnSystemContext.h"
#include "System/QnnSystemInterface.h"

const char IREE_HAL_QNN_CONTEXT_BINARY_FORMAT[] = "qnn-context-binary";

// -----------------------------------------------------------------------------
// iree_hal_qnn_executable_t
// -----------------------------------------------------------------------------

typedef struct iree_hal_qnn_executable_graph_t {
	iree_string_view_t entry_symbol;
	Qnn_GraphHandle_t graph_handle;

	// Cloned tensor prototypes (name, dataType, rank, dims) from
	// QnnSystemContext_getBinaryInfo. Owned by the executable; freed in
	// destroy(). We also clone the dimensions array because the system
	// context's memory is released as soon as enumeration is done.
	Qnn_Tensor_t *inputs;
	iree_host_size_t num_inputs;
	Qnn_Tensor_t *outputs;
	iree_host_size_t num_outputs;
} iree_hal_qnn_executable_graph_t;

typedef struct iree_hal_qnn_executable_t {
	iree_hal_resource_t resource;
	iree_allocator_t host_allocator;

	// Borrowed from the device.
	const QnnInterface_t *qnn_interface;
	Qnn_BackendHandle_t backend_handle;
	Qnn_DeviceHandle_t device_handle;

	// Owned: created by QnnContext_createFromBinary; freed in destroy().
	Qnn_ContextHandle_t context_handle;

	// Cached graph handles keyed by entry symbol.
	iree_host_size_t graph_count;
	iree_hal_qnn_executable_graph_t *graphs;
} iree_hal_qnn_executable_t;

static const iree_hal_executable_vtable_t iree_hal_qnn_executable_vtable;

static iree_hal_qnn_executable_t *iree_hal_qnn_executable_cast(
	iree_hal_executable_t *base_value) {
	IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_qnn_executable_vtable);
	return (iree_hal_qnn_executable_t *)base_value;
}

bool iree_hal_qnn_executable_isa(iree_hal_executable_t *executable) {
	return iree_hal_resource_is((const iree_hal_resource_t *)executable,
		&iree_hal_qnn_executable_vtable);
}

// Capacity for the lazily-built entry-symbol→graph_handle cache. Larger
// than any QNN graph count we expect per chunk in practice.
#define IREE_HAL_QNN_EXECUTABLE_MAX_GRAPHS 16

// Clones a single Qnn_Tensor_t prototype: copies the struct itself, the
// name string, and the dimensions array. The .qnn-ctx system context's
// memory is released after enumeration, so we must own these fields.
static iree_status_t iree_hal_qnn_clone_tensor_proto(
	iree_allocator_t host_allocator, const Qnn_Tensor_t *src,
	Qnn_Tensor_t *dst) {
	*dst = *src;
	// V1 / V2 prototype layouts agree on name+rank+dimensions positions.
	const Qnn_TensorV1_t *sv1 = NULL;
	Qnn_TensorV1_t *dv1 = NULL;
	if (src->version == QNN_TENSOR_VERSION_1) {
		sv1 = &src->v1;
		dv1 = &dst->v1;
	} else if (src->version == QNN_TENSOR_VERSION_2) {
		// V2 has identical layout for the fields we need; reach into v1 via
		// the shared union prefix.
		sv1 = (const Qnn_TensorV1_t *)&src->v1;
		dv1 = (Qnn_TensorV1_t *)&dst->v1;
	} else {
		return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
			"QNN tensor version %d not supported", (int)src->version);
	}

	// Clone name.
	if (sv1->name) {
		size_t name_len = strlen(sv1->name);
		char *name_buf = NULL;
		IREE_RETURN_IF_ERROR(iree_allocator_malloc(
			host_allocator, name_len + 1, (void **)&name_buf));
		memcpy(name_buf, sv1->name, name_len + 1);
		dv1->name = name_buf;
	}
	// Clone dimensions.
	if (sv1->dimensions && sv1->rank > 0) {
		uint32_t *dim_buf = NULL;
		IREE_RETURN_IF_ERROR(iree_allocator_malloc(
			host_allocator, sv1->rank * sizeof(uint32_t), (void **)&dim_buf));
		memcpy(dim_buf, sv1->dimensions, sv1->rank * sizeof(uint32_t));
		dv1->dimensions = dim_buf;
	}
	// clientBuf will be overwritten at dispatch time with IREE buffer
	// mappings; nuke it here so accidental reuse is obvious.
	dv1->clientBuf.data = NULL;
	dv1->clientBuf.dataSize = 0;
	return iree_ok_status();
}

// Adds an entry to the executable's graph cache. Called from lookup_graph
// after a successful graphRetrieve.
static iree_status_t iree_hal_qnn_executable_cache_add(
	iree_hal_qnn_executable_t *executable, iree_string_view_t entry_symbol,
	Qnn_GraphHandle_t graph_handle) {
	if (executable->graph_count >= IREE_HAL_QNN_EXECUTABLE_MAX_GRAPHS) {
		return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
			"QNN executable graph cache full (max=%d)",
			IREE_HAL_QNN_EXECUTABLE_MAX_GRAPHS);
	}
	// Allocate string storage and copy the name in.
	char *name_storage = NULL;
	IREE_RETURN_IF_ERROR(iree_allocator_malloc(
		executable->host_allocator, entry_symbol.size, (void **)&name_storage));
	memcpy(name_storage, entry_symbol.data, entry_symbol.size);
	iree_hal_qnn_executable_graph_t *g =
		&executable->graphs[executable->graph_count++];
	g->entry_symbol = iree_make_string_view(name_storage, entry_symbol.size);
	g->graph_handle = graph_handle;
	g->inputs = NULL;
	g->num_inputs = 0;
	g->outputs = NULL;
	g->num_outputs = 0;
	return iree_ok_status();
}

// Clones the per-graph IO prototype arrays into the executable-owned
// storage of the most recently added graph entry.
static iree_status_t iree_hal_qnn_executable_clone_graph_io(
	iree_hal_qnn_executable_t *executable, const Qnn_Tensor_t *src_inputs,
	iree_host_size_t num_inputs, const Qnn_Tensor_t *src_outputs,
	iree_host_size_t num_outputs) {
	iree_hal_qnn_executable_graph_t *g =
		&executable->graphs[executable->graph_count - 1];
	if (num_inputs > 0) {
		IREE_RETURN_IF_ERROR(iree_allocator_malloc(executable->host_allocator,
			num_inputs * sizeof(Qnn_Tensor_t), (void **)&g->inputs));
		for (iree_host_size_t i = 0; i < num_inputs; ++i) {
			IREE_RETURN_IF_ERROR(iree_hal_qnn_clone_tensor_proto(
				executable->host_allocator, &src_inputs[i], &g->inputs[i]));
		}
		g->num_inputs = num_inputs;
	}
	if (num_outputs > 0) {
		IREE_RETURN_IF_ERROR(iree_allocator_malloc(executable->host_allocator,
			num_outputs * sizeof(Qnn_Tensor_t), (void **)&g->outputs));
		for (iree_host_size_t i = 0; i < num_outputs; ++i) {
			IREE_RETURN_IF_ERROR(iree_hal_qnn_clone_tensor_proto(
				executable->host_allocator, &src_outputs[i], &g->outputs[i]));
		}
		g->num_outputs = num_outputs;
	}
	return iree_ok_status();
}

// Walks the QNN context-binary's graph list via QnnSystemContext (driver-
// cached when |system_interface| is non-NULL, otherwise per-call dlopen of
// libQnnSystem.so). For each graph: calls QnnGraph_retrieve + caches
// (name, handle), and clones the input/output Qnn_Tensor_t prototypes.
// Errors are absorbed (non-fatal): callers fall back to the by-name path.
static void iree_hal_qnn_executable_enumerate_graphs(
	iree_hal_qnn_executable_t *executable, iree_const_byte_span_t blob,
	const void *system_interface, iree_allocator_t host_allocator) {
	iree_dynamic_library_t *sys_lib = NULL;
	const QnnSystemInterface_t *sys_iface =
		(const QnnSystemInterface_t *)system_interface;

	if (sys_iface == NULL) {
		iree_status_t s = iree_dynamic_library_load_from_file("libQnnSystem.so",
			IREE_DYNAMIC_LIBRARY_FLAG_NONE, host_allocator, &sys_lib);
		if (iree_status_is_ok(s)) {
			typedef Qnn_ErrorHandle_t (*get_providers_fn_t)(
				const QnnSystemInterface_t ***providers, uint32_t *num);
			get_providers_fn_t get_providers = NULL;
			s = iree_dynamic_library_lookup_symbol(sys_lib,
				"QnnSystemInterface_getProviders", (void **)&get_providers);
			if (iree_status_is_ok(s) && get_providers) {
				const QnnSystemInterface_t **providers = NULL;
				uint32_t num_providers = 0;
				if (get_providers(&providers, &num_providers) == QNN_SUCCESS &&
					num_providers > 0 && providers != NULL) {
					sys_iface = providers[0];
				}
			}
		}
		iree_status_ignore(s);
	}
	if (sys_iface == NULL) {
		if (sys_lib)
			iree_dynamic_library_release(sys_lib);
		return;
	}

	QnnSystemContext_Handle_t sys_ctx = NULL;
	if (sys_iface->QNN_SYSTEM_INTERFACE_VER_NAME.systemContextCreate(
			&sys_ctx) != QNN_SUCCESS ||
		sys_ctx == NULL) {
		if (sys_lib)
			iree_dynamic_library_release(sys_lib);
		return;
	}

	const QnnSystemContext_BinaryInfo_t *info = NULL;
	Qnn_ContextBinarySize_t info_size = 0;
	Qnn_ErrorHandle_t binfo_rc =
		sys_iface->QNN_SYSTEM_INTERFACE_VER_NAME.systemContextGetBinaryInfo(
			sys_ctx, (void *)blob.data, (uint64_t)blob.data_length, &info,
			&info_size);
	if (binfo_rc == QNN_SUCCESS && info != NULL) {
		const QnnSystemContext_GraphInfo_t *graphs = NULL;
		uint32_t num_graphs = 0;
		switch (info->version) {
			case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1:
				graphs = info->contextBinaryInfoV1.graphs;
				num_graphs = info->contextBinaryInfoV1.numGraphs;
				break;
			case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2:
				graphs = info->contextBinaryInfoV2.graphs;
				num_graphs = info->contextBinaryInfoV2.numGraphs;
				break;
			case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3:
				graphs = info->contextBinaryInfoV3.graphs;
				num_graphs = info->contextBinaryInfoV3.numGraphs;
				break;
			default:
				break;
		}
		const QnnInterface_t *iface = executable->qnn_interface;
		for (uint32_t i = 0; i < num_graphs &&
			 executable->graph_count < IREE_HAL_QNN_EXECUTABLE_MAX_GRAPHS;
			 ++i) {
			const char *graph_name = NULL;
			const Qnn_Tensor_t *g_inputs = NULL;
			uint32_t g_num_inputs = 0;
			const Qnn_Tensor_t *g_outputs = NULL;
			uint32_t g_num_outputs = 0;
			switch (graphs[i].version) {
				case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1:
					graph_name = graphs[i].graphInfoV1.graphName;
					g_inputs = graphs[i].graphInfoV1.graphInputs;
					g_num_inputs = graphs[i].graphInfoV1.numGraphInputs;
					g_outputs = graphs[i].graphInfoV1.graphOutputs;
					g_num_outputs = graphs[i].graphInfoV1.numGraphOutputs;
					break;
				case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2:
					graph_name = graphs[i].graphInfoV2.graphName;
					g_inputs = graphs[i].graphInfoV2.graphInputs;
					g_num_inputs = graphs[i].graphInfoV2.numGraphInputs;
					g_outputs = graphs[i].graphInfoV2.graphOutputs;
					g_num_outputs = graphs[i].graphInfoV2.numGraphOutputs;
					break;
				case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3:
					graph_name = graphs[i].graphInfoV3.graphName;
					g_inputs = graphs[i].graphInfoV3.graphInputs;
					g_num_inputs = graphs[i].graphInfoV3.numGraphInputs;
					g_outputs = graphs[i].graphInfoV3.graphOutputs;
					g_num_outputs = graphs[i].graphInfoV3.numGraphOutputs;
					break;
				default:
					break;
			}
			if (graph_name == NULL)
				continue;
			Qnn_GraphHandle_t handle = NULL;
			Qnn_ErrorHandle_t rc = iface->QNN_INTERFACE_VER_NAME.graphRetrieve(
				executable->context_handle, graph_name, &handle);
			if (rc != QNN_SUCCESS || handle == NULL)
				continue;
			iree_string_view_t sv = iree_make_cstring_view(graph_name);
			iree_status_t add_status =
				iree_hal_qnn_executable_cache_add(executable, sv, handle);
			if (!iree_status_is_ok(add_status)) {
				iree_status_ignore(add_status);
				continue;
			}
			iree_status_t clone_status = iree_hal_qnn_executable_clone_graph_io(
				executable, g_inputs, (iree_host_size_t)g_num_inputs, g_outputs,
				(iree_host_size_t)g_num_outputs);
			if (!iree_status_is_ok(clone_status))
				iree_status_ignore(clone_status);
		}
	}
	sys_iface->QNN_SYSTEM_INTERFACE_VER_NAME.systemContextFree(sys_ctx);
	if (sys_lib)
		iree_dynamic_library_release(sys_lib);
}

iree_status_t iree_hal_qnn_executable_create(
	const iree_hal_executable_params_t *executable_params,
	iree_hal_qnn_interface_t qnn_interface,
	iree_hal_qnn_backend_handle_t backend_handle,
	iree_hal_qnn_device_handle_t device_handle, const void *system_interface,
	iree_allocator_t host_allocator, iree_hal_executable_t **out_executable) {
	IREE_ASSERT_ARGUMENT(executable_params);
	IREE_ASSERT_ARGUMENT(qnn_interface);
	IREE_ASSERT_ARGUMENT(out_executable);
	IREE_TRACE_ZONE_BEGIN(z0);
	*out_executable = NULL;

	const bool is_ctxbin =
		iree_string_view_equal(executable_params->executable_format,
			iree_make_cstring_view(IREE_HAL_QNN_CONTEXT_BINARY_FORMAT));
	// Accept "qnn-graph" and any "qnn-graph-{hta,gpu,htp}" suffixed form;
	// the cache's can_prepare_format already rejected mismatched suffixes.
	const bool is_graph =
		iree_string_view_starts_with(executable_params->executable_format,
			iree_make_cstring_view(IREE_HAL_QNN_GRAPH_FORMAT));
	if (!is_ctxbin && !is_graph) {
		IREE_TRACE_ZONE_END(z0);
		return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
			"QNN HAL executable expected format '%s' or '%s', got '%.*s'",
			IREE_HAL_QNN_CONTEXT_BINARY_FORMAT, IREE_HAL_QNN_GRAPH_FORMAT,
			(int)executable_params->executable_format.size,
			executable_params->executable_format.data);
	}

	// Allocate the executable + an inline cache for graph handles.
	iree_host_size_t total_size = sizeof(iree_hal_qnn_executable_t) +
		IREE_HAL_QNN_EXECUTABLE_MAX_GRAPHS *
			sizeof(iree_hal_qnn_executable_graph_t);
	iree_hal_qnn_executable_t *executable = NULL;
	IREE_RETURN_AND_END_ZONE_IF_ERROR(z0,
		iree_allocator_malloc(
			host_allocator, total_size, (void **)&executable));
	iree_hal_resource_initialize(
		&iree_hal_qnn_executable_vtable, &executable->resource);
	executable->host_allocator = host_allocator;
	executable->qnn_interface = (const QnnInterface_t *)qnn_interface;
	executable->backend_handle = (Qnn_BackendHandle_t)backend_handle;
	executable->device_handle = (Qnn_DeviceHandle_t)device_handle;
	executable->context_handle = NULL;
	executable->graphs = (iree_hal_qnn_executable_graph_t *)(executable + 1);
	executable->graph_count = 0;

	iree_const_byte_span_t blob = executable_params->executable_data;
	if (blob.data_length == 0 || blob.data == NULL) {
		iree_hal_executable_release((iree_hal_executable_t *)executable);
		IREE_TRACE_ZONE_END(z0);
		return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
			"qnn executable_data is empty (size=%zu)",
			(size_t)blob.data_length);
	}

	if (is_graph) {
		// In-compiler-codegen path: create a fresh QnnContext first (no
		// ctxbin to import — we're going to populate it via graphCreate /
		// tensorCreateGraphTensor / graphAddNode / graphFinalize), then
		// hand the context off to the graph builder which walks the binary
		// qnn-graph blob and materializes the graph on-device.
		const QnnInterface_t *iface = executable->qnn_interface;
		Qnn_ErrorHandle_t qnn_rc = iface->QNN_INTERFACE_VER_NAME.contextCreate(
			executable->backend_handle, executable->device_handle,
			/*config=*/NULL, &executable->context_handle);
		if (qnn_rc != QNN_SUCCESS || executable->context_handle == NULL) {
			iree_hal_executable_release((iree_hal_executable_t *)executable);
			IREE_TRACE_ZONE_END(z0);
			return iree_make_status(IREE_STATUS_INTERNAL,
				"QnnContext_create failed for qnn-graph executable rc=%lld",
				(long long)qnn_rc);
		}
		void *graph_handle_raw = NULL;
		iree_host_size_t input_count = 0, output_count = 0;
		void *in_protos_raw = NULL;
		void *out_protos_raw = NULL;
		iree_status_t bs = iree_hal_qnn_graph_builder_create(qnn_interface,
			backend_handle, device_handle, executable->context_handle,
			blob.data, blob.data_length, &graph_handle_raw, &input_count,
			&output_count, &in_protos_raw, &out_protos_raw);
		if (!iree_status_is_ok(bs)) {
			iree_hal_executable_release((iree_hal_executable_t *)executable);
			IREE_TRACE_ZONE_END(z0);
			return bs;
		}
		// Register the freshly-built graph in the cache so dispatch can
		// look it up by ordinal/symbol. The dispatch path's
		// lookup_graph_by_ordinal(0) fires for single-graph variants; the
		// entry symbol is the name we used during graphCreate.
		iree_string_view_t entry_sym =
			iree_make_cstring_view("merlin_qnn_graph");
		iree_status_t cache_status = iree_hal_qnn_executable_cache_add(
			executable, entry_sym, (Qnn_GraphHandle_t)graph_handle_raw);
		if (!iree_status_is_ok(cache_status)) {
			iree_hal_qnn_graph_builder_free_io(
				in_protos_raw, input_count, out_protos_raw, output_count);
			iree_hal_executable_release((iree_hal_executable_t *)executable);
			IREE_TRACE_ZONE_END(z0);
			return cache_status;
		}
		// Clone the IO prototypes into executable-owned storage so the
		// dispatch path can populate Qnn_Tensor_t structs for binding.
		iree_status_t clone_status = iree_hal_qnn_executable_clone_graph_io(
			executable, (const Qnn_Tensor_t *)in_protos_raw, input_count,
			(const Qnn_Tensor_t *)out_protos_raw, output_count);
		iree_hal_qnn_graph_builder_free_io(
			in_protos_raw, input_count, out_protos_raw, output_count);
		if (!iree_status_is_ok(clone_status)) {
			iree_hal_executable_release((iree_hal_executable_t *)executable);
			IREE_TRACE_ZONE_END(z0);
			return clone_status;
		}
		*out_executable = (iree_hal_executable_t *)executable;
		IREE_TRACE_ZONE_END(z0);
		return iree_ok_status();
	}

	// Legacy ctxbin path.
	const QnnInterface_t *iface = executable->qnn_interface;
	Qnn_ErrorHandle_t qnn_rc =
		iface->QNN_INTERFACE_VER_NAME.contextCreateFromBinary(
			executable->backend_handle, executable->device_handle,
			/*config=*/NULL, blob.data,
			(Qnn_ContextBinarySize_t)blob.data_length,
			&executable->context_handle, /*profile=*/NULL);
	if (qnn_rc != QNN_SUCCESS) {
		iree_hal_executable_release((iree_hal_executable_t *)executable);
		IREE_TRACE_ZONE_END(z0);
		return iree_make_status(IREE_STATUS_INTERNAL,
			"QnnContext_createFromBinary failed with rc=%lld for blob of size "
			"%zu",
			(long long)qnn_rc, (size_t)blob.data_length);
	}

	// Eagerly enumerate graphs in the binary via QnnSystemContext so we can
	// populate the ordinal → graph cache. Prefer the driver-cached
	// QnnSystemInterface (one dlopen per process). Fall back to a per-call
	// dlopen of libQnnSystem.so if the caller did not supply it (e.g.
	// standalone tests). Failure here is non-fatal: lookup-by-name still
	// works.
	iree_hal_qnn_executable_enumerate_graphs(
		executable, blob, system_interface, host_allocator);

	*out_executable = (iree_hal_executable_t *)executable;
	IREE_TRACE_ZONE_END(z0);
	return iree_ok_status();
}

static void iree_hal_qnn_executable_destroy(
	iree_hal_executable_t *base_executable) {
	iree_hal_qnn_executable_t *executable =
		iree_hal_qnn_executable_cast(base_executable);
	iree_allocator_t host_allocator = executable->host_allocator;
	IREE_TRACE_ZONE_BEGIN(z0);

	if (executable->context_handle && executable->qnn_interface) {
		const QnnInterface_t *iface = executable->qnn_interface;
		Qnn_ErrorHandle_t rc = iface->QNN_INTERFACE_VER_NAME.contextFree(
			executable->context_handle, /*profile=*/NULL);
		(void)rc;
	}
	// Free per-graph storage: entry symbol, IO prototype arrays, and the
	// cloned name+dimensions buffers inside each prototype.
	for (iree_host_size_t i = 0; i < executable->graph_count; ++i) {
		iree_hal_qnn_executable_graph_t *g = &executable->graphs[i];
		iree_allocator_free(host_allocator, (void *)g->entry_symbol.data);
		for (iree_host_size_t j = 0; j < g->num_inputs; ++j) {
			Qnn_TensorV1_t *v1 = (Qnn_TensorV1_t *)&g->inputs[j].v1;
			iree_allocator_free(host_allocator, (void *)v1->name);
			iree_allocator_free(host_allocator, (void *)v1->dimensions);
		}
		iree_allocator_free(host_allocator, g->inputs);
		for (iree_host_size_t j = 0; j < g->num_outputs; ++j) {
			Qnn_TensorV1_t *v1 = (Qnn_TensorV1_t *)&g->outputs[j].v1;
			iree_allocator_free(host_allocator, (void *)v1->name);
			iree_allocator_free(host_allocator, (void *)v1->dimensions);
		}
		iree_allocator_free(host_allocator, g->outputs);
	}
	iree_allocator_free(host_allocator, executable);
	IREE_TRACE_ZONE_END(z0);
}

iree_hal_qnn_graph_handle_t iree_hal_qnn_executable_lookup_graph_by_ordinal(
	iree_hal_executable_t *base_executable, iree_host_size_t ordinal) {
	iree_hal_qnn_executable_t *executable =
		iree_hal_qnn_executable_cast(base_executable);
	if (ordinal >= executable->graph_count)
		return NULL;
	return (iree_hal_qnn_graph_handle_t)executable->graphs[ordinal]
		.graph_handle;
}

iree_status_t iree_hal_qnn_executable_get_graph_io(
	iree_hal_executable_t *base_executable, iree_host_size_t ordinal,
	iree_hal_qnn_tensor_proto_t **out_inputs, iree_host_size_t *out_num_inputs,
	iree_hal_qnn_tensor_proto_t **out_outputs,
	iree_host_size_t *out_num_outputs) {
	iree_hal_qnn_executable_t *executable =
		iree_hal_qnn_executable_cast(base_executable);
	if (ordinal >= executable->graph_count) {
		return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
			"QNN executable: ordinal %zu out of range "
			"(graph_count=%zu)",
			(size_t)ordinal, (size_t)executable->graph_count);
	}
	iree_hal_qnn_executable_graph_t *g = &executable->graphs[ordinal];
	if (out_inputs)
		*out_inputs = (iree_hal_qnn_tensor_proto_t *)g->inputs;
	if (out_num_inputs)
		*out_num_inputs = g->num_inputs;
	if (out_outputs)
		*out_outputs = (iree_hal_qnn_tensor_proto_t *)g->outputs;
	if (out_num_outputs)
		*out_num_outputs = g->num_outputs;
	return iree_ok_status();
}

iree_hal_qnn_graph_handle_t iree_hal_qnn_executable_lookup_graph(
	iree_hal_executable_t *base_executable, iree_string_view_t entry_symbol) {
	iree_hal_qnn_executable_t *executable =
		iree_hal_qnn_executable_cast(base_executable);

	// Cache hit?
	for (iree_host_size_t i = 0; i < executable->graph_count; ++i) {
		if (iree_string_view_equal(
				executable->graphs[i].entry_symbol, entry_symbol)) {
			return (iree_hal_qnn_graph_handle_t)executable->graphs[i]
				.graph_handle;
		}
	}

	// Cache miss — call QnnGraph_retrieve to look up by name in the loaded
	// context, then cache for future calls.
	char name_cstr[256];
	iree_host_size_t copy_len = entry_symbol.size < sizeof(name_cstr) - 1
		? entry_symbol.size
		: sizeof(name_cstr) - 1;
	memcpy(name_cstr, entry_symbol.data, copy_len);
	name_cstr[copy_len] = '\0';
	const QnnInterface_t *iface = executable->qnn_interface;
	Qnn_GraphHandle_t handle = NULL;
	Qnn_ErrorHandle_t rc = iface->QNN_INTERFACE_VER_NAME.graphRetrieve(
		executable->context_handle, name_cstr, &handle);
	if (rc != QNN_SUCCESS || handle == NULL) {
		return NULL;
	}
	iree_status_t add_status =
		iree_hal_qnn_executable_cache_add(executable, entry_symbol, handle);
	if (!iree_status_is_ok(add_status)) {
		iree_status_ignore(add_status);
	}
	return (iree_hal_qnn_graph_handle_t)handle;
}

static const iree_hal_executable_vtable_t iree_hal_qnn_executable_vtable = {
	.destroy = iree_hal_qnn_executable_destroy,
};
