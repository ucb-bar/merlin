// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_QNN_QNN_EXECUTABLE_H_
#define IREE_HAL_DRIVERS_QNN_QNN_EXECUTABLE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// The "qnn-context-binary" executable format. The hal.executable.objects
// blob is fed verbatim into QnnContext_createFromBinary; entry symbols are
// QnnGraph_Handles within that context. IREE identifies formats by string
// so this is just a stable identifier matched against the executable_cache.
extern const char IREE_HAL_QNN_CONTEXT_BINARY_FORMAT[];

// Opaque QNN SDK handles. We forward-declare as void* so this header stays
// QNN-SDK-free; the .c file casts back to the real types from QnnInterface.h.
typedef void *iree_hal_qnn_context_handle_t; // Qnn_ContextHandle_t
typedef void *iree_hal_qnn_graph_handle_t; // Qnn_GraphHandle_t
typedef void *iree_hal_qnn_backend_handle_t; // Qnn_BackendHandle_t
typedef void *iree_hal_qnn_device_handle_t; // Qnn_DeviceHandle_t
typedef const void *iree_hal_qnn_interface_t; // const QnnInterface_t*

// Creates an executable from a hal.executable.objects payload that carries a
// QNN context binary (the output of `qnn-context-binary-generator`).
//
// The binary is fed into QnnContext_createFromBinary; the resulting
// Qnn_ContextHandle_t and a map of entry-symbol → Qnn_GraphHandle_t are
// retained for the lifetime of the executable.
//
// |qnn_interface|, |backend_handle|, |device_handle| are borrowed from the
// device — they must outlive the executable.
//
// |system_interface| (cast to const QnnSystemInterface_t*) may be non-NULL
// if the driver pre-loaded libQnnSystem.so. When NULL, executable_create
// falls back to dlopen'ing libQnnSystem.so itself; supplying a cached
// interface avoids that per-executable dlopen + getProviders cost.
iree_status_t iree_hal_qnn_executable_create(
	const iree_hal_executable_params_t *executable_params,
	iree_hal_qnn_interface_t qnn_interface,
	iree_hal_qnn_backend_handle_t backend_handle,
	iree_hal_qnn_device_handle_t device_handle, const void *system_interface,
	iree_allocator_t host_allocator, iree_hal_executable_t **out_executable);

// Returns true if |executable| was created by iree_hal_qnn_executable_create.
bool iree_hal_qnn_executable_isa(iree_hal_executable_t *executable);

// Looks up a Qnn_GraphHandle_t by entry-symbol name. Returns NULL if absent.
iree_hal_qnn_graph_handle_t iree_hal_qnn_executable_lookup_graph(
	iree_hal_executable_t *executable, iree_string_view_t entry_symbol);

// Looks up a Qnn_GraphHandle_t by ordinal. The runtime QNN HAL is currently
// agnostic to graph names — IREE's HAL tracks executable exports by ordinal
// and the only durable mapping into the QNN context is "first graph in the
// binary" for ordinal=0. This helper enumerates the binary via
// QnnSystemContext_getBinaryInfo and returns graph[ordinal]. Returns NULL
// when the ordinal is out of range or enumeration fails. The graph handle
// is cached so subsequent lookups are O(1).
iree_hal_qnn_graph_handle_t iree_hal_qnn_executable_lookup_graph_by_ordinal(
	iree_hal_executable_t *executable, iree_host_size_t ordinal);

// Retrieves cached input/output Qnn_Tensor_t prototypes for a graph (by
// ordinal). The prototypes carry name, dataType, rank, dimensions etc. as
// declared in the .qnn-ctx — the dispatch path copies these into local
// arrays and overrides clientBuf with the IREE buffer mapping. Output
// pointers reference storage owned by the executable and remain valid
// until the executable is destroyed.
//
// Returns ok and zeroes the out_* fields when the ordinal is in range but
// the graph has no IO recorded (shouldn't happen for well-formed binaries
// but defensive). Returns INVALID_ARGUMENT when ordinal is out of range.
//
// The Qnn_Tensor_t* type is opaque to consumers (forward-declared as
// void*); callers cast back to the QNN SDK type.
typedef const void iree_hal_qnn_tensor_proto_t;

iree_status_t iree_hal_qnn_executable_get_graph_io(
	iree_hal_executable_t *executable, iree_host_size_t ordinal,
	iree_hal_qnn_tensor_proto_t **out_inputs, iree_host_size_t *out_num_inputs,
	iree_hal_qnn_tensor_proto_t **out_outputs,
	iree_host_size_t *out_num_outputs);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // IREE_HAL_DRIVERS_QNN_QNN_EXECUTABLE_H_
