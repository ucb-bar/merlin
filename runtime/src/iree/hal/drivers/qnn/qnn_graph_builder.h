// QNN graph builder — runtime side of the "qnn-graph" executable format.
//
// The compiler's serializeGraph (compiler/plugins/target/QNN/Codegen/) emits
// a versioned binary description of a QNN graph. At load time the runtime
// walks the description, calls QnnTensor_createGraphTensor / QnnGraph_addNode
// for each entry, and finalizes the graph via QnnGraph_finalize. The result
// is the same QnnGraph_handle that QnnContext_createFromBinary produces for
// HTA's pre-built ctxbin path — execution is identical from there on.
//
// Format spec lives in compiler/plugins/target/QNN/Codegen/SerializeGraph.h.
//
// This file is the C runtime parser/builder; it's loaded by qnn_executable.c
// when the executable's format string is "qnn-graph" (vs the existing
// "qnn-context-binary" which goes through QnnContext_createFromBinary).

#ifndef IREE_HAL_DRIVERS_QNN_QNN_GRAPH_BUILDER_H_
#define IREE_HAL_DRIVERS_QNN_QNN_GRAPH_BUILDER_H_

#include <stdint.h>

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Format string sister of IREE_HAL_QNN_CONTEXT_BINARY_FORMAT.
extern const char IREE_HAL_QNN_GRAPH_FORMAT[];

// `qnn_executable.h` defines `iree_hal_qnn_interface_t` as `const void*`
// (an opaque alias for `const QnnInterface_t*`); pull it in.
#include "qnn_executable.h"

// Build a finalized graph on `backend_handle` from the binary description in
// `data` (size `data_size`). On success `*out_graph_handle` holds the
// finalized handle and `*out_input_count` / `*out_output_count` describe the
// graph's IO surface so the executable can wire bindings.
//
// The graph is finalized eagerly here (vs deferred). For HTP this includes
// HTP-specific finalize-time optimizations; for GPU it's a relatively cheap
// JIT. For HTA this is also the in-compiler-codegen entry point (it
// bypasses ctxbin since we're constructing the graph directly via the
// QNN API).
//
// `*out_inputs` / `*out_outputs` (typed as `void*` to keep this header
// QNN-header-free) are heap-allocated arrays of `Qnn_Tensor_t` describing
// the graph's IO prototypes. Caller takes ownership of the arrays AND
// each tensor's nested name/dimensions buffers (allocated via
// `host_allocator`); free with `iree_hal_qnn_graph_builder_free_io`.
iree_status_t iree_hal_qnn_graph_builder_create(
	iree_hal_qnn_interface_t qnn_interface, void *backend_handle,
	void *device_handle, void *context_handle, const uint8_t *data,
	size_t data_size, void **out_graph_handle,
	iree_host_size_t *out_input_count, iree_host_size_t *out_output_count,
	void **out_inputs, void **out_outputs);

// Frees the IO arrays returned by `iree_hal_qnn_graph_builder_create`.
// Mirrors the per-tensor name/dimensions free path in qnn_executable.c.
void iree_hal_qnn_graph_builder_free_io(void *inputs,
	iree_host_size_t input_count, void *outputs, iree_host_size_t output_count);

#ifdef __cplusplus
}
#endif

#endif // IREE_HAL_DRIVERS_QNN_QNN_GRAPH_BUILDER_H_
