// xpu-rt: minimal public C API for the scheduler/runtime wrapper.
//
// This header is intentionally IREE-agnostic and can be installed and used
// from projects that do not directly depend on IREE headers.

#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct xpurt_graph xpurt_graph_t;
typedef struct xpurt_context xpurt_context_t;

typedef enum {
	XPURT_STATUS_OK = 0,
	XPURT_STATUS_ERROR = 1,
} xpurt_status_t;

// Loads a dispatch-graph JSON description into an opaque graph handle.
xpurt_status_t xpurt_graph_load(
	const char *json_path, xpurt_graph_t **out_graph);

// Destroys a graph handle returned by xpurt_graph_load.
void xpurt_graph_destroy(xpurt_graph_t *graph);

// Graph accessors (for backend use; indices refer to node array order).
size_t xpurt_graph_node_count(xpurt_graph_t *graph);
const int *xpurt_graph_execution_order(xpurt_graph_t *graph);
const char *xpurt_graph_node_key(xpurt_graph_t *graph, size_t index);
int xpurt_graph_node_id(xpurt_graph_t *graph, size_t index);
const char *xpurt_graph_node_vmfb_path(xpurt_graph_t *graph, size_t index);

// Convenience helper: one-shot run of a dispatch-graph JSON using a given
// driver name (e.g. "local-task"). This will internally create/destroy any
// runtime state needed for the run.
xpurt_status_t xpurt_run_dispatch_graph(
	const char *json_path, const char *driver_name);

#ifdef __cplusplus
} // extern "C"
#endif
