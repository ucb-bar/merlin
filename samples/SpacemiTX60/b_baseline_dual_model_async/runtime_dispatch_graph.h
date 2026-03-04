#ifndef RUNTIME_DISPATCH_GRAPH_H_
#define RUNTIME_DISPATCH_GRAPH_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct dispatch_graph_config_t {
  const char* graph_json_path;   // required
  const char* driver_name;       // default "local-task"
  int graph_iters;               // default 1
  int dispatch_iters;            // default 1 (passed if entry takes i32)
  int report_every;              // default 0 (no periodic; final only)

  // Optional: core mask pinning for local-task (best-effort).
  // 0 means "use default device".
  uint64_t core_mask;

  // If >1, enables parallel execution of independent ready nodes.
  // Note: calls sharing same VMFB path are serialized via a per-module mutex.
  int parallelism;
} dispatch_graph_config_t;

// Returns 0 on success, 1 on failure (prints errors to stderr).
int dispatch_graph_run(const dispatch_graph_config_t* cfg);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // RUNTIME_DISPATCH_GRAPH_H_