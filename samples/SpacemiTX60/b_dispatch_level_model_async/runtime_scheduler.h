#ifndef RUNTIME_DISPATCH_GRAPH_H_
#define RUNTIME_DISPATCH_GRAPH_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct dispatch_graph_config_t {
  // Required:
  const char* graph_json_path;   // path to dronet.q.int8_dispatch_graph.json

  // Optional:
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

  // Optional: directory containing VMFBs on the target.
  // IMPORTANT:
  // - Your JSON has vmfb_path like "dispatches/<file>.vmfb"
  // - If you pass vmfb_root_dir pointing to the *dispatches folder itself*,
  //   the runner will strip the leading "dispatches/" prefix before joining.
  // Example:
  //   --vmfb_dir=/home/baseline/dispatches
  //   JSON vmfb_path "dispatches/module_...vmfb" -> resolves to
  //   /home/baseline/dispatches/module_...vmfb
  const char* vmfb_root_dir;

  // Optional outputs (for reconstruction/plotting):
  // - out_json_path: summary stats + deps + topo order + resolved paths
  // - out_dot_path: DOT graph with latency labels
  // - trace_csv_path: per-dispatch timeline events (start_us, dur_us, thread)
  const char* out_json_path;
  const char* out_dot_path;
  const char* trace_csv_path;
} dispatch_graph_config_t;

// Returns 0 on success, 1 on failure (prints errors to stderr).
int dispatch_graph_run(const dispatch_graph_config_t* cfg);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // RUNTIME_DISPATCH_GRAPH_H_