// samples/SpacemiTX60/b_dispatch_level_model_async/runtime_scheduler.h

#ifndef RUNTIME_DISPATCH_GRAPH_H_
#define RUNTIME_DISPATCH_GRAPH_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct dispatch_graph_config_t {
	// Required:
	const char *graph_json_path;

	// Optional:
	const char *driver_name; // must be "local-task" for strict split
							 // CPU_P/CPU_E execution
	int graph_iters; // default 1
	int dispatch_iters; // default 1; passed when export signature is irr
	int report_every; // default 0

	// Optional:
	const char *vmfb_root_dir; // if JSON uses module_name, runner resolves
							   // <vmfb_root_dir>/<module_name>.vmfb

	// Required for strict heterogeneous local-task partitioning:
	const char *cpu_p_cpu_ids; // exactly 4 logical CPUs, default "0,1,2,3"
	const char *cpu_e_cpu_ids; // exactly 2 logical CPUs, default "4,5"
	int visible_cores; // validate IDs in [0, visible_cores), default 8

	// Optional outputs:
	const char *out_json_path;
	const char *out_dot_path;
	const char *trace_csv_path;
} dispatch_graph_config_t;

// Returns 0 on success, 1 on failure.
int dispatch_graph_run(const dispatch_graph_config_t *cfg);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // RUNTIME_DISPATCH_GRAPH_H_
