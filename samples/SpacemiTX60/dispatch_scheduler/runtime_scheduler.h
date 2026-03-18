// samples/SpacemiTX60/b_dispatch_level_model_async/runtime_scheduler.h

#ifndef RUNTIME_DISPATCH_GRAPH_H_
#define RUNTIME_DISPATCH_GRAPH_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct dispatch_graph_config_t {
	const char *graph_json_path;

	const char *driver_name;
	int graph_iters;
	int dispatch_iters;
	int report_every;

	const char *vmfb_root_dir;

	const char *cpu_p_cpu_ids;
	const char *cpu_e_cpu_ids;
	int visible_cores;

	const char *out_json_path;
	const char *out_dot_path;
	const char *trace_csv_path;
} dispatch_graph_config_t;

int dispatch_graph_run(const dispatch_graph_config_t *cfg);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // RUNTIME_DISPATCH_GRAPH_H_
