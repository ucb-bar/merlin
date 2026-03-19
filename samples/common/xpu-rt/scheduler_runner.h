// samples/common/xpu-rt/scheduler_runner.h
//
// Generic two-cluster dispatch scheduler: CPU_P + CPU_E worker threads with
// pinned local-task devices, phase-locked release timing, and dependency-driven
// dispatch chains.  Target-agnostic — hardware-specific defaults (core layout,
// ISA variants, platform name) are supplied by the caller via the config
// struct.

#ifndef MERLIN_RUNNERS_SCHEDULER_RUNNER_H_
#define MERLIN_RUNNERS_SCHEDULER_RUNNER_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct scheduler_runner_config_t {
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

	// Target configuration — callers set these to their platform values.
	const char
		*target_platform; // e.g. "spacemit_x60" (for VMFB path resolution)
	const char *variant_p_dir; // e.g. "RVV" — ISA variant dir for CPU_P
	const char *variant_e_dir; // e.g. "scalar" — ISA variant dir for CPU_E
	const char *elf_marker; // e.g. "_embedded_elf_riscv_64" (NULL = skip)
} scheduler_runner_config_t;

int scheduler_runner_run(const scheduler_runner_config_t *cfg);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // MERLIN_RUNNERS_SCHEDULER_RUNNER_H_
