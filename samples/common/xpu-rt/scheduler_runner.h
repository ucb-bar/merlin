// samples/common/xpu-rt/scheduler_runner.h
//
// Generic N-target dispatch scheduler with pinned local-task devices,
// phase-locked release timing, and dependency-driven dispatch chains.
// Target-agnostic — hardware-specific defaults (core layout, ISA variants,
// platform name) are supplied by the caller via the config struct.

#ifndef MERLIN_RUNNERS_SCHEDULER_RUNNER_H_
#define MERLIN_RUNNERS_SCHEDULER_RUNNER_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Maximum number of hardware targets the scheduler supports. */
#define SCHEDULER_MAX_TARGETS 16

typedef struct scheduler_runner_config_t {
	const char *graph_json_path;

	const char *driver_name;
	int graph_iters;
	int warmup_iters; // Graph iterations to run before tracing (0 = none).
	int dispatch_iters;
	int report_every;

	const char *vmfb_root_dir;

	// --- N-target configuration ---
	// Set num_targets > 0 and populate the arrays to use N-target mode.
	// Each array must have at least num_targets entries.
	int num_targets;
	const char **target_names; // e.g. ["CPU_P", "CPU_E", "CPU_AUX"]
	const char **target_cpu_ids; // e.g. ["0,1,2,3", "4,5", "6,7"]
	const char **target_variant_dirs; // e.g. ["RVV", "scalar", "scalar"]

	// --- Legacy 2-target configuration ---
	// If num_targets == 0 and cpu_p_cpu_ids is set, auto-creates a 2-target
	// configuration from these fields for backward compatibility.
	const char *cpu_p_cpu_ids;
	const char *cpu_e_cpu_ids;

	int visible_cores;

	const char *out_json_path;
	const char *out_dot_path;
	const char *trace_csv_path;

	// Target configuration — callers set these to their platform values.
	const char
		*target_platform; // e.g. "spacemit_x60" (for VMFB path resolution)

	// Legacy per-target variant dirs (used when num_targets == 0).
	const char *variant_p_dir; // e.g. "RVV" — ISA variant dir for CPU_P
	const char *variant_e_dir; // e.g. "scalar" — ISA variant dir for CPU_E

	const char *elf_marker; // e.g. "_embedded_elf_riscv_64" (NULL = skip)
} scheduler_runner_config_t;

int scheduler_runner_run(const scheduler_runner_config_t *cfg);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // MERLIN_RUNNERS_SCHEDULER_RUNNER_H_
