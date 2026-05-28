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

	// Unified CPU device: when cpu_cpu_ids is non-empty the runner additionally
	// creates a single pinned local-task device covering ALL listed cores, and
	// schedule nodes with hardware_target == "CPU" route to it. This
	// complements the CPU_P / CPU_E split: a heterogeneous schedule may use
	// {CPU, QNN_GPU, QNN_HTA} (3-way: CPU as a single 8-core cluster) instead
	// of the older {CPU_P, CPU_E, ...} split (4-way). Empty = the unified CPU
	// device is not created and CPU-tagged schedules will fail at dispatch
	// time.
	const char *cpu_cpu_ids;

	// QNN (Qualcomm Neural Network HAL) target flags. When non-zero the
	// runner additionally creates an iree_hal_device_t via the QNN HAL driver
	// (libQnn{Gpu,Hta}.so under the hood) and spawns a worker thread for
	// dispatching .qnn-ctx executables to it. Defaults to off so pure-CPU
	// schedules don't depend on QNN runtime libs.
	int qnn_gpu_enabled;
	int qnn_hta_enabled;

	// When non-zero, CPU devices are created as iree_hal_local_sync
	// instead of iree_hal_local_task. local-sync executes each
	// dispatch inline on the worker thread (no task-queue submit + no
	// fence wait round-trip), reducing per-call overhead from
	// ~500us-1ms to ~50us — material for tiny per-layer dispatches
	// like mobilenet's. Default 0 keeps the multi-core local-task
	// path (better for big single-dispatch ops).
	int cpu_use_local_sync;

	const char *out_json_path;
	const char *out_dot_path;
	const char *trace_csv_path;

	// XPU-RT telemetry sink for hardware-in-the-loop feedback.
	// One JSON-Lines record per dispatch end carrying:
	//   {"epoch": ..., "dispatch_id": ..., "target": ..., "start_us": ...,
	//    "end_us": ..., "deadline_miss": ..., "skip_fired": ...,
	//    "planned_duration_us": ..., "planned_start_us": ...}
	// Both fields are independent; if both are set the fd takes priority.
	// When neither is set, no telemetry is emitted (zero overhead — the
	// runner skips the sink path entirely). See docs/merlin_integration.md.
	const char
		*telemetry_jsonl_path; // path to open in append mode (NULL = off)
	int telemetry_fd; // pre-opened fd (caller-owned). > 0 to use; otherwise
					  // unused (memset(0) sets this to 0 which is treated
					  // as "off" so existing callers stay inert).

	// Optional hot-swap path: when set, the runner watches this file
	// between graph iterations and atomically swaps in a new schedule
	// (release-time table) at the next epoch boundary. NULL = disabled.
	const char *schedule_next_path;

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
