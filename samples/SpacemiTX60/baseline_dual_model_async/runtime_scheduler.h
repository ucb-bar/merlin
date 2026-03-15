#ifndef MERLIN_SAMPLES_BASELINE_DUAL_MODEL_ASYNC_RUNTIME_SCHEDULER_H_
#define MERLIN_SAMPLES_BASELINE_DUAL_MODEL_ASYNC_RUNTIME_SCHEDULER_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// Runtime configuration for the dual-model baseline.
//
// Important:
// - Both VMFBs are loaded into the same IREE runtime session.
// - Function names must be fully-qualified (<module>.<function>), for example:
//   "dronet.main_graph" and "mlp.main_graph".
// - If both compiled modules are still named "module", rename at compile time
//   so both can coexist in one session.
// - Functions must use async-external ABI:
//   iree.abi.model = "coarse-fences" (compile with
//   --iree-execution-model=async-external).
typedef struct merlin_dual_model_runtime_config_t {
	const char *dronet_vmfb_path;
	const char *mlp_vmfb_path;
	const char *dronet_function;
	const char *mlp_function;
	const char *driver_name;

	// Target periodic invocation frequency for MLP.
	// Must be > 0.
	double mlp_frequency_hz;

	// Frequency of synthetic Dronet sensor samples.
	// Dronet runs as fast as possible and consumes the latest generated sample.
	// Must be > 0.
	double dronet_sensor_frequency_hz;

	// Frequency of synthetic MLP sensor samples.
	// MLP invocations run at mlp_frequency_hz and consume the latest sample.
	// Must be > 0.
	double mlp_sensor_frequency_hz;

	// Stats report frequency to stdout.
	// Must be > 0.
	double report_frequency_hz;

	// Run duration in milliseconds. If <= 0, runs until externally terminated.
	int64_t run_duration_ms;
} merlin_dual_model_runtime_config_t;

// Runs the dual-model baseline runtime.
// Returns 0 on success, non-zero on error.
int merlin_dual_model_runtime_run(
	const merlin_dual_model_runtime_config_t *config);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // MERLIN_SAMPLES_BASELINE_DUAL_MODEL_ASYNC_RUNTIME_SCHEDULER_H_
