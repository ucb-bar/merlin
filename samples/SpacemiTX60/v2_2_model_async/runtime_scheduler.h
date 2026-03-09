#ifndef DUAL_MODEL_ASYNC_SCHEDULER_H_
#define DUAL_MODEL_ASYNC_SCHEDULER_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct dual_model_async_config_t {
  const char* dronet_vmfb_path;
  const char* mlp_vmfb_path;

  const char* dronet_function;  // e.g. "dronet.main_graph$async"
  const char* mlp_function;     // e.g. "mlp.main_graph$async"
  const char* driver_name;      // e.g. "local-task"

  // Target periodic invocation frequency for MLP (scheduler release rate).
  // Must be > 0.
  double mlp_frequency_hz;

  // Frequency of synthetic input production for Dronet and MLP.
  // These drive the input-ready timelines that invocations wait on.
  // Must be > 0.
  double dronet_sensor_frequency_hz;
  double mlp_sensor_frequency_hz;

  // Stats report frequency to stdout. Must be > 0.
  double report_frequency_hz;

  // Run duration in milliseconds. If <= 0, runs until externally terminated.
  int64_t run_duration_ms;
} dual_model_async_config_t;

int dual_model_async_scheduler_run(const dual_model_async_config_t* config);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // DUAL_MODEL_ASYNC_SCHEDULER_H_