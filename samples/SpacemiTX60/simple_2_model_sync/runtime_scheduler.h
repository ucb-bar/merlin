#ifndef MERLIN_SAMPLES_ASYNC_SMOKE_RUNTIME_SCHEDULER_H_
#define MERLIN_SAMPLES_ASYNC_SMOKE_RUNTIME_SCHEDULER_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct merlin_async_smoke_config_t {
  const char* dronet_vmfb_path;
  const char* mlp_vmfb_path;

  // Fully qualified, including module name.
  // Example:
  //   "dronet.main_graph$async"
  //   "mlp.main_graph$async"
  const char* dronet_function;
  const char* mlp_function;

  // Example: "local-task"
  const char* driver_name;
} merlin_async_smoke_config_t;

// Returns 0 on success, non-zero on error.
int merlin_async_smoke_run(const merlin_async_smoke_config_t* config);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // MERLIN_SAMPLES_ASYNC_SMOKE_RUNTIME_SCHEDULER_H_