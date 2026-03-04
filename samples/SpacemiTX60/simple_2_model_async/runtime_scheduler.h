#ifndef MERLIN_SAMPLES_DUAL_MODEL_ASYNC_GATE_H_
#define MERLIN_SAMPLES_DUAL_MODEL_ASYNC_GATE_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct merlin_dual_model_async_gate_config_t {
  const char* dronet_vmfb_path;
  const char* mlp_vmfb_path;

  // Fully qualified exports, e.g. "dronet.main_graph$async"
  const char* dronet_function;
  const char* mlp_function;

  // e.g. "local-task"
  const char* driver_name;

  // If non-zero, print first bytes of outputs (debug).
  int dump_output_bytes;
} merlin_dual_model_async_gate_config_t;

int merlin_dual_model_async_gate_run(
    const merlin_dual_model_async_gate_config_t* config);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // MERLIN_SAMPLES_DUAL_MODEL_ASYNC_GATE_H_