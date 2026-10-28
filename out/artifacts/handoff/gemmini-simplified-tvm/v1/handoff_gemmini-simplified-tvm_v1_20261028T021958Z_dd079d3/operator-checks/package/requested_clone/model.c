#include <stdint.h>
#include <stddef.h>
#ifdef TVM_MODEL_CONSTANTS_ADDRESS
#define model_constants ((const uint8_t *)(uintptr_t)(TVM_MODEL_CONSTANTS_ADDRESS))
#else
extern const uint8_t model_constants[];
#endif
extern const char *model_last_error;
const char *stage_23_model_get_last_error(void) { return model_last_error; }
static int model_span(const void *pointer, uint64_t bytes) { return pointer && !((uintptr_t)pointer % 64) && bytes <= UINTPTR_MAX - (uintptr_t)pointer; }
static int model_overlap(const void *a, uint64_t an, const void *b, uint64_t bn) { return an && bn && (uintptr_t)a < (uintptr_t)b + bn && (uintptr_t)b < (uintptr_t)a + an; }
int32_t stage_23_model_run(const void *const *inputs, void *const *outputs, uint8_t *workspace, uint64_t workspace_bytes) {
  model_last_error = 0;
  if (!outputs || !inputs) return -1;
  if (workspace_bytes < UINT64_C(0)) return -1;
  if (!model_span(inputs[0], 6)) return -1;
  if (!model_span(inputs[1], 6)) return -1;
  if (!model_span(inputs[2], 3)) return -1;
  if (!model_span(outputs[0], 6)) return -1;
  if (model_overlap(outputs[0], 6, inputs[0], 6)) return -2;
  if (model_overlap(outputs[0], 6, inputs[1], 6)) return -2;
  if (model_overlap(outputs[0], 6, inputs[2], 3)) return -2;
  if (model_overlap(outputs[0], 6, model_constants, 576)) return -2;
  for (uint64_t i = 0; i < UINT64_C(6); ++i) ((uint8_t*)outputs[0])[i] = ((const uint8_t*)inputs[0])[i];
  return 0;
}
