#include <stdint.h>
#include <stddef.h>
#ifdef TVM_MODEL_CONSTANTS_ADDRESS
#define model_constants ((const uint8_t *)(uintptr_t)(TVM_MODEL_CONSTANTS_ADDRESS))
#else
extern const uint8_t model_constants[];
#endif
extern const char *model_last_error;
const char *stage_54_model_get_last_error(void) { return model_last_error; }
extern int32_t stage_54_operator_kernel(int8_t*, int8_t*, int8_t*);
static int model_span(const void *pointer, uint64_t bytes) { return pointer && !((uintptr_t)pointer % 64) && bytes <= UINTPTR_MAX - (uintptr_t)pointer; }
static int model_overlap(const void *a, uint64_t an, const void *b, uint64_t bn) { return an && bn && (uintptr_t)a < (uintptr_t)b + bn && (uintptr_t)b < (uintptr_t)a + an; }
int32_t stage_54_model_run(const void *const *inputs, void *const *outputs, uint8_t *workspace, uint64_t workspace_bytes) {
  model_last_error = 0;
  if (!outputs || !inputs) return -1;
  if (workspace_bytes < UINT64_C(0)) return -1;
  if (!model_span(inputs[0], 323)) return -1;
  if (!model_span(inputs[1], 323)) return -1;
  if (!model_span(outputs[0], 323)) return -1;
  if (model_overlap(outputs[0], 323, inputs[0], 323)) return -2;
  if (model_overlap(outputs[0], 323, inputs[1], 323)) return -2;
  if (model_overlap(outputs[0], 323, model_constants, 576)) return -2;
  if (stage_54_operator_kernel((int8_t*)inputs[0], (int8_t*)inputs[1], (int8_t*)outputs[0])) return -3;
  return 0;
}
