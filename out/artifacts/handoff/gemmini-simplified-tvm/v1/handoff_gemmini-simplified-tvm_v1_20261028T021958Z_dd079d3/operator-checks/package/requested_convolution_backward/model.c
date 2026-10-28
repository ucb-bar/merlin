#include <stdint.h>
#include <stddef.h>
#ifdef TVM_MODEL_CONSTANTS_ADDRESS
#define model_constants ((const uint8_t *)(uintptr_t)(TVM_MODEL_CONSTANTS_ADDRESS))
#else
extern const uint8_t model_constants[];
#endif
extern const char *model_last_error;
const char *stage_4_model_get_last_error(void) { return model_last_error; }
extern int32_t stage_4_operator_conv2d_transpose(float*, float*, float*);
extern int32_t stage_4_operator_conv2d1(float*, float*, float*);
extern int32_t stage_4_operator_sum(float*, float*);
static int model_span(const void *pointer, uint64_t bytes) { return pointer && !((uintptr_t)pointer % 64) && bytes <= UINTPTR_MAX - (uintptr_t)pointer; }
static int model_overlap(const void *a, uint64_t an, const void *b, uint64_t bn) { return an && bn && (uintptr_t)a < (uintptr_t)b + bn && (uintptr_t)b < (uintptr_t)a + an; }
int32_t stage_4_model_run(const void *const *inputs, void *const *outputs, uint8_t *workspace, uint64_t workspace_bytes) {
  model_last_error = 0;
  if (!outputs || !inputs) return -1;
  if (workspace_bytes < UINT64_C(0)) return -1;
  if (!model_span(inputs[0], 48)) return -1;
  if (!model_span(inputs[1], 48)) return -1;
  if (!model_span(inputs[2], 32)) return -1;
  if (!model_span(outputs[0], 48)) return -1;
  if (!model_span(outputs[1], 32)) return -1;
  if (!model_span(outputs[2], 8)) return -1;
  if (model_overlap(outputs[0], 48, inputs[0], 48)) return -2;
  if (model_overlap(outputs[0], 48, inputs[1], 48)) return -2;
  if (model_overlap(outputs[0], 48, inputs[2], 32)) return -2;
  if (model_overlap(outputs[0], 48, model_constants, 576)) return -2;
  if (model_overlap(outputs[1], 32, inputs[0], 48)) return -2;
  if (model_overlap(outputs[1], 32, inputs[1], 48)) return -2;
  if (model_overlap(outputs[1], 32, inputs[2], 32)) return -2;
  if (model_overlap(outputs[1], 32, model_constants, 576)) return -2;
  if (model_overlap(outputs[1], 32, outputs[0], 48)) return -2;
  if (model_overlap(outputs[2], 8, inputs[0], 48)) return -2;
  if (model_overlap(outputs[2], 8, inputs[1], 48)) return -2;
  if (model_overlap(outputs[2], 8, inputs[2], 32)) return -2;
  if (model_overlap(outputs[2], 8, model_constants, 576)) return -2;
  if (model_overlap(outputs[2], 8, outputs[0], 48)) return -2;
  if (model_overlap(outputs[2], 8, outputs[1], 32)) return -2;
  if (stage_4_operator_conv2d_transpose((float*)inputs[0], (float*)inputs[2], (float*)outputs[0])) return -3;
  if (stage_4_operator_conv2d1((float*)inputs[1], (float*)inputs[0], (float*)outputs[1])) return -3;
  if (stage_4_operator_sum((float*)inputs[0], (float*)outputs[2])) return -3;
  return 0;
}
