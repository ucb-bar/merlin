#ifndef SMOL_VISION_OPS_H
#define SMOL_VISION_OPS_H
#include <stddef.h>
#include <stdint.h>

/* Vision-native BF16 GELU via a pinned complete finite-BF16 function table. */
int smol_vision_gelu_bf16(const uint16_t *input, const uint16_t *table,
                         uint16_t *output, size_t count);

/* Fixed vision width768, native CPU8 Welford order, explicit fused operations. */
int smol_vision_layernorm_bf16(const uint16_t *input, const uint16_t *weight,
                              const uint16_t *bias, float epsilon,
                              uint16_t *output, size_t rows);
#endif
