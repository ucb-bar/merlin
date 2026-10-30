#ifndef SMOL_HOST_OPS_H
#define SMOL_HOST_OPS_H

#include <stddef.h>
#include <stdint.h>

/* Source-native arithmetic primitives. Callers retain tensor/layout ownership. */
int smol_host_euler_f32(const float *state, const float *velocity, float dt,
                        float *output, size_t count);
int smol_host_residual_bf16(const uint16_t *left, const uint16_t *right,
                           uint16_t *output, size_t count);
#ifndef SMOL_HOST_EXCLUDE_MIXED_RESIDUAL
int smol_host_residual_bf16_f32(const uint16_t *left, const float *right,
                               uint16_t *output, size_t count);
#endif
int smol_host_silu_mul_bf16(const uint16_t *gate, const uint16_t *up,
                           uint16_t *output, size_t count);
int smol_host_rmsnorm_bf16(const uint16_t *input, const uint16_t *weight,
                          float epsilon, uint16_t *output, size_t rows, size_t width);
int smol_host_rmsnorm_f32_to_bf16(const float *input, const uint16_t *weight,
                                 float epsilon, uint16_t *output, size_t rows, size_t width);

#endif
