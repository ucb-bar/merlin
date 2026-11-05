#ifndef SMOL_POLICY_OPS_H
#define SMOL_POLICY_OPS_H
#include <stddef.h>
#include <stdint.h>
int smol_policy_silu_f32(const float *input, float *output, size_t count);
/* Odd expert layers: BF16 query and exact BF16 views of FP32 K/V replies.
 * Scores/probabilities and PV accumulation remain FP32 until the O bridge.
 * Packed query/output[50,15,64], K/V[177,5,64], Boolean mask[50,177].
 * Caller supplies disjoint score/probability arrays[15,50,177] and workspace[177]. */
int smol_policy_cross_attention_bf16(const uint16_t *query, const uint16_t *key,
                                    const uint16_t *value, const uint8_t *mask,
                                    float *scores, float *probabilities,
                                    float *workspace, uint16_t *output);
#endif
