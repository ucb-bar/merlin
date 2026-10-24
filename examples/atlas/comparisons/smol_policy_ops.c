/* Scalar SLEEF xexpf adaptation for native FP32 policy SiLU. Constants and
 * explicit FMA order match the existing smol_host_softmax.c adaptation of
 * SLEEF revision5a1d179df9cf652951b59010a2d2075372d67f68.
 * Copyright Naoki Shibata and contributors2010-2024. Boost Software License1.0,
 * provided in smol_host_sleef_LICENSE.txt. Compile without fast-math or implicit
 * contraction. This API preserves FP32 output, unlike the BF16 MLP helper.
 */
#include "smol_policy_ops.h"
#include "smol_host_bits.h"
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

static float bits_float(uint32_t bits) {
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static int nearest_integer(float value) {
    int whole = (int) value;
    float remainder = value - (float) whole;
    if (remainder > 0.5f || (remainder == 0.5f && whole % 2)) ++whole;
    if (remainder < -0.5f || (remainder == -0.5f && whole % 2)) --whole;
    return whole;
}

static float native_exp(float d) {
    if (d < -104.0f) return 0;
    if (d > 100.0f) return bits_float(UINT32_C(0x7f800000));
    int q = nearest_integer(d * 1.442695040888963407359924681001892137426645954152985934135449406931f);
    float s = fmaf((float) q, -0.693145751953125f, d);
    s = fmaf((float) q, -1.428606765330187045e-06f, s);
    float u = 0.000198527617612853646278381f;
    u = fmaf(u, s, 0.00139304355252534151077271f);
    u = fmaf(u, s, 0.00833336077630519866943359f);
    u = fmaf(u, s, 0.0416664853692054748535156f);
    u = fmaf(u, s, 0.166666671633720397949219f);
    u = fmaf(u, s, 0.5f);
    u = 1.0f + fmaf(s * s, u, s);
    int half = q >= 0 ? q / 2 : -((-q + 1) / 2);
    float first = bits_float((uint32_t) (127 + half) << 23);
    float second = bits_float((uint32_t) (127 + q - half) << 23);
    return (u * first) * second;
}

int smol_policy_silu_f32(const float *input, float *output, size_t count) {
    if (!input || !output || !count) return 1;
    for (size_t i = 0; i < count; ++i) if (!isfinite(input[i])) return 2;
    for (size_t i = 0; i < count; ++i)
        output[i] = input[i] / (1.0f + native_exp(-input[i]));
    return 0;
}

static float native_sum16(const float *values) {
    float lanes[16] = {0};
    for (size_t i = 0; i < 177; ++i) lanes[i % 16] += values[i];
    for (size_t delta = 8; delta; delta /= 2)
        for (size_t lane = 0; lane < delta; ++lane) lanes[lane] += lanes[lane + delta];
    return lanes[0];
}

int smol_policy_cross_attention_bf16(const uint16_t *query, const uint16_t *key,
                                    const uint16_t *value, const uint8_t *mask,
                                    float *scores, float *probabilities,
                                    float *workspace, uint16_t *output) {
    if (!query || !key || !value || !mask || !scores || !probabilities || !workspace || !output) return 1;
    for (size_t i = 0; i < 50 * 15 * 64; ++i) if (!isfinite(bf16_to_float(query[i]))) return 2;
    for (size_t i = 0; i < 177 * 5 * 64; ++i)
        if (!isfinite(bf16_to_float(key[i])) || !isfinite(bf16_to_float(value[i]))) return 2;
    for (size_t i = 0; i < 50 * 177; ++i) if (mask[i] > 1) return 2;
    for (size_t head = 0; head < 15; ++head) {
        size_t kv = head / 3;
        for (size_t row = 0; row < 50; ++row) {
            size_t base = (head * 50 + row) * 177;
            for (size_t column = 0; column < 177; ++column) {
                float sum = 0;
                for (size_t d = 0; d < 64; ++d) {
                    float product = bf16_to_float(query[(row * 15 + head) * 64 + d])
                                  * bf16_to_float(key[(column * 5 + kv) * 64 + d]);
                    sum += product;
                }
                scores[base + column] = mask[row * 177 + column] ? sum * 0.125f : -FLT_MAX;
                if (!isfinite(scores[base + column])) return 3;
            }
            float maximum = scores[base];
            for (size_t column = 1; column < 177; ++column)
                if (scores[base + column] > maximum) maximum = scores[base + column];
            for (size_t column = 0; column < 177; ++column)
                workspace[column] = native_exp(scores[base + column] - maximum);
            float reciprocal = 1.0f / native_sum16(workspace);
            for (size_t column = 0; column < 177; ++column)
                probabilities[base + column] = workspace[column] * reciprocal;
            for (size_t d = 0; d < 64; ++d) {
                float sum = 0;
                for (size_t column = 0; column < 177; ++column)
                    sum = fmaf(probabilities[base + column], bf16_to_float(value[(column * 5 + kv) * 64 + d]), sum);
                output[(row * 15 + head) * 64 + d] = float_to_bf16(sum);
            }
        }
    }
    return 0;
}
