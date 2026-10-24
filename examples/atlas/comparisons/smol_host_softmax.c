#include "smol_host_softmax.h"
#include "smol_host_bits.h"

#include <fenv.h>
#include <float.h>
#include <math.h>

/* Scalar adaptation of SLEEF xexpf (sleefsimdsp.c), revision
 * 5a1d179df9cf652951b59010a2d2075372d67f68, as pinned by PyTorch
 * 449b1768410104d3ed79d3bcfe4ba1d65c7f22c0. Constants and explicit FMA order
 * follow the native AVX512 implementation. Compile with -ffp-contract=off
 * and without fast-math. Copyright Naoki Shibata and contributors 2010-2024.
 * Distributed under the Boost Software License, Version 1.0; see
 * smol_host_sleef_LICENSE.txt. Domain here is d <= 0 (negative infinity from
 * finite-score subtraction overflow is mapped to zero before conversion).
 */
static float native_exp_nonpositive(float d) {
    if (d < -104.0f) return 0.0f;
    int q = (int)nearbyintf(d * 1.442695040888963407359924681001892137426645954152985934135449406931f);
    float s = fmaf((float)q, -0.693145751953125f, d);
    s = fmaf((float)q, -1.428606765330187045e-06f, s);
    float u = 0.000198527617612853646278381f;
    u = fmaf(u, s, 0.00139304355252534151077271f);
    u = fmaf(u, s, 0.00833336077630519866943359f);
    u = fmaf(u, s, 0.0416664853692054748535156f);
    u = fmaf(u, s, 0.166666671633720397949219f);
    u = fmaf(u, s, 0.5f);
    u = 1.0f + fmaf(s * s, u, s);
    int half = q >= 0 ? q / 2 : -((-q + 1) / 2);
    return scalbnf(scalbnf(u, half), q - half);
}

/* PyTorch functional_base.h reduce_all: sequential lane accumulation,
 * partial final-vector update, then AVX512 xor8/xor4/xor2/xor1 reduction.
 * For fewer than sixteen elements its scalar path reduces left to right.
 */
static float native_sum16(const float *values, size_t count) {
    if (count < 16) {
        float result = values[0];
        for (size_t i = 1; i < count; ++i) result += values[i];
        return result;
    }
    float lanes[16] = {0};
    for (size_t i = 0; i < count; ++i) lanes[i % 16] += values[i];
    for (size_t delta = 8; delta; delta /= 2) {
        for (size_t lane = 0; lane < delta; ++lane) lanes[lane] += lanes[lane + delta];
    }
    return lanes[0];
}

int smol_host_softmax_cpu16_bf16(const float *scores, size_t rows, size_t keys,
                               float *workspace, uint16_t *output) {
    if (!scores || !workspace || !output || !rows || !keys
        || keys > SIZE_MAX / sizeof(float)
        || rows > SIZE_MAX / sizeof(float) / keys
        || FLT_RADIX != 2 || FLT_MANT_DIG != 24 || sizeof(float) != 4
        || fegetround() != FE_TONEAREST) return -1;
    for (size_t i = 0; i < rows * keys; ++i) {
        if (!isfinite(scores[i])) return -1;
    }
    for (size_t row = 0; row < rows; ++row) {
        size_t base = row * keys;
        float maximum = scores[base];
        for (size_t key = 1; key < keys; ++key) {
            if (scores[base + key] > maximum) maximum = scores[base + key];
        }
        for (size_t key = 0; key < keys; ++key) {
            workspace[key] = native_exp_nonpositive(scores[base + key] - maximum);
        }
        float reciprocal = 1.0f / native_sum16(workspace, keys);
        for (size_t key = 0; key < keys; ++key) {
            output[base + key] = float_to_bf16(workspace[key] * reciprocal);
        }
    }
    return 0;
}
