#include "smol_vision_attention.h"
#include "smol_host_bits.h"

#include <fenv.h>
#include <float.h>
#include <math.h>

/* Negative-domain expf adaptation of Arm optimized-routines v25.01,
 * math/expf.c and math/exp2f_data.c, Copyright (c) 2017-2024 Arm Limited.
 * The installed native GLIBC_2.27 expf contracts range reduction and the
 * polynomial into the explicit double FMAs below. This helper needs neither
 * arbitrary-input guest expf nor a learned-input lookup table.
 *
 * Copyright (c) 1999-2022, Arm Limited.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
static double vision_fma64(double left, double right, double addend) {
#if defined(__riscv)
    double result;
    __asm__("fmadd.d %0,%1,%2,%3" : "=f"(result)
            : "f"(left), "f"(right), "f"(addend));
    return result;
#else
    return fma(left, right, addend);
#endif
}

static float vision_prior_exp(float value) {
    /* Running maxima only request nonpositive inputs, including -infinity. */
    if (value < -0x1.9fe368p6f) return 0.0f;
    static const uint64_t table[32] = {
    UINT64_C(0x3ff0000000000000), UINT64_C(0x3fefd9b0d3158574), UINT64_C(0x3fefb5586cf9890f), UINT64_C(0x3fef9301d0125b51),
    UINT64_C(0x3fef72b83c7d517b), UINT64_C(0x3fef54873168b9aa), UINT64_C(0x3fef387a6e756238), UINT64_C(0x3fef1e9df51fdee1),
    UINT64_C(0x3fef06fe0a31b715), UINT64_C(0x3feef1a7373aa9cb), UINT64_C(0x3feedea64c123422), UINT64_C(0x3feece086061892d),
    UINT64_C(0x3feebfdad5362a27), UINT64_C(0x3feeb42b569d4f82), UINT64_C(0x3feeab07dd485429), UINT64_C(0x3feea47eb03a5585),
    UINT64_C(0x3feea09e667f3bcd), UINT64_C(0x3fee9f75e8ec5f74), UINT64_C(0x3feea11473eb0187), UINT64_C(0x3feea589994cce13),
    UINT64_C(0x3feeace5422aa0db), UINT64_C(0x3feeb737b0cdc5e5), UINT64_C(0x3feec49182a3f090), UINT64_C(0x3feed503b23e255d),
    UINT64_C(0x3feee89f995ad3ad), UINT64_C(0x3feeff76f2fb5e47), UINT64_C(0x3fef199bdd85529c), UINT64_C(0x3fef3720dcef9069),
    UINT64_C(0x3fef5818dcfba487), UINT64_C(0x3fef7c97337b9b5f), UINT64_C(0x3fefa4afa2a490da), UINT64_C(0x3fefd0765b6e4540),
    };
    const double inverse_log = 0x1.71547652b82fep5;
    const double shift = 0x1.8p52;
    double rounded = vision_fma64(inverse_log, (double)value, shift);
    uint64_t bits;
    memcpy(&bits, &rounded, sizeof(bits));
    rounded -= shift;
    double remainder = vision_fma64(inverse_log, (double)value, -rounded);
    uint64_t scale_bits = table[bits % 32] + (bits << 47);
    double scale;
    memcpy(&scale, &scale_bits, sizeof(scale));
    double polynomial = vision_fma64(0x1.c6af84b912394p-20, remainder,
                                    0x1.ebfce50fac4f3p-13);
    double square = remainder * remainder;
    double result = vision_fma64(0x1.62e42ff0c52d6p-6, remainder, 1.0);
    result = vision_fma64(polynomial, square, result);
    return (float)(result * scale);
}

/* Arithmetic contract: PyTorch2.10 CPU FlashAttentionKernel, BF16 CPU16.
 * Retain512-key running-max blocks, FLOAT lane sums, BF16 unnormalized
 * exponentials, and final FLOAT normalization. This is not eager attention.
 */
static float vision_exp(float value) {
    if (value < -87.3365478515625f) return 0.0f;
    /* log2(e) requires all23 fraction bits, unlike BF16 operands. */
    uint32_t log_bits = UINT32_C(0x3fb8aa3b);
    float log2e;
    memcpy(&log2e, &log_bits, sizeof(log2e));
    float source = value * log2e;
    int whole = (int)source;
    float floor_value = source == 0.0f ? source : (float)(whole - (source < (float)whole));
    float fraction = source - floor_value;
    float correction = fmaf(fraction, -0.079204240219773236f, -0.22433836478672356f);
    correction = fmaf(fraction, correction, 0.30354260500649682f);
    correction = fmaf(fraction, correction, 0.00010703434948458272f);
    source -= correction;
    float encoded = fmaf(8388608.0f, source, 1065353216.0f);
    uint32_t bits = (uint32_t)(int32_t)encoded;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static int overlap(const uint16_t *left, size_t left_words,
                   const uint16_t *right, size_t right_words) {
    uintptr_t a = (uintptr_t)left, b = (uintptr_t)right;
    return a < b + right_words * sizeof(uint16_t)
        && b < a + left_words * sizeof(uint16_t);
}

int smol_vision_attention_cpu16_bf16(
    const uint16_t *query, const uint16_t *key, const uint16_t *value,
    size_t rows, size_t keys, size_t heads, uint16_t *output) {
    if (!query || !key || !value || !output || !rows || rows > 1024
        || !keys || keys > 1024 || keys % 16 || !heads || heads > 12
        || FLT_RADIX != 2 || FLT_MANT_DIG != 24 || sizeof(float) != 4
        || DBL_MANT_DIG != 53 || sizeof(double) != 8
        || fegetround() != FE_TONEAREST) return -1;
    size_t width = heads * 64, q_words = rows * width, kv_words = keys * width;
    if (overlap(output, q_words, query, q_words)
        || overlap(output, q_words, key, kv_words)
        || overlap(output, q_words, value, kv_words)) return -1;
    for (size_t i = 0; i < q_words; ++i) {
        if ((query[i] & UINT16_C(0x7f80)) == UINT16_C(0x7f80)) return -1;
    }
    for (size_t i = 0; i < kv_words; ++i) {
        if ((key[i] & UINT16_C(0x7f80)) == UINT16_C(0x7f80)
            || (value[i] & UINT16_C(0x7f80)) == UINT16_C(0x7f80)) return -1;
    }
    for (size_t head = 0; head < heads; ++head) {
        for (size_t row = 0; row < rows; ++row) {
            float maximum = bf16_to_float(UINT16_C(0xff80)), denominator = 0.0f, accumulator[64] = {0};
            for (size_t block = 0; block < keys; block += 512) {
                size_t count = keys - block < 512 ? keys - block : 512;
                float scores[512], lane_sums[16] = {0};
                uint16_t exponentials[512];
                float next_maximum = maximum;
                for (size_t column = 0; column < count; ++column) {
                    float dot = 0.0f;
                    size_t q_offset = row * width + head * 64;
                    size_t k_offset = (block + column) * width + head * 64;
                    /* Native BF16->FP32 MKL QK uses ascending FMA order. */
                    for (size_t d = 0; d < 64; ++d) {
                        dot = fmaf(bf16_to_float(query[q_offset + d]),
                                   bf16_to_float(key[k_offset + d]), dot);
                    }
                    scores[column] = dot * 0.125f + 0.0f;
                    if (!isfinite(scores[column])) return -1;
                    if (scores[column] > next_maximum) next_maximum = scores[column];
                }
                for (size_t column = 0; column < count; ++column) {
                    float exponential = vision_exp(scores[column] - next_maximum);
                    lane_sums[column % 16] += exponential;
                    exponentials[column] = float_to_bf16(exponential);
                }
                for (size_t stride = 8; stride; stride /= 2) {
                    for (size_t lane = 0; lane < stride; ++lane) {
                        lane_sums[lane] += lane_sums[lane + stride];
                    }
                }
                float prior_scale = vision_prior_exp(maximum - next_maximum);
                /* Native AVX512 contracts the running-sum update into one FMA. */
                denominator = fmaf(prior_scale, denominator, lane_sums[0]);
                maximum = next_maximum;
                for (size_t d = 0; d < 64; ++d) {
                    float current = block ? accumulator[d] * prior_scale : 0.0f;
                    /* The native float-output GEMM splits K into256 terms.
                     * Each ascending dot starts at zero, then adds beta*C.
                     */
                    for (size_t part = 0; part < count; part += 256) {
                        size_t end = count - part < 256 ? count : part + 256;
                        float sum = 0.0f;
                        for (size_t column = part; column < end; ++column) {
                            sum = fmaf(bf16_to_float(exponentials[column]),
                                       bf16_to_float(value[(block + column) * width + head * 64 + d]), sum);
                        }
                        current = block || part ? sum + current : sum;
                    }
                    accumulator[d] = current;
                }
            }
            float reciprocal = 1.0f / denominator;
            if (!isfinite(reciprocal)) return -1;
            for (size_t d = 0; d < 64; ++d) {
                if (!isfinite(accumulator[d])) return -1;
                output[row * width + head * 64 + d] = float_to_bf16(accumulator[d] * reciprocal);
            }
        }
    }
    return 0;
}
