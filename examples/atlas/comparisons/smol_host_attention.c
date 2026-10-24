#include "smol_host_attention.h"
#include "smol_host_bits.h"
#include "smol_host_softmax.h"

#include <fenv.h>
#include <float.h>
#include <math.h>

int smol_host_attention_cpu16_bf16(
    const uint16_t *query, const uint16_t *key, const uint16_t *value,
    const uint8_t *mask, size_t rows, size_t keys, size_t heads,
    size_t kv_heads, size_t dimension, float *workspace,
    float *scores, uint16_t *probabilities, uint16_t *output) {
    if (!query || !key || !value || !mask || !workspace || !scores
        || !probabilities || !output || !rows || !keys || !heads
        || !kv_heads || !dimension || heads % kv_heads
        || heads > SIZE_MAX / dimension || kv_heads > SIZE_MAX / dimension
        || rows > SIZE_MAX / sizeof(uint16_t) / (heads * dimension)
        || keys > SIZE_MAX / sizeof(uint16_t) / (kv_heads * dimension)
        || heads > SIZE_MAX / rows
        || heads * rows > SIZE_MAX / sizeof(float) / keys
        || keys > SIZE_MAX / sizeof(float)
        || FLT_RADIX != 2 || FLT_MANT_DIG != 24 || sizeof(float) != 4
        || fegetround() != FE_TONEAREST) return -1;
    for (size_t i = 0; i < rows * heads * dimension; ++i) {
        if (!isfinite(bf16_to_float(query[i]))) return -1;
    }
    for (size_t i = 0; i < keys * kv_heads * dimension; ++i) {
        if (!isfinite(bf16_to_float(key[i])) || !isfinite(bf16_to_float(value[i]))) return -1;
    }
    for (size_t i = 0; i < rows * keys; ++i) {
        if (mask[i] > 1) return -1;
    }
    float scale = (float)pow((double)dimension, -0.5);
    size_t groups = heads / kv_heads;
    for (size_t head = 0; head < heads; ++head) {
        size_t kv_head = head / groups;
        for (size_t row = 0; row < rows; ++row) {
            size_t base = (head * rows + row) * keys;
            for (size_t column = 0; column < keys; ++column) {
                float sum = 0.0f;
                for (size_t d = 0; d < dimension; ++d) {
                    float product = bf16_to_float(query[(row * heads + head) * dimension + d])
                                    * bf16_to_float(key[(column * kv_heads + kv_head) * dimension + d]);
                    sum += product;
                }
                float score = mask[row * keys + column] ? sum * scale : -FLT_MAX;
                if (!isfinite(score)) return -1;
                scores[base + column] = score;
            }
        }
    }
    if (smol_host_softmax_cpu16_bf16(scores, heads * rows, keys,
                                   workspace, probabilities)) return -1;
    for (size_t head = 0; head < heads; ++head) {
        size_t kv_head = head / groups;
        for (size_t row = 0; row < rows; ++row) {
            size_t base = (head * rows + row) * keys;
            for (size_t d = 0; d < dimension; ++d) {
                float sum = 0.0f;
                /* Source contract: oneDNN 8d263e693366ef8db40acc569cc7d8edf644556d,
                 * bf16_emulation_t::vdpbf16ps, jit_avx512_core_bf16cvt.hpp.
                 * Each packed pair's upper BF16 member precedes its lower.
                 */
                for (size_t column = 0; column < keys; column += 2) {
                    if (column + 1 < keys) {
                        sum = fmaf(bf16_to_float(probabilities[base + column + 1]),
                                   bf16_to_float(value[((column + 1) * kv_heads + kv_head) * dimension + d]), sum);
                    }
                    sum = fmaf(bf16_to_float(probabilities[base + column]),
                               bf16_to_float(value[(column * kv_heads + kv_head) * dimension + d]), sum);
                }
                output[(row * heads + head) * dimension + d] = float_to_bf16(sum);
            }
        }
    }
    return 0;
}
