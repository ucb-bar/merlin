/* Explicit host arithmetic policy for independently gated BF16 attention
 * boundaries. The pair order follows pinned oneDNN BF16 emulation sources;
 * this source has no device, workload-output or compiler-pass selection. */
#include "llama_host_bmm.h"
#include "smol_host_bits.h"

#include <fenv.h>
#include <math.h>
#include <string.h>

static int shape(size_t batches, size_t rows, size_t columns) {
    return batches && rows && columns && rows <= SIZE_MAX / batches
        && columns <= SIZE_MAX / (batches * rows)
        && batches * rows * columns <= SIZE_MAX / sizeof(uint16_t);
}

static int finite_words(const uint16_t *words, size_t count) {
    for (size_t i = 0; i < count; ++i)
        if ((words[i] & UINT16_C(0x7f80)) == UINT16_C(0x7f80)) return 0;
    return 1;
}

/* The pinned CPU has no AVX512 BF16 dot instruction. CPUBlas routes the
 * transposed, single-row decode QK to bf16_dot_with_fp32_arith instead of
 * oneDNN GEMM. Its K64 AVX512 path performs four consecutive 16-lane FMA
 * updates, followed by XOR-distance 8,4,2,1 FP32 horizontal additions.
 * This is a portable arithmetic policy, independent of the target ISA. */
static float decode_qk_dot64(const uint16_t *left, const uint16_t *right) {
    float lanes[16] = {0.0f}, next[16];
    for (size_t start = 0; start < 64; start += 16)
        for (size_t lane = 0; lane < 16; ++lane)
            lanes[lane] = fmaf(bf16_to_float(left[start + lane]),
                               bf16_to_float(right[start + lane]), lanes[lane]);
    for (size_t distance = 8; distance; distance /= 2) {
        for (size_t lane = 0; lane < 16; ++lane)
            next[lane] = lanes[lane] + lanes[lane ^ distance];
        memcpy(lanes, next, sizeof(lanes));
    }
    return lanes[0];
}

/* oneDNN's nontransposed BF16 GEMV accumulates at most eight reduction
 * members locally, then adds that block to the prior FP32 output. Its
 * remaining blocks have lengths4,2,1, each with the same upper/lower pairs. */
static float decode_pv_dot(const uint16_t *left, const uint16_t *right,
                           size_t reduction, size_t stride) {
    float total = 0.0f;
    size_t start = 0;
    for (size_t block = 8; block; block /= 2) {
        while (reduction - start >= block) {
            float partial = 0.0f;
            for (size_t pair = start; pair < start + block; pair += 2) {
                if (pair + 1 < start + block)
                    partial = fmaf(bf16_to_float(left[pair + 1]),
                                   bf16_to_float(right[(pair + 1) * stride]), partial);
                partial = fmaf(bf16_to_float(left[pair]),
                               bf16_to_float(right[pair * stride]), partial);
            }
            total += partial;
            start += block;
        }
    }
    return total;
}

int llama_host_bmm_bf16_pair(const uint16_t *left, const uint16_t *right,
                            uint16_t *output, size_t batches, size_t rows,
                            size_t reduction, size_t columns, int right_transposed) {
    if (!left || !right || !output || output == left || output == right
        || (right_transposed != 0 && right_transposed != 1)
        || !shape(batches, rows, reduction) || !shape(batches, reduction, columns)
        || !shape(batches, rows, columns) || fegetround() != FE_TONEAREST) return -1;
    if (!finite_words(left, batches * rows * reduction)
        || !finite_words(right, batches * reduction * columns)) return -1;
    for (size_t batch = 0; batch < batches; ++batch) {
        for (size_t row = 0; row < rows; ++row) {
            for (size_t column = 0; column < columns; ++column) {
                float total = 0.0f;
                if (rows == 1 && right_transposed && reduction == 64) {
                    total = decode_qk_dot64(left + batch * reduction,
                        right + batch * reduction * columns + column * reduction);
                } else if (rows == 1 && !right_transposed && columns == 64) {
                    total = decode_pv_dot(left + batch * reduction,
                        right + batch * reduction * columns + column, reduction, columns);
                } else for (size_t pair = 0; pair < reduction; pair += 2) {
                    if (pair + 1 < reduction) {
                        size_t r = right_transposed ? column * reduction + pair + 1
                                                    : (pair + 1) * columns + column;
                        total = fmaf(bf16_to_float(left[(batch * rows + row) * reduction + pair + 1]),
                                     bf16_to_float(right[batch * reduction * columns + r]), total);
                    }
                    size_t r = right_transposed ? column * reduction + pair
                                                : pair * columns + column;
                    total = fmaf(bf16_to_float(left[(batch * rows + row) * reduction + pair]),
                                 bf16_to_float(right[batch * reduction * columns + r]), total);
                }
                output[(batch * rows + row) * columns + column] = float_to_bf16(total);
            }
        }
    }
    return 0;
}

int llama_host_scale_mask_bf16(const uint16_t *scores, const uint16_t *mask,
                               float scale, uint16_t *output, size_t batches,
                               size_t rows, size_t columns) {
    if (!scores || !mask || !output || scores == output || mask == output
        || !shape(batches, rows, columns) || !isfinite(scale)
        || fegetround() != FE_TONEAREST) return -1;
    if (!finite_words(scores, batches * rows * columns)
        || !finite_words(mask, rows * columns)) return -1;
    for (size_t batch = 0; batch < batches; ++batch) {
        for (size_t row = 0; row < rows; ++row) {
            for (size_t column = 0; column < columns; ++column) {
                size_t offset = (batch * rows + row) * columns + column;
                uint16_t scaled = float_to_bf16(bf16_to_float(scores[offset]) * scale);
                output[offset] = float_to_bf16(bf16_to_float(scaled)
                                               + bf16_to_float(mask[row * columns + column]));
            }
        }
    }
    return 0;
}

int llama_host_merge_heads_bf16(const uint16_t *input, uint16_t *output,
                                size_t rows, size_t heads, size_t width) {
    if (!input || !output || input == output || !shape(heads, rows, width)) return -1;
    for (size_t row = 0; row < rows; ++row)
        for (size_t head = 0; head < heads; ++head)
            memcpy(output + (row * heads + head) * width,
                   input + (head * rows + row) * width, width * sizeof(uint16_t));
    return 0;
}

int llama_host_causal_mask_bf16(const int64_t *positions, uint16_t *output,
                                size_t rows, size_t keys) {
    if (!positions || !output || (const void *)positions == (const void *)output
        || !shape(1, rows, keys) || rows > SIZE_MAX / sizeof(int64_t)) return -1;
    for (size_t row = 0; row < rows; ++row)
        if (positions[row] < 0 || (uint64_t)positions[row] >= keys) return -1;
    for (size_t row = 0; row < rows; ++row)
        for (size_t key = 0; key < keys; ++key)
            output[row * keys + key] = key > (uint64_t)positions[row]
                ? UINT16_C(0xff7f) : UINT16_C(0);
    return 0;
}
