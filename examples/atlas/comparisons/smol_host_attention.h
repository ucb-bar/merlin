#ifndef SMOL_HOST_ATTENTION_H
#define SMOL_HOST_ATTENTION_H

#include <stddef.h>
#include <stdint.h>

/* Source-native Smol arithmetic: BF16 BLHD Q/K/V -> FP32 QK/scale/finite-mask
 * -> full-axis CPU16 softmax -> BF16 probabilities -> BF16 PV. Query and
 * output are [rows, heads, dimension], K/V [keys, kv_heads, dimension], mask
 * [rows, keys]. Scratch scores/probabilities are [heads, rows, keys]; workspace
 * has keys FP32 elements. All buffers must be disjoint and caller-sized.
 * Finite Q/K/V, Boolean mask, nonzero bounded extents and nearest rounding
 * are checked before writes. An overflowing QK score refuses before output
 * or probabilities are written, but can modify score scratch. PV follows the
 * current native oneDNN AVX512 BF16 emulation's upper-then-lower pair order.
 * This fixed software policy is not universal Torch/ISA/matmul parity.
 */
int smol_host_attention_cpu16_bf16(
    const uint16_t *query, const uint16_t *key, const uint16_t *value,
    const uint8_t *mask, size_t rows, size_t keys, size_t heads,
    size_t kv_heads, size_t dimension, float *workspace,
    float *scores, uint16_t *probabilities, uint16_t *output);

#endif
