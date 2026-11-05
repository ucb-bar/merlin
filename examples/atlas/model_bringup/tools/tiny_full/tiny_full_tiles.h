#ifndef TINY_FULL_TILES_H
#define TINY_FULL_TILES_H
#include <stddef.h>

/* M32,N32 tiles, with two consecutive 32x16 BF16 halves per tile. */
static inline size_t tiny_full_bf16_index(size_t row, size_t column, size_t width) {
    return ((row / 32) * (width / 32) + column / 32) * 1024
        + (column % 32) / 16 * 512 + (row % 32) * 16 + column % 16;
}
/* Shared by greedy selection and generated step checks. */
static inline size_t tiny_full_logits_index(size_t row, size_t column, int tiled) {
    return tiled ? tiny_full_bf16_index(row, column, 32000) : row * 32000 + column;
}
#endif
