#ifndef HOST_MATRIX_FP8PACK_H
#define HOST_MATRIX_FP8PACK_H

#include <stddef.h>
#include <stdint.h>

/* Exact finite BF16 expansion and source-bound FP8Pack numeric policy. The
 * translation unit includes an immutable, hash-qualified runtime-math source
 * for its existing scalar conversion, scale and FP8 encoding primitives.
 * Matrix tiling is caller-parameterized; padding emits positive-zero bytes. */
int host_matrix_expand_bf16(const uint16_t *input, float *output, size_t count);
int host_matrix_scale_bf16(const uint16_t *input, size_t rows, size_t columns,
                           int *scale);
int host_matrix_pack_bf16_at_scale(const uint16_t *input,
                                  size_t rows, size_t columns,
                                  size_t physical_rows, size_t physical_columns,
                                  size_t tile_rows, size_t tile_columns,
                                  int scale, uint8_t *output, size_t capacity);
int host_matrix_pack_bf16(const uint16_t *input,
                         size_t rows, size_t columns,
                         size_t physical_rows, size_t physical_columns,
                         size_t tile_rows, size_t tile_columns,
                         uint8_t *output, size_t capacity, int *scale);

#endif
