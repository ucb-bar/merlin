#include "host_matrix_fp8pack.h"

#include <string.h>
/* Supplied by the build as an exact immutable source snapshot. Reuse the
 * qualified numeric primitives without changing the convolution baseline. */
#include "atlas_resnet_graph_math.c"

static int matrix_extent(size_t rows, size_t columns, size_t element_bytes) {
    return rows && columns && rows <= SIZE_MAX / columns
        && rows * columns <= SIZE_MAX / element_bytes;
}

static int buffer_span(const void *pointer, size_t bytes) {
    return pointer && bytes && bytes - 1 <= UINTPTR_MAX - (uintptr_t)pointer;
}

static int overlap(const void *a, size_t a_bytes, const void *b, size_t b_bytes) {
    uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
    return x <= y ? y - x < a_bytes : x - y < b_bytes;
}

static int finite_input(const uint16_t *input, size_t count) {
    for (size_t i = 0; i < count; ++i)
        if ((input[i] & UINT16_C(0x7f80)) == UINT16_C(0x7f80)) return 0;
    return 1;
}

int host_matrix_expand_bf16(const uint16_t *input, float *output, size_t count) {
    if (sizeof(float) != sizeof(uint32_t) || !count || count > SIZE_MAX / sizeof(float)
        || !buffer_span(input, count * sizeof(uint16_t))
        || !buffer_span(output, count * sizeof(float))
        || overlap(input, count * sizeof(uint16_t), output, count * sizeof(float))
        || !finite_input(input, count)) return -1;
    for (size_t i = 0; i < count; ++i) {
        uint32_t word = (uint32_t)input[i] << 16;
        memcpy(output + i, &word, sizeof(word));
    }
    return 0;
}

int host_matrix_scale_bf16(const uint16_t *input, size_t rows, size_t columns,
                           int *scale) {
    if (!matrix_extent(rows, columns, sizeof(uint16_t))
        || !buffer_span(input, rows * columns * sizeof(uint16_t))
        || !buffer_span(scale, sizeof(*scale))
        || overlap(input, rows * columns * sizeof(uint16_t), scale, sizeof(*scale))
        || !finite_input(input, rows * columns)) return -1;
    uint16_t maximum = 0;
    for (size_t i = 0; i < rows * columns; ++i) {
        uint16_t magnitude = input[i] & UINT16_C(0x7fff);
        if (magnitude > maximum) maximum = magnitude;
    }
    int found;
    if (tensor_shift(maximum, &found)) return -1;
    *scale = found;
    return 0;
}

int host_matrix_pack_bf16_at_scale(const uint16_t *input,
                                  size_t rows, size_t columns,
                                  size_t physical_rows, size_t physical_columns,
                                  size_t tile_rows, size_t tile_columns,
                                  int scale, uint8_t *output, size_t capacity) {
    if (!matrix_extent(rows, columns, sizeof(uint16_t))
        || !matrix_extent(physical_rows, physical_columns, 1)
        || rows > physical_rows || columns > physical_columns
        || !tile_rows || !tile_columns || physical_rows % tile_rows
        || physical_columns % tile_columns || scale < -126 || scale > 127
        || capacity < physical_rows * physical_columns
        || !buffer_span(input, rows * columns * sizeof(uint16_t))
        || !buffer_span(output, physical_rows * physical_columns)
        || overlap(input, rows * columns * sizeof(uint16_t), output, physical_rows * physical_columns)
        || !finite_input(input, rows * columns)) return -1;
    size_t cursor = 0;
    for (size_t tile_row = 0; tile_row < physical_rows; tile_row += tile_rows)
        for (size_t tile_column = 0; tile_column < physical_columns; tile_column += tile_columns)
            for (size_t row = 0; row < tile_rows; ++row)
                for (size_t column = 0; column < tile_columns; ++column) {
                    size_t r = tile_row + row, c = tile_column + column;
                    output[cursor++] = r < rows && c < columns
                        ? pack_bf16(input[r * columns + c], scale) : 0;
                }
    return 0;
}

int host_matrix_pack_bf16(const uint16_t *input,
                         size_t rows, size_t columns,
                         size_t physical_rows, size_t physical_columns,
                         size_t tile_rows, size_t tile_columns,
                         uint8_t *output, size_t capacity, int *scale) {
    if (!matrix_extent(rows, columns, sizeof(uint16_t))
        || !matrix_extent(physical_rows, physical_columns, 1)
        || !buffer_span(scale, sizeof(*scale))
        || !buffer_span(input, rows * columns * sizeof(uint16_t))
        || !buffer_span(output, physical_rows * physical_columns)
        || overlap(scale, sizeof(*scale), input, rows * columns * sizeof(uint16_t))
        || overlap(scale, sizeof(*scale), output, physical_rows * physical_columns)) return -1;
    int found;
    if (host_matrix_scale_bf16(input, rows, columns, &found)) return -1;
    int result = host_matrix_pack_bf16_at_scale(input, rows, columns, physical_rows,
                                              physical_columns, tile_rows, tile_columns,
                                              found, output, capacity);
    if (!result) *scale = found;
    return result;
}
