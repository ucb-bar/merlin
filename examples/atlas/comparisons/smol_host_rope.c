#include "smol_host_rope.h"
#include "smol_host_bits.h"

int smol_host_rope_bf16(const uint16_t *input, const int64_t *positions,
                        const float *cosine, const float *sine,
                        int64_t first_position, size_t table_rows,
                        size_t rows, size_t heads, size_t head_dimension,
                        uint16_t *output) {
    if (!input || !positions || !cosine || !sine || !output || !table_rows
        || !rows || !heads || !head_dimension || head_dimension % 2
        || heads > SIZE_MAX / head_dimension
        || rows > SIZE_MAX / sizeof(int64_t)
        || rows > SIZE_MAX / sizeof(uint16_t) / (heads * head_dimension)
        || table_rows > SIZE_MAX / sizeof(float) / (head_dimension / 2)) return -1;
    size_t half = head_dimension / 2;
    /* Validate every runtime position before writing any output. */
    for (size_t row = 0; row < rows; ++row) {
        if (positions[row] < first_position
            || (uint64_t)positions[row] - (uint64_t)first_position >= table_rows) return -1;
    }
    for (size_t row = 0; row < rows; ++row) {
        size_t table = (size_t)((uint64_t)positions[row] - (uint64_t)first_position) * half;
        for (size_t head = 0; head < heads; ++head) {
            size_t base = (row * heads + head) * head_dimension;
            for (size_t column = 0; column < half; ++column) {
                float left = bf16_to_float(input[base + column]);
                float right = bf16_to_float(input[base + half + column]);
                float cos_value = cosine[table + column], sin_value = sine[table + column];
                float left_cos = left * cos_value, right_sin = right * sin_value;
                float right_cos = right * cos_value, left_sin = left * sin_value;
                output[base + column] = float_to_bf16(left_cos - right_sin);
                output[base + half + column] = float_to_bf16(right_cos + left_sin);
            }
        }
    }
    return 0;
}
