/* Source-native BF16 boundaries for batch-one Llama host layout operations.
 * Compile with contraction and fast math disabled. Rotary cos/sin are caller
 * inputs; generating native tables and QK/softmax/PV remain separate gates. */
#include "llama_host_layout.h"

#include <string.h>

static int valid_shape(size_t heads, size_t rows, size_t width) {
    return heads && rows && width && rows <= SIZE_MAX / heads
        && width <= SIZE_MAX / (heads * rows)
        && heads * rows * width <= SIZE_MAX / sizeof(uint16_t);
}

static float bf16_float(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float output;
    memcpy(&output, &bits, sizeof(output));
    return output;
}

static uint16_t float_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    if ((bits & UINT32_C(0x7fffffff)) > UINT32_C(0x7f800000))
        return (uint16_t)((bits >> 16) | UINT32_C(0x40));
    return (uint16_t)((bits + UINT32_C(0x7fff) + ((bits >> 16) & 1u)) >> 16);
}

int llama_host_split_heads_bf16(const uint16_t *linear, uint16_t *output,
                                size_t rows, size_t heads, size_t width) {
    if (!linear || !output || linear == output || !valid_shape(heads, rows, width)) return -1;
    for (size_t head = 0; head < heads; ++head)
        for (size_t row = 0; row < rows; ++row)
            memcpy(output + (head * rows + row) * width,
                   linear + (row * heads + head) * width, width * sizeof(uint16_t));
    return 0;
}

int llama_host_rotary_bf16(const uint16_t *input, const uint16_t *cosine,
                          const uint16_t *sine, uint16_t *output,
                          size_t heads, size_t rows, size_t width) {
    if (!input || !cosine || !sine || !output || output == input
        || output == cosine || output == sine || !valid_shape(heads, rows, width)
        || width % 2) return -1;
    for (size_t head = 0; head < heads; ++head) {
        for (size_t row = 0; row < rows; ++row) {
            for (size_t column = 0; column < width; ++column) {
                size_t offset = (head * rows + row) * width;
                size_t other = column < width / 2 ? column + width / 2 : column - width / 2;
                /* rotate_half is bit-exact negation; each tensor multiply
                 * rounds separately to BF16 before the BF16 addition. */
                uint16_t rotated = input[offset + other];
                if (column < width / 2) rotated ^= UINT16_C(0x8000);
                uint16_t first = float_bf16(bf16_float(input[offset + column])
                                           * bf16_float(cosine[row * width + column]));
                uint16_t second = float_bf16(bf16_float(rotated)
                                            * bf16_float(sine[row * width + column]));
                output[offset + column] = float_bf16(bf16_float(first) + bf16_float(second));
            }
        }
    }
    return 0;
}

int llama_host_cache_append_bf16(const uint16_t *prefix, const uint16_t *tail,
                                uint16_t *output, size_t heads,
                                size_t past_rows, size_t rows, size_t width) {
    if (!tail || !output || (past_rows && !prefix) || output == tail
        || (prefix && output == prefix) || past_rows > SIZE_MAX - rows
        || !valid_shape(heads, past_rows + rows, width) || !rows) return -1;
    for (size_t head = 0; head < heads; ++head) {
        uint16_t *destination = output + head * (past_rows + rows) * width;
        if (past_rows) memcpy(destination, prefix + head * past_rows * width,
                              past_rows * width * sizeof(uint16_t));
        memcpy(destination + past_rows * width, tail + head * rows * width,
               rows * width * sizeof(uint16_t));
    }
    return 0;
}

int llama_host_repeat_kv_bf16(const uint16_t *input, uint16_t *output,
                             size_t heads, size_t groups, size_t rows,
                             size_t width) {
    if (!input || !output || input == output || !heads || !groups
        || groups > SIZE_MAX / heads || !valid_shape(heads * groups, rows, width)) return -1;
    for (size_t head = 0; head < heads; ++head)
        for (size_t group = 0; group < groups; ++group)
            memcpy(output + (head * groups + group) * rows * width,
                   input + head * rows * width, rows * width * sizeof(uint16_t));
    return 0;
}
