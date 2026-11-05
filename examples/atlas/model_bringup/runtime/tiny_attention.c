#include "tiny_attention.h"
#include "atlas_numeric.h"
#include <string.h>

size_t tiny_attention_index(size_t row, size_t column, size_t width) {
    size_t tile = (row / 32) * (width / 32) + column / 32;
    return tile * 1024 + ((column % 32) / 16) * 512 + (row % 32) * 16 + column % 16;
}

int tiny_attention_pack(const uint16_t *input, size_t rows, size_t width, int scale,
                        uint8_t *output, size_t capacity) {
    size_t physical_rows = (rows + 31) / 32 * 32;
    if (!input || !output || (rows != 1 && rows != 128) || width != 2048
        || scale < -126 || scale > 127 || capacity < physical_rows * width) return -1;
    size_t cursor = 0;
    for (size_t tr = 0; tr < physical_rows; tr += 32)
        for (size_t tc = 0; tc < width; tc += 32)
            for (size_t r = 0; r < 32; ++r)
                for (size_t c = 0; c < 32; ++c) {
                    uint16_t value = input[tiny_attention_index(tr + r, tc + c, width)];
                    if ((value & 0x7f80u) == 0x7f80u) return -1;
                    output[cursor++] = tr + r < rows ? pack_bf16(value, scale) : 0;
                }
    return 0;
}

static float bf16_float(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float output;
    memcpy(&output, &bits, sizeof(output));
    return output;
}

static int rotary_word(const uint16_t input[64], size_t column, uint16_t cosine, uint16_t sine,
                        uint16_t *output) {
    uint16_t rotated = input[column < 32 ? column + 32 : column - 32];
    if (column < 32) rotated ^= 0x8000u;
    uint16_t first, second;
    if (f32_bf16(bf16_float(input[column]) * bf16_float(cosine), &first)
        || f32_bf16(bf16_float(rotated) * bf16_float(sine), &second)) return -1;
    return f32_bf16(bf16_float(first) + bf16_float(second), output);
}

int tiny_attention_prepare(size_t rows, size_t past_rows, uint16_t *query,
    const uint16_t *key, const uint16_t *value, const uint16_t *past_keys, const uint16_t *past_values,
    const uint16_t *cosine, const uint16_t *sine, uint16_t *keys, uint16_t *values) {
    if ((rows != 1 && rows != 128) || past_rows > 160 - rows || !query || !key || !value
        || !cosine || !sine || !keys || !values || (past_rows && (!past_keys || !past_values))) return -1;
    if (past_rows) {
        memcpy(keys, past_keys, 4 * 160 * 64 * sizeof(uint16_t));
        memcpy(values, past_values, 4 * 160 * 64 * sizeof(uint16_t));
    } else {
        memset(keys, 0, 4 * 160 * 64 * sizeof(uint16_t));
        memset(values, 0, 4 * 160 * 64 * sizeof(uint16_t));
    }
    for (size_t row = 0; row < rows; ++row) {
        size_t position = past_rows + row;
        for (size_t head = 0; head < 32; ++head) {
            uint16_t input[64];
            for (size_t d = 0; d < 64; ++d) input[d] = query[tiny_attention_index(row, head * 64 + d, 2048)];
            for (size_t d = 0; d < 64; ++d)
                if (rotary_word(input, d, cosine[position * 64 + d], sine[position * 64 + d],
                    query + tiny_attention_index(row, head * 64 + d, 2048))) return -1;
        }
        for (size_t head = 0; head < 4; ++head) {
            uint16_t input[64];
            for (size_t d = 0; d < 64; ++d) input[d] = key[tiny_attention_index(row, head * 64 + d, 256)];
            for (size_t d = 0; d < 64; ++d) {
                size_t index = tiny_attention_index(position, head * 64 + d, 256);
                if (rotary_word(input, d, cosine[position * 64 + d], sine[position * 64 + d], keys + index)) return -1;
                values[index] = value[tiny_attention_index(row, head * 64 + d, 256)];
            }
        }
    }
    return 0;
}

int tiny_attention_prepare_rotated(size_t rows, const uint16_t *query, const uint16_t *key, const uint16_t *value,
                                   uint16_t *keys, uint16_t *values) {
    if (rows != 128 || !query || !key || !value || !keys || !values) return -1;
    for (size_t i = 0; i < 128 * 2048; ++i)
        if ((query[i] & 0x7f80u) == 0x7f80u || (i < 128 * 256 && (key[i] & 0x7f80u) == 0x7f80u)) return -1;
    memset(keys, 0, 4 * 160 * 64 * sizeof(uint16_t));
    memset(values, 0, 4 * 160 * 64 * sizeof(uint16_t));
    /* Positions 0..127 occupy the first four row tiles of the 160-row cache layout. */
    memcpy(keys, key, 128 * 256 * sizeof(uint16_t));
    memcpy(values, value, 128 * 256 * sizeof(uint16_t));
    return 0;
}

int tiny_attention_append_rotated(size_t past_rows, const uint16_t *query, const uint16_t *key,
    const uint16_t *value, const uint16_t *past_keys, const uint16_t *past_values,
    uint16_t *keys, uint16_t *values) {
    if (past_rows < 128 || past_rows > 129 || !query || !key || !value || !past_keys || !past_values
        || !keys || !values) return -1;
    for (size_t column = 0; column < 2048; ++column)
        if ((query[tiny_attention_index(0, column, 2048)] & 0x7f80u) == 0x7f80u
            || (column < 256 && (key[tiny_attention_index(0, column, 256)] & 0x7f80u) == 0x7f80u)) return -1;
    memcpy(keys, past_keys, 4 * 160 * 64 * sizeof(uint16_t));
    memcpy(values, past_values, 4 * 160 * 64 * sizeof(uint16_t));
    for (size_t column = 0; column < 256; ++column) {
        size_t destination = tiny_attention_index(past_rows, column, 256);
        size_t source = tiny_attention_index(0, column, 256);
        keys[destination] = key[source];
        values[destination] = value[source];
    }
    return 0;
}
