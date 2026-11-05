#include "smol_host_controls.h"

static int boolean_values(const uint8_t *values, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (values[i] > 1) return 0;
    }
    return 1;
}

int smol_host_prefix_controls(const uint8_t *pad, const uint8_t *attention,
                             size_t rows, int64_t *groups,
                             uint8_t *mask, int64_t *positions) {
    if (!pad || !attention || !groups || !mask || !positions || !rows
        || rows > INT64_MAX || rows > SIZE_MAX / sizeof(int64_t)
        || rows > SIZE_MAX / rows) return -1;
    if (!boolean_values(pad, rows) || !boolean_values(attention, rows)) return -1;
    int64_t group = 0, position = -1;
    for (size_t row = 0; row < rows; ++row) {
        group += attention[row];
        position += pad[row];
        groups[row] = group;
        positions[row] = position;
    }
    for (size_t row = 0; row < rows; ++row) {
        for (size_t key = 0; key < rows; ++key) {
            mask[row * rows + key] = (uint8_t)(pad[row] && pad[key]
                                                   && groups[key] <= groups[row]);
        }
    }
    return 0;
}

int smol_host_suffix_controls(const uint8_t *prefix_pad, size_t prefix_rows,
                             const uint8_t *suffix_pad,
                             const uint8_t *suffix_attention,
                             size_t suffix_rows, int64_t *groups,
                             uint8_t *mask, int64_t *positions) {
    if ((prefix_rows && !prefix_pad) || !suffix_pad || !suffix_attention
        || !groups || !mask || !positions || !suffix_rows
        || prefix_rows > SIZE_MAX - suffix_rows
        || prefix_rows + suffix_rows > INT64_MAX
        || suffix_rows > SIZE_MAX / sizeof(int64_t)
        || suffix_rows > SIZE_MAX / (prefix_rows + suffix_rows)) return -1;
    if (!boolean_values(prefix_pad, prefix_rows)
        || !boolean_values(suffix_pad, suffix_rows)
        || !boolean_values(suffix_attention, suffix_rows)) return -1;
    size_t boundaries = 0;
    for (size_t row = 0; row < suffix_rows; ++row) {
        boundaries += suffix_attention[row];
        if (boundaries > 256) return -1;
    }
    int64_t position = -1, group = 0;
    for (size_t row = 0; row < prefix_rows; ++row) position += prefix_pad[row];
    for (size_t row = 0; row < suffix_rows; ++row) {
        position += suffix_pad[row];
        group += suffix_attention[row];
        positions[row] = position;
        groups[row] = group;
    }
    size_t keys = prefix_rows + suffix_rows;
    for (size_t row = 0; row < suffix_rows; ++row) {
        for (size_t key = 0; key < prefix_rows; ++key) {
            mask[row * keys + key] = prefix_pad[key];
        }
        for (size_t key = 0; key < suffix_rows; ++key) {
            mask[row * keys + prefix_rows + key] = (uint8_t)(suffix_pad[row]
                && suffix_pad[key] && groups[key] <= groups[row]);
        }
    }
    return 0;
}
