/* Runtime table lookup only. Tables are immutable model/config parameters,
 * generated offline by the admitted native rotary-table producer. */
#include "llama_rotary_table.h"

#include <string.h>

static int valid_shape(size_t rows, size_t width) {
    return rows && width && rows <= SIZE_MAX / width
        && rows * width <= SIZE_MAX / sizeof(uint16_t);
}

int llama_host_rotary_lookup_bf16(const uint16_t *cosine_table,
                                 const uint16_t *sine_table,
                                 size_t table_rows, size_t width,
                                 const int64_t *positions, size_t rows,
                                 uint16_t *cosine, uint16_t *sine) {
    if (!cosine_table || !sine_table || !positions || !cosine || !sine
        || !valid_shape(table_rows, width) || !valid_shape(rows, width)
        || rows > SIZE_MAX / sizeof(int64_t)
        || cosine == sine || cosine == cosine_table || cosine == sine_table
        || sine == cosine_table || sine == sine_table
        || (const void *)cosine == (const void *)positions
        || (const void *)sine == (const void *)positions) return -1;
    for (size_t row = 0; row < rows; ++row)
        if (positions[row] < 0 || (uint64_t)positions[row] >= table_rows) return -1;
    for (size_t row = 0; row < rows; ++row) {
        memcpy(cosine + row * width, cosine_table + (size_t)positions[row] * width,
               width * sizeof(uint16_t));
        memcpy(sine + row * width, sine_table + (size_t)positions[row] * width,
               width * sizeof(uint16_t));
    }
    return 0;
}
