#ifndef SMOL_HOST_ROPE_H
#define SMOL_HOST_ROPE_H

#include <stddef.h>
#include <stdint.h>

/* Tables are FP32 constants derived from the pinned source operator parameters,
 * with row zero corresponding to first_position; runtime positions choose rows.
 */
int smol_host_rope_bf16(const uint16_t *input, const int64_t *positions,
                        const float *cosine, const float *sine,
                        int64_t first_position, size_t table_rows,
                        size_t rows, size_t heads, size_t head_dimension,
                        uint16_t *output);

#endif
