#ifndef LLAMA_ROTARY_TABLE_H
#define LLAMA_ROTARY_TABLE_H

#include <stddef.h>
#include <stdint.h>

/* Slice immutable model/config-derived tables using actual runtime positions.
 * All outputs require storage disjoint from inputs and each other. Invalid
 * positions are rejected before any output write. No trig or attention here. */
int llama_host_rotary_lookup_bf16(const uint16_t *cosine_table,
                                 const uint16_t *sine_table,
                                 size_t table_rows, size_t width,
                                 const int64_t *positions, size_t rows,
                                 uint16_t *cosine, uint16_t *sine);

#endif
