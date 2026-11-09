#ifndef MERLIN_ROW_MAJOR_K_WINDOW_H
#define MERLIN_ROW_MAJOR_K_WINDOW_H

#include <stddef.h>
#include <stdint.h>

/* Copy one contiguous K window from each row of a logical tensor. The caller
 * owns the numerical policy and any target-specific input-domain check. Both
 * byte lengths must match the declared shapes exactly. Invalid parameters
 * leave destination untouched. Source and destination must not overlap. */
static inline int merlin_pack_row_major_k_window(
    const uint8_t *source, size_t source_bytes,
    uint8_t *destination, size_t destination_bytes,
    size_t rows, size_t source_columns, size_t k_start,
    size_t tile_columns, size_t element_bytes) {
  if (!source || !destination || !rows || !source_columns ||
      !tile_columns || !element_bytes || k_start > source_columns ||
      tile_columns > source_columns - k_start ||
      source_columns > SIZE_MAX / element_bytes ||
      tile_columns > SIZE_MAX / element_bytes)
    return 1;

  size_t source_row_bytes = source_columns * element_bytes;
  size_t tile_row_bytes = tile_columns * element_bytes;
  if (rows > SIZE_MAX / source_row_bytes ||
      rows > SIZE_MAX / tile_row_bytes ||
      source_bytes != rows * source_row_bytes ||
      destination_bytes != rows * tile_row_bytes)
    return 1;

  uintptr_t source_address = (uintptr_t)source;
  uintptr_t destination_address = (uintptr_t)destination;
  if (source_bytes > UINTPTR_MAX - source_address ||
      destination_bytes > UINTPTR_MAX - destination_address ||
      (source_address < destination_address + destination_bytes &&
       destination_address < source_address + source_bytes))
    return 1;

  size_t offset_bytes = k_start * element_bytes;
  for (size_t row = 0; row < rows; ++row) {
    size_t source_offset = row * source_row_bytes + offset_bytes;
    size_t destination_offset = row * tile_row_bytes;
    for (size_t byte = 0; byte < tile_row_bytes; ++byte)
      destination[destination_offset + byte] = source[source_offset + byte];
  }
  return 0;
}

#endif
