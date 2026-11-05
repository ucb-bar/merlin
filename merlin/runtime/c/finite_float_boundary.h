#ifndef MERLIN_FINITE_FLOAT_BOUNDARY_H
#define MERLIN_FINITE_FLOAT_BOUNDARY_H

/* Integer-only binary32 -> symmetric eight-bit finite-float conversion.
 * The compiler supplies the selected format's positive finite-normal codes,
 * exact RNE transition bit patterns, flush threshold and admitted-code mask.
 * Inputs are immutable for both passes. No floating-point operation, target
 * instruction, or model-specific value is selected here. */

#include <float.h>
#include <stddef.h>
#include <stdint.h>

#if FLT_RADIX != 2 || FLT_MANT_DIG != 24 || FLT_MAX_EXP != 128
#error "finite float boundary requires IEEE binary32 float"
#endif
_Static_assert(sizeof(float) == 4, "finite float boundary requires four-byte float");

typedef struct {
  const uint8_t *positive_codes;
  const uint32_t *midpoint_bits;
  size_t code_count;
  uint32_t zero_gate_bits;
  uint32_t admitted_mask[8];
  uint8_t canonicalize_negative_zero;
} merlin_f32_u8_codebook;

enum {
  MERLIN_F32_U8_OK = 0,
  MERLIN_F32_U8_INVALID = 1,
  MERLIN_F32_U8_OUT_OF_DOMAIN = 2,
};

static inline int merlin_f32_u8_valid(const merlin_f32_u8_codebook *table) {
  if (!table || !table->positive_codes || !table->code_count ||
      table->code_count > 127 || !table->zero_gate_bits ||
      table->canonicalize_negative_zero > 1u ||
      (table->zero_gate_bits & 0x7f800000u) == 0x7f800000u ||
      (table->code_count > 1 && !table->midpoint_bits))
    return 0;
  for (size_t i = 0; i < table->code_count; ++i) {
    if (!table->positive_codes[i] || table->positive_codes[i] >= 128 ||
        (i && table->positive_codes[i - 1] >= table->positive_codes[i]))
      return 0;
  }
  for (size_t i = 0; i + 1 < table->code_count; ++i) {
    uint32_t boundary = table->midpoint_bits[i];
    if (boundary <= table->zero_gate_bits ||
        (boundary & 0x7f800000u) == 0x7f800000u ||
        (i && table->midpoint_bits[i - 1] >= boundary))
      return 0;
  }
  return 1;
}

static inline int merlin_f32_u8_encode_checked_table(
    uint32_t bits, const merlin_f32_u8_codebook *table, uint8_t *result) {
  uint32_t magnitude = bits & 0x7fffffffu;
  if ((magnitude & 0x7f800000u) == 0x7f800000u)
    return MERLIN_F32_U8_INVALID;
  uint8_t code = 0;
  if (magnitude >= table->zero_gate_bits) {
    size_t low = 0, high = table->code_count - 1;
    while (low < high) {
      size_t middle = low + (high - low) / 2;
      uint32_t midpoint = table->midpoint_bits[middle];
      if (magnitude < midpoint ||
          (magnitude == midpoint && !(table->positive_codes[middle] & 1u)))
        high = middle;
      else
        low = middle + 1;
    }
    code = table->positive_codes[low];
  }
  if (bits & 0x80000000u)
    code |= 0x80u;
  if (code == 0x80u && table->canonicalize_negative_zero)
    code = 0u;
  if (!(table->admitted_mask[code >> 5] & (1u << (code & 31))))
    return MERLIN_F32_U8_OUT_OF_DOMAIN;
  *result = code;
  return MERLIN_F32_U8_OK;
}

static inline int merlin_f32_u8_encode_bits(uint32_t bits,
                                            const merlin_f32_u8_codebook *table,
                                            uint8_t *result) {
  if (!result || !merlin_f32_u8_valid(table))
    return MERLIN_F32_U8_INVALID;
  return merlin_f32_u8_encode_checked_table(bits, table, result);
}

static inline int merlin_pack_f32_u8(const float *source, uint8_t *destination,
                                     size_t count,
                                     const merlin_f32_u8_codebook *table) {
  if (!source || !destination || !count || !merlin_f32_u8_valid(table) ||
      count > SIZE_MAX / sizeof(float))
    return MERLIN_F32_U8_INVALID;
  uintptr_t src = (uintptr_t)source, dst = (uintptr_t)destination;
  size_t source_bytes = count * sizeof(float);
  if (src > UINTPTR_MAX - source_bytes || dst > UINTPTR_MAX - count ||
      (src < dst + count && dst < src + source_bytes))
    return MERLIN_F32_U8_INVALID;
  uint8_t code;
  for (size_t i = 0; i < count; ++i) {
    uint32_t bits;
    __builtin_memcpy(&bits, &source[i], sizeof(bits));
    int status = merlin_f32_u8_encode_checked_table(bits, table, &code);
    if (status != MERLIN_F32_U8_OK)
      return status;
  }
  for (size_t i = 0; i < count; ++i) {
    uint32_t bits;
    __builtin_memcpy(&bits, &source[i], sizeof(bits));
    (void)merlin_f32_u8_encode_checked_table(bits, table, &code);
    destination[i] = code;
  }
  return MERLIN_F32_U8_OK;
}

/* Read two contiguous row-major BF16 column groups into logical row-major
 * binary32 values. The caller supplies the physical split from its target
 * layout and must establish DMA completion and cache visibility first. */
static inline int merlin_unpack_split_bf16_f32(
    const volatile uint8_t *source, float *destination, size_t rows,
    size_t columns, size_t first_columns) {
  if (!source || !destination || !rows || !columns || !first_columns ||
      first_columns >= columns || rows > SIZE_MAX / columns)
    return MERLIN_F32_U8_INVALID;
  size_t count = rows * columns;
  if (count > SIZE_MAX / sizeof(float))
    return MERLIN_F32_U8_INVALID;
  size_t source_bytes = count * sizeof(uint16_t);
  size_t output_bytes = count * sizeof(float);
  uintptr_t src = (uintptr_t)source, dst = (uintptr_t)destination;
  if (src > UINTPTR_MAX - source_bytes || dst > UINTPTR_MAX - output_bytes ||
      (src < dst + output_bytes && dst < src + source_bytes))
    return MERLIN_F32_U8_INVALID;
  size_t second_columns = columns - first_columns;
  size_t second_start = rows * first_columns;
  for (size_t row = 0; row < rows; ++row) {
    for (size_t column = 0; column < columns; ++column) {
      size_t physical = column < first_columns
                            ? row * first_columns + column
                            : second_start + row * second_columns +
                                  column - first_columns;
      uint16_t code = (uint16_t)source[2 * physical] |
                      ((uint16_t)source[2 * physical + 1] << 8);
      uint32_t bits = (uint32_t)code << 16;
      __builtin_memcpy(&destination[row * columns + column], &bits,
                       sizeof(bits));
    }
  }
  return MERLIN_F32_U8_OK;
}

#endif
