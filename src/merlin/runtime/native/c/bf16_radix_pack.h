#ifndef MERLIN_BF16_RADIX_PACK_H
#define MERLIN_BF16_RADIX_PACK_H

/* Optional target-neutral row-scaled signed-radix packing of BF16 operands.
 * Inputs are losslessly widened BF16 binary32 values. Digits are signed7bit
 * magnitudes, independent of any accelerator layout; caller supplies strides.
 * Dynamic step is 2^(floor(log2(max(abs(row))))+1-7*digits), zero-row exponent0.
 * Coefficients use nearest-even then saturate to 2^(7*digits)-1, as the ordinary
 * floating divider/lrint reference does. Eligibility uses the same IEEE/RNE
 * contract as the optional FMA certificate; no FMA execution state is consumed.
 * Failure selects the original source path. Outputs may be partially written
 * on failure and must then be discarded. Valid storage/nonoverlap is caller's
 * responsibility. Digits1..3 keep coefficients exactly binary32 representable.
 */
#include "source_f32_math.h"
#include "ordered_fma_bounds.h"

static inline int merlin_bf16_radix_row(
    const merlin_fma_bound *eligibility, const float *source, size_t length,
    size_t source_stride, float *reconstructed, size_t reconstructed_stride,
    int8_t *planes, size_t digit_stride, size_t element_stride,
    unsigned digits, float *step) {
  if (!eligibility || !eligibility->valid || !source || !reconstructed || !planes || !step ||
      !length || digits < 1 || digits > 3 ||
      (source_stride && length-1 > SIZE_MAX/source_stride) ||
      (reconstructed_stride && length-1 > SIZE_MAX/reconstructed_stride) ||
      (element_stride && length-1 > SIZE_MAX/element_stride) ||
      (digits > 1 && digit_stride > SIZE_MAX/(digits-1))) return 0;
  const size_t last_element = (length-1)*element_stride;
  if ((digits-1)*digit_stride > SIZE_MAX-last_element) return 0;
  uint32_t maximum = 0;
  for (size_t z=0; z<length; ++z) {
    uint32_t raw; MERLIN_SOURCE_BITCAST_COPY(&raw,source+z*source_stride,sizeof(raw));
    const uint32_t magnitude = raw & UINT32_C(0x7fffffff);
    if ((raw & UINT32_C(0xffff)) || magnitude >= UINT32_C(0x7f800000)) return 0;
    if (magnitude > maximum) maximum = magnitude;
  }
  int exponent = 0;
  if (maximum) {
    const unsigned biased = maximum >> 23;
    if (biased) exponent = (int)biased-127;
    else {
      unsigned fraction = maximum >> 16;
      exponent = -133;
      while (fraction >>= 1) ++exponent;
    }
  }
  const int step_exponent = exponent+1-7*(int)digits;
  if (step_exponent < -149 || step_exponent > 127) return 0;
  const uint32_t step_bits = step_exponent >= -126 ?
      (uint32_t)(step_exponent+127) << 23 : UINT32_C(1) << (step_exponent+149);
  MERLIN_SOURCE_BITCAST_COPY(step,&step_bits,sizeof(step_bits));
  const uint32_t limit = (UINT32_C(1) << (7*digits))-1;
  for (size_t z=0; z<length; ++z) {
    uint32_t raw; MERLIN_SOURCE_BITCAST_COPY(&raw,source+z*source_stride,sizeof(raw));
    const unsigned biased = (raw >> 23) & 255;
    const unsigned mantissa = ((raw >> 16) & 127) | (biased ? 128 : 0);
    const int value_exponent = biased ? (int)biased-127-7 : -133;
    const int shift = value_exponent-step_exponent;
    uint32_t coefficient;
    if (shift >= 0) coefficient = mantissa << shift;
    else {
      const unsigned right = (unsigned)-shift;
      if (right >= 9) coefficient = 0;
      else {
        const unsigned quotient = mantissa >> right;
        const unsigned remainder = mantissa & ((1U << right)-1);
        const unsigned half = 1U << (right-1);
        coefficient = quotient + (remainder > half ||
            (remainder == half && (quotient & 1)));
      }
    }
    if (coefficient > limit) coefficient = limit;
    const int negative = (raw >> 31) != 0;
    const int32_t signed_coefficient = negative ? -(int32_t)coefficient : (int32_t)coefficient;
    reconstructed[z*reconstructed_stride] = (float)signed_coefficient * *step;
    for (unsigned digit=0; digit<digits; ++digit) {
      const int value = (int)((coefficient >> (7*digit)) & 127);
      planes[digit*digit_stride+z*element_stride] = (int8_t)(negative ? -value : value);
    }
  }
  return 1;
}

#endif
