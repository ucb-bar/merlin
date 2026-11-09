#ifndef MERLIN_F32_FLOOR_BITS_H
#define MERLIN_F32_FLOOR_BITS_H
#include "source_f32_math.h"
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

/* Explicit IEEE binary32 value legalization. The caller proves the matching
 * float/uint32 encoding, nontrapping arithmetic and unobserved exception flags.
 * Every finite result is the original floorf value, including signed zero.
 * No rounding environment is changed. Nonfinite inputs keep the source call.
 * This helper grants no permission to replace a source operation by default. */
_Static_assert(FLT_RADIX == 2 && FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128,
               "IEEE binary32 float required");
_Static_assert(sizeof(float) == sizeof(uint32_t), "binary32 storage required");

static inline float merlin_f32_floor_bits(float value) {
  uint32_t word;
  MERLIN_SOURCE_BITCAST_COPY(&word, &value, sizeof(word));
  unsigned exponent = (word >> 23) & 255;
  if (exponent == 255) return floorf(value);
  if (exponent >= 150) return value; /* Every such finite value is integral. */
  if (exponent < 127) {
    if ((word & UINT32_C(0x7fffffff)) == 0) return value;
    word = word >> 31 ? UINT32_C(0xbf800000) : 0;
  } else {
    /* 1 <= abs(value) < 2^23: discard fractional significand bits, adding
     * one integer magnitude for a negative value with a nonzero fraction.
     * An exponent carry at a power of two is the correct exact result. */
    uint32_t unit = UINT32_C(1) << (150 - exponent);
    uint32_t fraction = unit - 1;
    if ((word & fraction) == 0) return value;
    word = (word & ~fraction) + (word >> 31 ? unit : 0);
  }
  MERLIN_SOURCE_BITCAST_COPY(&value, &word, sizeof(value));
  return value;
}
#endif
