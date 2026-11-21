/* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LLVM-LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * Binary16/binary32 specialization adapted from compiler-rt's extendhfsf2.c,
 * fp_extend.h, fp_extend_impl.inc, truncsfhf2.c and fp_trunc_impl.inc at
 * a47bddccec30255619bb8c37fa59700e661d4e66.
 * Preserve its normal/subnormal/zero/nonfinite word mapping and actual floating
 * argument ABI. No half-to-float expression may recursively call this helper.
 */
#ifndef MERLIN_HALF_EXTEND_H
#define MERLIN_HALF_EXTEND_H
#include <stdint.h>

static uint32_t merlin_half_extend_word(uint16_t input) {
  uint32_t sign = (uint32_t)(input & 0x8000u) << 16;
  uint32_t exponent = (input >> 10) & 31u, fraction = input & 1023u;
  if (exponent == 31u) return sign | 0x7f800000u | (fraction << 13);
  if (exponent) return sign | ((exponent + 112u) << 23) | (fraction << 13);
  if (!fraction) return sign;
  exponent = 113u;
  while (!(fraction & 1024u)) { fraction <<= 1; exponent--; }
  return sign | (exponent << 23) | ((fraction & 1023u) << 13);
}

/* Round binary32 to binary16, nearest ties to even with gradual underflow.
 * Integer rounding includes the hidden bit for subnormals and carries into
 * the exponent at normal/subnormal and finite/infinity boundaries. NaNs retain
 * sign and truncated payload and become quiet, matching compiler-rt. */
static uint16_t merlin_half_trunc_word(uint32_t input) {
  uint16_t sign = (uint16_t)((input >> 16) & 0x8000u);
  uint32_t exponent = (input >> 23) & 255u, fraction = input & 0x7fffffu;
  if (exponent == 255u)
    return sign | 0x7c00u | (fraction ? 0x0200u | (fraction >> 13) : 0u);
  if (exponent >= 143u) return sign | 0x7c00u;
  if (exponent < 102u) return sign;
  unsigned shift = exponent < 113u ? 126u - exponent : 13u;
  uint32_t significand = fraction | 0x800000u;
  uint32_t rounded = significand >> shift;
  uint32_t remainder = significand & ((1u << shift) - 1u);
  uint32_t halfway = 1u << (shift - 1u);
  rounded += remainder > halfway || (remainder == halfway && (rounded & 1u));
  uint32_t magnitude = exponent < 113u ? rounded : ((exponent - 113u) << 10) + rounded;
  return sign | (uint16_t)magnitude;
}

#if defined(__FLT16_MANT_DIG__) && __FLT16_MANT_DIG__ == 11
float __extendhfsf2(_Float16 value) {
  uint16_t input;
  __builtin_memcpy(&input, &value, sizeof(input));
  uint32_t word = merlin_half_extend_word(input);
  float result;
  __builtin_memcpy(&result, &word, sizeof(result));
  return result;
}

_Float16 __truncsfhf2(float value) {
  uint32_t input;
  __builtin_memcpy(&input, &value, sizeof(input));
  uint16_t word = merlin_half_trunc_word(input);
  _Float16 result;
  __builtin_memcpy(&result, &word, sizeof(result));
  return result;
}
#endif
#endif
