/* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LLVM-LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * Binary16/binary32 specialization adapted from compiler-rt's extendhfsf2.c,
 * fp_extend.h and fp_extend_impl.inc at a47bddccec30255619bb8c37fa59700e661d4e66.
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

#if defined(__FLT16_MANT_DIG__) && __FLT16_MANT_DIG__ == 11
float __extendhfsf2(_Float16 value) {
  uint16_t input;
  __builtin_memcpy(&input, &value, sizeof(input));
  uint32_t word = merlin_half_extend_word(input);
  float result;
  __builtin_memcpy(&result, &word, sizeof(result));
  return result;
}
#endif
#endif
