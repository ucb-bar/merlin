/* Fixed-six-place IEEE binary64 formatting with exact nearest-even rounding.
 * This is diagnostic transport, not a source arithmetic implementation.
 * Integer scaling avoids overflow, double rounding and floating environment
 * dependencies. The caller supplies 320 bytes, including the final NUL.
 */
#ifndef MERLIN_FIXED_DECIMAL_H
#define MERLIN_FIXED_DECIMAL_H
#include <stddef.h>
#include <stdint.h>

static unsigned merlin_decimal_bit(const uint32_t *words, unsigned bit) {
  return bit < 34 * 32 ? (words[bit / 32] >> (bit % 32)) & 1u : 0;
}

static size_t merlin_binary64_fixed6(char *text, uint64_t bits) {
  size_t size = 0;
  if (bits >> 63) text[size++] = '-';
  unsigned exponent = (unsigned)(bits >> 52) & 2047u;
  uint64_t fraction = bits & UINT64_C(0xfffffffffffff);
  if (exponent == 2047u) {
    const char *word = fraction ? "nan" : "inf";
    while (*word) text[size++] = *word++;
    text[size] = 0;
    return size;
  }
  uint64_t significand = fraction | (exponent ? UINT64_C(1) << 52 : 0);
  uint32_t words[34] = {(uint32_t)significand, (uint32_t)(significand >> 32)};
  uint64_t carry = 0;
  /* value * 10^6 = significand * 15625 * 2^(binary_shift + 6). */
  for (unsigned i = 0; i < 34; i++) {
    uint64_t product = (uint64_t)words[i] * 15625u + carry;
    words[i] = (uint32_t)product;
    carry = product >> 32;
  }
  int shift = exponent ? (int)exponent - 1069 : -1068;
  if (shift >= 0) {
    unsigned whole = (unsigned)shift / 32, part = (unsigned)shift % 32;
    for (int i = 33; i >= 0; i--) {
      uint32_t value = (unsigned)i >= whole ? words[i - whole] << part : 0;
      if (part && (unsigned)i > whole) value |= words[i - whole - 1] >> (32 - part);
      words[i] = value;
    }
  } else {
    unsigned right = (unsigned)-shift;
    unsigned guard = merlin_decimal_bit(words, right - 1), sticky = 0;
    for (unsigned i = 0; i + 1 < right; i++) sticky |= merlin_decimal_bit(words, i);
    unsigned increment = guard && (sticky || merlin_decimal_bit(words, right));
    unsigned whole = right / 32, part = right % 32;
    for (unsigned i = 0; i < 34; i++) {
      uint32_t value = i + whole < 34 ? words[i + whole] >> part : 0;
      if (part && i + whole + 1 < 34) value |= words[i + whole + 1] << (32 - part);
      words[i] = value;
    }
    carry = increment;
    for (unsigned i = 0; i < 34; i++) {
      uint64_t sum = (uint64_t)words[i] + carry;
      words[i] = (uint32_t)sum;
      carry = sum >> 32;
    }
  }
  char digits[320];
  unsigned count = 0, nonzero;
  do {
    carry = 0; nonzero = 0;
    for (int i = 33; i >= 0; i--) {
      uint64_t current = (carry << 32) | words[i];
      words[i] = (uint32_t)(current / 10);
      carry = current % 10;
      nonzero |= words[i];
    }
    digits[count++] = (char)('0' + carry);
  } while (nonzero);
  while (count < 7) digits[count++] = '0';
  while (count) {
    text[size++] = digits[--count];
    if (count == 6) text[size++] = '.';
  }
  text[size] = 0;
  return size;
}
#endif
