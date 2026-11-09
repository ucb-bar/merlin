#ifndef MERLIN_BENCHMARK_BUFFER_H
#define MERLIN_BENCHMARK_BUFFER_H
#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* Exact benchmark validation outside the measured ROI. Every byte is checked;
 * this is not a checksum. Fixed-size object copies permit unaligned inputs and
 * avoid aliasing violations. No target alignment or endian order is assumed.
 * A nonempty extent must designate readable storage, as with memcmp. */
#if defined(__GNUC__) || defined(__clang__)
#define MERLIN_BENCHMARK_COPY(d,s,n) __builtin_memcpy((d),(s),(n))
#else
#define MERLIN_BENCHMARK_COPY(d,s,n) memcpy((d),(s),(n))
#endif

static inline size_t merlin_benchmark_word_difference(
    const void *expected, const void *actual, size_t size) {
#if defined(__GNUC__) || defined(__clang__)
  expected = __builtin_assume_aligned(expected, _Alignof(uint64_t));
  actual = __builtin_assume_aligned(actual, _Alignof(uint64_t));
#endif
  const unsigned char *a = (const unsigned char *)expected;
  const unsigned char *b = (const unsigned char *)actual;
  const size_t word_bytes = sizeof(uint64_t);
  size_t i = 0;
  while (size - i >= word_bytes) {
    uint64_t x, y;
    MERLIN_BENCHMARK_COPY(&x, a + i, word_bytes);
    MERLIN_BENCHMARK_COPY(&y, b + i, word_bytes);
    if (x != y) {
      for (size_t j = 0; j < word_bytes; ++j)
        if (a[i + j] != b[i + j]) return i + j;
    }
    i += word_bytes;
  }
  for (; i < size; ++i)
    if (a[i] != b[i]) return i;
  return size;
}

static inline size_t merlin_benchmark_first_difference(
    const void *expected, const void *actual, size_t size) {
#if defined(__GNUC__) || defined(__clang__)
  const size_t alignment = _Alignof(uint64_t);
  if (size && (uintptr_t)expected % alignment == 0 &&
              (uintptr_t)actual % alignment == 0) {
    return merlin_benchmark_word_difference(expected, actual, size);
  }
#endif
  const unsigned char *a = (const unsigned char *)expected;
  const unsigned char *b = (const unsigned char *)actual;
  for (size_t i = 0; i < size; ++i)
    if (a[i] != b[i]) return i;
  return size;
}

/* Reinitialize before each measured call, outside its ROI. All bytes, including
 * tails, are written; the caller separately preserves its exterior guards. */
static inline void merlin_benchmark_fill_words(
    void *destination, unsigned char value, size_t size) {
#if defined(__GNUC__) || defined(__clang__)
  destination = __builtin_assume_aligned(destination, _Alignof(uint64_t));
#endif
  unsigned char *out = (unsigned char *)destination;
  unsigned char block[sizeof(uint64_t)];
  for (size_t j = 0; j < sizeof(block); ++j) block[j] = value;
  size_t i = 0;
  while (size - i >= sizeof(block)) {
    MERLIN_BENCHMARK_COPY(out + i, block, sizeof(block));
    i += sizeof(block);
  }
  for (; i < size; ++i) out[i] = value;
}

static inline void merlin_benchmark_fill(
    void *destination, unsigned char value, size_t size) {
#if defined(__GNUC__) || defined(__clang__)
  if (size && (uintptr_t)destination % _Alignof(uint64_t) == 0) {
    merlin_benchmark_fill_words(destination, value, size);
    return;
  }
#endif
  unsigned char *out = (unsigned char *)destination;
  for (size_t i = 0; i < size; ++i) out[i] = value;
}
#undef MERLIN_BENCHMARK_COPY
#endif
