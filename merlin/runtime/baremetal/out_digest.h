/* Target-independent 64-bit output digest: XXH64 (seed 0) over a buffer's actual bytes.
 *
 * Rendered by the trusted grader harness, never by candidate code. It reads the output buffer the
 * candidate wrote and prints nothing itself; the harness prints `OUT_DIGEST <name> <nbytes> <hex>`.
 * The digest detects any accidental difference from the expected bytes (a full-value readback of the
 * same ELF on a cheap engine is the correctness evidence); it is not a cryptographic commitment.
 * Byte order is little-endian, as the RISC-V harness stores every container word.
 */
#ifndef MERLIN_OUT_DIGEST_H
#define MERLIN_OUT_DIGEST_H

#include <stddef.h>
#include <stdint.h>

#define MERLIN_XXH_P1 UINT64_C(0x9E3779B185EBCA87)
#define MERLIN_XXH_P2 UINT64_C(0xC2B2AE3D27D4EB4F)
#define MERLIN_XXH_P3 UINT64_C(0x165667B19E3779F9)
#define MERLIN_XXH_P4 UINT64_C(0x85EBCA77C2B2AE63)
#define MERLIN_XXH_P5 UINT64_C(0x27D4EB2F165667C5)

static inline uint64_t merlin_xxh_rotl(uint64_t x, unsigned r) { return (x << r) | (x >> (64u - r)); }

static inline uint64_t merlin_xxh_read64(const unsigned char *p) {
  uint64_t v = 0;
  for (unsigned i = 0; i < 8u; ++i) v |= (uint64_t)p[i] << (8u * i);
  return v;
}

static inline uint64_t merlin_xxh_round(uint64_t acc, uint64_t lane) {
  acc += lane * MERLIN_XXH_P2;
  return merlin_xxh_rotl(acc, 31) * MERLIN_XXH_P1;
}

static inline uint64_t merlin_xxh_merge(uint64_t acc, uint64_t val) {
  acc ^= merlin_xxh_round(0, val);
  return acc * MERLIN_XXH_P1 + MERLIN_XXH_P4;
}

static uint64_t merlin_out_digest(const void *data, uint64_t len) {
  const unsigned char *p = (const unsigned char *)data;
  const unsigned char *end = p + len;
  uint64_t h;
  if (len >= 32u) {
    uint64_t v1 = MERLIN_XXH_P1 + MERLIN_XXH_P2, v2 = MERLIN_XXH_P2, v3 = 0, v4 = (uint64_t)0 - MERLIN_XXH_P1;
    const unsigned char *limit = end - 32;
    /* An aligned buffer reads whole words; the byte assembly above is only for the unaligned tail. */
    if (((uintptr_t)p & 7u) == 0u) {
      do {
        const uint64_t *w = (const uint64_t *)(const void *)p;
        v1 = merlin_xxh_round(v1, w[0]);
        v2 = merlin_xxh_round(v2, w[1]);
        v3 = merlin_xxh_round(v3, w[2]);
        v4 = merlin_xxh_round(v4, w[3]);
        p += 32;
      } while (p <= limit);
    } else {
      do {
        v1 = merlin_xxh_round(v1, merlin_xxh_read64(p));
        v2 = merlin_xxh_round(v2, merlin_xxh_read64(p + 8));
        v3 = merlin_xxh_round(v3, merlin_xxh_read64(p + 16));
        v4 = merlin_xxh_round(v4, merlin_xxh_read64(p + 24));
        p += 32;
      } while (p <= limit);
    }
    h = merlin_xxh_rotl(v1, 1) + merlin_xxh_rotl(v2, 7) + merlin_xxh_rotl(v3, 12) + merlin_xxh_rotl(v4, 18);
    h = merlin_xxh_merge(h, v1);
    h = merlin_xxh_merge(h, v2);
    h = merlin_xxh_merge(h, v3);
    h = merlin_xxh_merge(h, v4);
  } else {
    h = MERLIN_XXH_P5;
  }
  h += len;
  while (p + 8 <= end) {
    h ^= merlin_xxh_round(0, merlin_xxh_read64(p));
    h = merlin_xxh_rotl(h, 27) * MERLIN_XXH_P1 + MERLIN_XXH_P4;
    p += 8;
  }
  if (p + 4 <= end) {
    uint64_t k = (uint64_t)p[0] | ((uint64_t)p[1] << 8) | ((uint64_t)p[2] << 16) | ((uint64_t)p[3] << 24);
    h ^= k * MERLIN_XXH_P1;
    h = merlin_xxh_rotl(h, 23) * MERLIN_XXH_P2 + MERLIN_XXH_P3;
    p += 4;
  }
  while (p < end) {
    h ^= (uint64_t)(*p) * MERLIN_XXH_P5;
    h = merlin_xxh_rotl(h, 11) * MERLIN_XXH_P1;
    ++p;
  }
  h ^= h >> 33;
  h *= MERLIN_XXH_P2;
  h ^= h >> 29;
  h *= MERLIN_XXH_P3;
  h ^= h >> 32;
  return h;
}

#endif
