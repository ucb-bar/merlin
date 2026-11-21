/* Target-independent, lossless binary output transport over a length-taking writer.
 *
 * The callback must use a coherent guest-to-host write service. It receives
 * exactly raw_len bytes, including NULs, and reports whether all were written.
 * This codec sees actual container words only; the caller owns source layout,
 * frame metadata and terminal DONE. No golden, sample or numeric tolerance is
 * present here. The checksum detects accidental transport damage, not an
 * adversarially forged payload and not numerical correctness.
 * The writer may change not-yet-read source values, but must not mutate the
 * encoder's configuration or state while a scalar or bulk call is active.
 */
#ifndef MERLIN_OUT_BIN_H
#define MERLIN_OUT_BIN_H

#include <stddef.h>
#include <stdint.h>

#define MERLIN_OUT_BIN_RAW_CAP 1024u

typedef struct {
  unsigned char raw[MERLIN_OUT_BIN_RAW_CAP];
  int (*write_bytes)(const void *, size_t);
  uint64_t expected_words;
  uint64_t seen_words;
  uint64_t checksum;
  unsigned raw_len;
  unsigned word_bytes;
  int signed_words;
  int valid;
} merlin_out_bin;

static int merlin_out_bin_width_valid(unsigned width) {
  return width == 1u || width == 2u || width == 4u || width == 8u;
}

static void merlin_out_bin_init(merlin_out_bin *out, uint64_t expected_words,
                                unsigned word_bytes, int signed_words,
                                int (*write_bytes)(const void *, size_t)) {
  out->write_bytes = write_bytes;
  out->expected_words = expected_words;
  out->seen_words = 0;
  out->checksum = UINT64_C(0xcbf29ce484222325);
  out->raw_len = 0;
  out->word_bytes = word_bytes;
  out->signed_words = signed_words;
  out->valid = write_bytes != 0 && expected_words != 0 &&
               merlin_out_bin_width_valid(word_bytes) &&
               (signed_words == 0 || signed_words == 1);
}

static int merlin_out_bin_flush(merlin_out_bin *out) {
  if (!out->valid || !out->write_bytes)
    return 0;
  if (out->raw_len == 0)
    return 1;
  if (!out->write_bytes(out->raw, out->raw_len))
    return 0;
  out->raw_len = 0;
  return 1;
}

static int merlin_out_bin_word(merlin_out_bin *out, uint64_t word) {
  uint64_t mask;
  uint64_t narrowed;
  unsigned i;
  if (!out->valid || out->seen_words >= out->expected_words)
    return 0;
  mask = out->word_bytes == 8u ? UINT64_MAX :
         (UINT64_C(1) << (8u * out->word_bytes)) - UINT64_C(1);
  narrowed = word & mask;
  if (out->signed_words) {
    uint64_t sign_bit = UINT64_C(1) << (8u * out->word_bytes - 1u);
    uint64_t extended = (narrowed & sign_bit) ? narrowed | ~mask : narrowed;
    if (word != extended)
      return 0;
  } else if (word != narrowed) {
    return 0;
  }
  for (i = 0; i < out->word_bytes; ++i) {
    unsigned char byte = (unsigned char)(narrowed >> (8u * i));
    out->raw[out->raw_len++] = byte;
    out->checksum = (out->checksum ^ byte) * UINT64_C(0x100000001b3);
    if (out->raw_len == MERLIN_OUT_BIN_RAW_CAP && !merlin_out_bin_flush(out))
      return 0;
  }
  ++out->seen_words;
  return 1;
}

/* Optional contiguous int32 transport. A caller must prove that `words`
 * traverses the actual logical output in order; this helper knows no layout,
 * reference values or target. Validate each word AFTER any preceding callback
 * (which may mutate not-yet-read output), but keep the mask, counters and FNV
 * state local between 1024-byte flushes. Scalar callers are unchanged.
 */
static inline int merlin_out_bin_words_i32(merlin_out_bin *out,
                                            const int32_t *words, uint64_t count) {
  int64_t lower, upper;
  uint64_t seen, checksum, mask;
  unsigned width, raw_len;
  if (!out)
    return 0;
  if ((count && !words) || count > SIZE_MAX / sizeof(*words) ||
      !out->valid || !out->write_bytes ||
      !merlin_out_bin_width_valid(out->word_bytes) ||
      (out->signed_words != 0 && out->signed_words != 1) ||
      out->raw_len >= MERLIN_OUT_BIN_RAW_CAP ||
      out->raw_len % out->word_bytes != 0 ||
      out->seen_words > out->expected_words ||
      count > out->expected_words - out->seen_words) {
    out->valid = 0;
    return 0;
  }
  width = out->word_bytes;
  switch (width) {
  case 1u:
    lower = out->signed_words ? INT8_MIN : 0;
    upper = out->signed_words ? INT8_MAX : UINT8_MAX;
    break;
  case 2u:
    lower = out->signed_words ? INT16_MIN : 0;
    upper = out->signed_words ? INT16_MAX : UINT16_MAX;
    break;
  case 4u:
  case 8u:
    lower = out->signed_words ? INT32_MIN : 0;
    upper = INT32_MAX;
    break;
  default:
    return 0;
  }
  mask = width == 8u ? UINT64_MAX :
         (UINT64_C(1) << (8u * width)) - UINT64_C(1);
  seen = out->seen_words;
  checksum = out->checksum;
  raw_len = out->raw_len;
  for (uint64_t index = 0; index < count; ++index) {
    int64_t value = words[index];
    if (value < lower || value > upper) {
      out->seen_words = seen;
      out->checksum = checksum;
      out->raw_len = raw_len;
      out->valid = 0;
      return 0;
    }
    uint64_t narrowed = (uint64_t)value & mask;
    for (unsigned byte_index = 0; byte_index < width; ++byte_index) {
      unsigned char byte = (unsigned char)(narrowed >> (8u * byte_index));
      out->raw[raw_len++] = byte;
      checksum = (checksum ^ byte) * UINT64_C(0x100000001b3);
    }
    if (raw_len == MERLIN_OUT_BIN_RAW_CAP) {
      out->seen_words = seen;
      out->checksum = checksum;
      out->raw_len = raw_len;
      if (!merlin_out_bin_flush(out)) {
        out->valid = 0;
        return 0;
      }
      raw_len = 0;
    }
    ++seen;
  }
  out->seen_words = seen;
  out->checksum = checksum;
  out->raw_len = raw_len;
  return 1;
}

static int merlin_out_bin_finish(merlin_out_bin *out) {
  return out->valid && out->seen_words == out->expected_words &&
         merlin_out_bin_flush(out);
}

#endif
