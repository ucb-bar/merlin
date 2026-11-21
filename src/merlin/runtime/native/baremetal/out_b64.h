/* Merlin's target-independent, lossless bare-metal output transport.
 *
 * A caller supplies a text writer that sends one NUL-terminated block in one
 * host transaction.  The output WORDS are copied into this bounded staging
 * buffer before the callback: the host never reads a possibly dirty model
 * output buffer directly.  The ASCII framing keeps existing text-only console
 * capture paths usable, while the host decoder reconstructs every container
 * bit.  This header does not know an accelerator, a golden, or a tolerance.
 */
#ifndef MERLIN_OUT_B64_H
#define MERLIN_OUT_B64_H

#include <stdint.h>

#define MERLIN_OUT_B64_RAW_CAP 768u
#define MERLIN_OUT_B64_LINE_CAP 1120u

typedef struct {
  unsigned char raw[MERLIN_OUT_B64_RAW_CAP];
  char line[MERLIN_OUT_B64_LINE_CAP];
  void (*write_text)(const char *);
  uint64_t expected_words;
  uint64_t seen_words;
  uint32_t next_chunk;
  unsigned raw_len;
  unsigned word_bytes;
  int signed_words;
  int valid;
} merlin_out_b64;

/* A first pass over the ACTUAL output buffer chooses a lossless wire width.
 * The declared C container supplies signedness and maximum width; no expected
 * output, reference value, or source shape influences this decision.
 */
typedef struct {
  uint64_t expected_words;
  uint64_t seen_words;
  uint64_t max_unsigned;
  int64_t min_signed;
  int64_t max_signed;
  unsigned logical_bytes;
  int signed_words;
  int valid;
} merlin_out_b64_range;

static int merlin_out_b64_width_valid(unsigned bytes) {
  return bytes == 1u || bytes == 2u || bytes == 4u || bytes == 8u;
}

static void merlin_out_b64_range_init(merlin_out_b64_range *range,
                                      uint64_t expected_words,
                                      unsigned logical_bytes, int signed_words) {
  range->expected_words = expected_words;
  range->seen_words = 0;
  range->max_unsigned = 0;
  range->min_signed = INT64_MAX;
  range->max_signed = INT64_MIN;
  range->logical_bytes = logical_bytes;
  range->signed_words = signed_words;
  range->valid = expected_words != 0 && merlin_out_b64_width_valid(logical_bytes) &&
                 (signed_words == 0 || signed_words == 1);
}

static int merlin_out_b64_range_signed(merlin_out_b64_range *range, int64_t value) {
  if (!range->valid || !range->signed_words || range->seen_words >= range->expected_words ||
      (range->logical_bytes == 1u && (value < INT8_MIN || value > INT8_MAX)) ||
      (range->logical_bytes == 2u && (value < INT16_MIN || value > INT16_MAX)) ||
      (range->logical_bytes == 4u && (value < INT32_MIN || value > INT32_MAX)))
    return 0;
  if (value < range->min_signed)
    range->min_signed = value;
  if (value > range->max_signed)
    range->max_signed = value;
  ++range->seen_words;
  return 1;
}

static int merlin_out_b64_range_unsigned(merlin_out_b64_range *range, uint64_t value) {
  if (!range->valid || range->signed_words || range->seen_words >= range->expected_words ||
      (range->logical_bytes == 1u && value > UINT8_MAX) ||
      (range->logical_bytes == 2u && value > UINT16_MAX) ||
      (range->logical_bytes == 4u && value > UINT32_MAX))
    return 0;
  if (value > range->max_unsigned)
    range->max_unsigned = value;
  ++range->seen_words;
  return 1;
}

static unsigned merlin_out_b64_range_width(const merlin_out_b64_range *range) {
  if (!range->valid || range->seen_words != range->expected_words)
    return 0;
  if (range->signed_words && range->min_signed < 0) {
    if (range->min_signed >= INT8_MIN && range->max_signed <= INT8_MAX)
      return 1u;
    if (range->min_signed >= INT16_MIN && range->max_signed <= INT16_MAX)
      return 2u;
    if (range->min_signed >= INT32_MIN && range->max_signed <= INT32_MAX)
      return 4u;
    return 8u;
  }
  /* A nonnegative signed value has the same integer value under unsigned
   * transport.  This permits 128..255 to use u8 without changing its declared
   * signed source type or the host checker's logical dtype. */
  uint64_t maximum = range->signed_words ? (uint64_t)range->max_signed : range->max_unsigned;
  if (maximum <= UINT8_MAX)
    return 1u;
  if (maximum <= UINT16_MAX)
    return 2u;
  if (maximum <= UINT32_MAX)
    return 4u;
  return 8u;
}

static int merlin_out_b64_range_wire_signed(const merlin_out_b64_range *range) {
  if (!range->valid || range->seen_words != range->expected_words)
    return -1;
  return range->signed_words && range->min_signed < 0;
}

static int merlin_out_b64_flush(merlin_out_b64 *out) {
  static const char alphabet[] =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  static const char hex[] = "0123456789abcdef";
  static const char prefix[] = "OUT_B64_CHUNK ";
  char *p = out->line;
  unsigned i;
  if (!out->valid || !out->write_text)
    return 0;
  if (out->raw_len == 0)
    return 1;
  if (out->next_chunk == UINT32_MAX)
    return 0;
  for (i = 0; prefix[i]; ++i)
    *p++ = prefix[i];
  for (i = 0; i < 8; ++i)
    *p++ = hex[(out->next_chunk >> (28u - 4u * i)) & 15u];
  *p++ = ' ';
  for (i = 0; i < 4; ++i)
    *p++ = hex[(out->raw_len >> (12u - 4u * i)) & 15u];
  *p++ = ' ';
  for (i = 0; i < out->raw_len; i += 3u) {
    unsigned a = out->raw[i];
    unsigned b = i + 1u < out->raw_len ? out->raw[i + 1u] : 0u;
    unsigned c = i + 2u < out->raw_len ? out->raw[i + 2u] : 0u;
    *p++ = alphabet[a >> 2];
    *p++ = alphabet[((a & 3u) << 4) | (b >> 4)];
    *p++ = i + 1u < out->raw_len ? alphabet[((b & 15u) << 2) | (c >> 6)] : '=';
    *p++ = i + 2u < out->raw_len ? alphabet[c & 63u] : '=';
  }
  *p++ = '\n';
  *p = '\0';
  out->write_text(out->line);
  out->raw_len = 0;
  ++out->next_chunk;
  return 1;
}

static void merlin_out_b64_init(merlin_out_b64 *out, uint64_t expected_words,
                                unsigned word_bytes, int signed_words,
                                void (*write_text)(const char *)) {
  out->write_text = write_text;
  out->expected_words = expected_words;
  out->seen_words = 0;
  out->next_chunk = 0;
  out->raw_len = 0;
  out->word_bytes = word_bytes;
  out->signed_words = signed_words;
  out->valid = write_text != 0 && expected_words != 0 &&
               merlin_out_b64_width_valid(word_bytes) &&
               (signed_words == 0 || signed_words == 1);
}

static int merlin_out_b64_word(merlin_out_b64 *out, uint64_t word) {
  unsigned i;
  uint64_t mask;
  uint64_t narrowed;
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
  if (out->raw_len + out->word_bytes > MERLIN_OUT_B64_RAW_CAP && !merlin_out_b64_flush(out))
    return 0;
  for (i = 0; i < out->word_bytes; ++i)
    out->raw[out->raw_len++] = (unsigned char)(word >> (8u * i));
  ++out->seen_words;
  return 1;
}

/* A typed contiguous buffer can select the wire-width checks once instead of
 * rebuilding a mask and sign extension for every word.  This is an optional
 * packing path: callers must prove their logical traversal is contiguous and
 * pass the same count they would have fed to merlin_out_b64_word.  Bounds are
 * still checked on every ACTUAL value, including values changed since a range
 * scan.  The staging buffer, chunk framing, and completion checks are shared.
 */
static inline int merlin_out_b64_words_i32_bounded(
    merlin_out_b64 *out, const int32_t *words, uint64_t count,
    int64_t lower, int64_t upper, unsigned wire_bytes) {
  for (uint64_t index = 0; index < count; ++index) {
    int64_t value = words[index];
    if (value < lower || value > upper)
      return 0;
    if (out->raw_len + wire_bytes > MERLIN_OUT_B64_RAW_CAP && !merlin_out_b64_flush(out))
      return 0;
    uint64_t raw = (uint64_t)value;
    for (unsigned byte = 0; byte < wire_bytes; ++byte)
      out->raw[out->raw_len++] = (unsigned char)(raw >> (8u * byte));
    ++out->seen_words;
  }
  return 1;
}

static inline int merlin_out_b64_words_i32(merlin_out_b64 *out,
                                     const int32_t *words, uint64_t count) {
  if (!out || (count && !words) || !out->valid || !out->write_text ||
      !merlin_out_b64_width_valid(out->word_bytes) ||
      (out->signed_words != 0 && out->signed_words != 1) ||
      out->raw_len > MERLIN_OUT_B64_RAW_CAP ||
      out->raw_len % out->word_bytes != 0 ||
      out->seen_words > out->expected_words ||
      count > out->expected_words - out->seen_words)
    return 0;
  switch (out->word_bytes) {
  case 1u:
    return out->signed_words
               ? merlin_out_b64_words_i32_bounded(out, words, count, INT8_MIN, INT8_MAX, 1u)
               : merlin_out_b64_words_i32_bounded(out, words, count, 0, UINT8_MAX, 1u);
  case 2u:
    return out->signed_words
               ? merlin_out_b64_words_i32_bounded(out, words, count, INT16_MIN, INT16_MAX, 2u)
               : merlin_out_b64_words_i32_bounded(out, words, count, 0, UINT16_MAX, 2u);
  case 4u:
    return merlin_out_b64_words_i32_bounded(out, words, count,
                                             out->signed_words ? INT32_MIN : 0,
                                             INT32_MAX, 4u);
  case 8u:
    return merlin_out_b64_words_i32_bounded(out, words, count, INT32_MIN, INT32_MAX, 8u);
  default:
    return 0;
  }
}

static int merlin_out_b64_finish(merlin_out_b64 *out) {
  if (!out->valid || out->seen_words != out->expected_words)
    return 0;
  return merlin_out_b64_flush(out);
}

#endif
