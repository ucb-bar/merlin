/* One-pass, exact output packet codec. The selected harness owns the closed
 * output roster and coherent sink. This codec consumes only actual container
 * words and never consults a reference value or accelerator fact.
 *
 * For signed widths >1, every value in [-127,127] is one byte except -128;
 * 0x80 escapes to the full little-endian word. Unsigned widths >1 inline
 * [0,254] and escape 0xff. Width-one and floating bit patterns use raw mode.
 * FNV-1a covers every payload byte, including escape markers. A checksum is
 * transport integrity only; ordinary numerical comparison remains mandatory.
 */
#ifndef MERLIN_OUT_PACKET_V2_H
#define MERLIN_OUT_PACKET_V2_H

#include <stddef.h>
#include <stdint.h>

#define MERLIN_OUT_PACKET_V2_BUFFER 1024u

typedef struct {
  unsigned char buffer[MERLIN_OUT_PACKET_V2_BUFFER];
  int (*write_bytes)(const void *, size_t);
  uint64_t expected;
  uint64_t seen;
  uint64_t checksum;
  unsigned buffered;
  unsigned width;
  int signed_words;
  int escaped;
  int valid;
} merlin_out_packet_v2;

static inline void merlin_out_packet_v2_init(
    merlin_out_packet_v2 *out, uint64_t count, unsigned width,
    int signed_words, int escaped, int (*write_bytes)(const void *, size_t)) {
  if (!out) return;
  out->write_bytes = write_bytes;
  out->expected = count;
  out->seen = 0;
  out->checksum = UINT64_C(0xcbf29ce484222325);
  out->buffered = 0;
  out->width = width;
  out->signed_words = signed_words;
  out->escaped = escaped;
  out->valid = write_bytes && count &&
      (width == 1u || width == 2u || width == 4u || width == 8u) &&
      (signed_words == 0 || signed_words == 1) &&
      (escaped == 0 || (escaped == 1 && width > 1u));
}

static inline int merlin_out_packet_v2_flush(merlin_out_packet_v2 *out) {
  if (!out || !out->valid || !out->write_bytes) return 0;
  if (!out->buffered) return 1;
  if (!out->write_bytes(out->buffer, out->buffered)) {
    out->valid = 0;
    return 0;
  }
  out->buffered = 0;
  return 1;
}

static inline int merlin_out_packet_v2_byte(merlin_out_packet_v2 *out,
                                            unsigned char byte) {
  if (!out || !out->valid) return 0;
  out->buffer[out->buffered++] = byte;
  out->checksum = (out->checksum ^ byte) * UINT64_C(0x100000001b3);
  if (out->buffered == MERLIN_OUT_PACKET_V2_BUFFER)
    return merlin_out_packet_v2_flush(out);
  return 1;
}

static inline int merlin_out_packet_v2_word(merlin_out_packet_v2 *out,
                                            uint64_t word) {
  if (!out || !out->valid || out->seen >= out->expected) return 0;
  uint64_t mask = out->width == 8u ? UINT64_MAX :
      (UINT64_C(1) << (out->width * 8u)) - 1u;
  uint64_t narrowed = word & mask;
  if (out->signed_words) {
    uint64_t sign = UINT64_C(1) << (out->width * 8u - 1u);
    if (word != ((narrowed & sign) ? narrowed | ~mask : narrowed)) return 0;
  } else if (word != narrowed) return 0;
  if (out->escaped) {
    if (out->signed_words && (int64_t)word >= -127 && (int64_t)word <= 127) {
      if (!merlin_out_packet_v2_byte(out, (unsigned char)word)) return 0;
      ++out->seen;
      return 1;
    }
    if (!out->signed_words && word <= 254u) {
      if (!merlin_out_packet_v2_byte(out, (unsigned char)word)) return 0;
      ++out->seen;
      return 1;
    }
    if (!merlin_out_packet_v2_byte(out, out->signed_words ? 0x80u : 0xffu)) return 0;
  }
  for (unsigned byte = 0; byte < out->width; ++byte)
    if (!merlin_out_packet_v2_byte(out, (unsigned char)(narrowed >> (8u * byte)))) return 0;
  ++out->seen;
  return 1;
}

/* Character-backed output arrays may contain i32 words written by a device.
 * The alias-qualified load lets the same bulk encoder read those bytes without
 * asserting a C effective type that the device never established. */
#if defined(__GNUC__) || defined(__clang__)
typedef int32_t merlin_out_packet_v2_i32_alias __attribute__((__may_alias__));
#else
typedef int32_t merlin_out_packet_v2_i32_alias;
#endif

static inline int merlin_out_packet_v2_words_i32(
    merlin_out_packet_v2 *out, const merlin_out_packet_v2_i32_alias *words,
    uint64_t count) {
  if (!out || (count && !words) || count > SIZE_MAX / sizeof(*words) ||
      !out->valid || out->width != 4u || !out->signed_words ||
      out->seen > out->expected || count > out->expected - out->seen) return 0;
  if (!out->escaped) {
    for (uint64_t index = 0; index < count; ++index)
      if (!merlin_out_packet_v2_word(out, (uint64_t)(int64_t)words[index])) return 0;
    return 1;
  }
  /* Keep invariant fields and the hot byte cursor in registers. Only the
   * uncommon full-word escape or a full buffer enters the slow path. */
  uint64_t checksum = out->checksum;
  unsigned buffered = out->buffered;
  uint64_t start = out->seen;
  for (uint64_t index = 0; index < count; ++index) {
    int32_t value = words[index];
    if (value >= -127 && value <= 127) {
      unsigned char byte = (unsigned char)value;
      out->buffer[buffered++] = byte;
      checksum = (checksum ^ byte) * UINT64_C(0x100000001b3);
      if (buffered == MERLIN_OUT_PACKET_V2_BUFFER) {
        out->buffered = buffered;
        out->checksum = checksum;
        out->seen = start + index;
        if (!merlin_out_packet_v2_flush(out)) return 0;
        buffered = 0;
      }
    } else {
      uint32_t bits = (uint32_t)value;
      for (unsigned part = 0; part != 5u; ++part) {
        unsigned char byte = part == 0u ? 0x80u :
            (unsigned char)(bits >> (8u * (part - 1u)));
        out->buffer[buffered++] = byte;
        checksum = (checksum ^ byte) * UINT64_C(0x100000001b3);
        if (buffered == MERLIN_OUT_PACKET_V2_BUFFER) {
          out->buffered = buffered;
          out->checksum = checksum;
          out->seen = start + index;
          if (!merlin_out_packet_v2_flush(out)) return 0;
          buffered = 0;
        }
      }
    }
  }
  out->buffered = buffered;
  out->checksum = checksum;
  out->seen = start + count;
  return 1;
}

static inline int merlin_out_packet_v2_finish(merlin_out_packet_v2 *out) {
  return out && out->valid && out->seen == out->expected &&
      merlin_out_packet_v2_flush(out);
}

#endif
