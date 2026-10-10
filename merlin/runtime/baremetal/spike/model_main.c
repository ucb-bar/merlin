/* Bare-metal whole-model driver for spike.
 *
 * Weights are linked in as a binary blob (objcopy/ld -b binary -> _binary_weights_bin_*);
 * the Merlin C runtime (merlin_model.c) builds memref descriptors from the generated arg
 * table and invokes the compiled forward(). The output is emitted over HTIF as raw
 * words (exact, deterministic) for the host to reinterpret and compare. FP32 keeps
 * its existing protocol; signed i64 uses two 32-bit words per element, low first.
 * Boolean results transmit their actual storage byte, without truth-value normalization.
 *
 * Bit-exact reproducibility is the point: the host x86 build and this rv64gcv build share
 * the same LLVM IR, so the harness gates `spike == host`.
 */
#include <stdint.h>
#include <string.h>

#include "merlin_model.h"
#include "model_gen.h"
#include "model_io.h"
#ifdef MERLIN_OUTPUT_SHA256
#include "output_sha256.h"
#endif

void console_init(void);
void htif_puts(const char *);
unsigned long long merlin_memref_rank_mismatches(void);
void htif_putd(long);
void htif_putc(char);
void htif_exit(int);
#ifdef MERLIN_PROF_BAREMETAL
void merlin_prof_dump(void);
#endif

/* Weights are loaded at a fixed absolute address (a separate ELF section, see
 * model_link.ld) and addressed by literal constant — with multi-GB blobs they sit
 * beyond medany's ±2GB PC-relative reach, so a symbol reference would truncate. */
#ifndef MERLIN_WEIGHTS_BASE_ADDR
#define MERLIN_WEIGHTS_BASE_ADDR 0x200000000ULL
#endif

#ifdef MERLIN_FULL_READBACK
/* Every forward result, complete, as OUT_BIN v1 frames (merlin.runtime.whole_model_readback); the
 * prefix protocol below is not emitted, so the two transports never mix on one console. */
#include "model_outputs.h"
void htif_line_flush(int);
static void merlin_put_u64(unsigned long long v) {
  char buf[24];
  int i = 0;
  do { buf[i++] = (char)('0' + (v % 10)); v /= 10; } while (v);
  while (i) htif_putc(buf[--i]);
}
static void merlin_emit_full_outputs(void) {
  for (int o = 0; o < MERLIN_N_OUTPUTS; o++) {
    const unsigned char *p = (const unsigned char *)MERLIN_OUTPUT_PTR[o];
    unsigned long long n = MERLIN_OUTPUT_BYTES[o];
    uint64_t h = 0xcbf29ce484222325ULL;
    htif_puts("OUT_BIN_BEGIN v1 ");
    htif_puts(MERLIN_OUTPUT_NAMES[o]);
    htif_putc(' '); merlin_put_u64(MERLIN_OUTPUT_ROWS[o]);
    htif_putc(' '); merlin_put_u64(MERLIN_OUTPUT_COLS[o]);
    htif_putc(' '); merlin_put_u64(MERLIN_OUTPUT_WIDTH[o]);
    htif_putc(' '); htif_putc(MERLIN_OUTPUT_SIGN[o]);
    htif_putc(' '); merlin_put_u64(n);
    htif_putc('\n');
    /* Raw payload bytes: a newline byte inside it must not split the buffered write. */
    htif_line_flush(0);
    for (unsigned long long i = 0; i < n; i++) {
      htif_putc((char)p[i]);
      h = (h ^ p[i]) * 0x100000001b3ULL;
    }
    htif_line_flush(1);
    htif_puts("OUT_BIN_END v1 ");
    for (int shift = 60; shift >= 0; shift -= 4)
      htif_putc("0123456789abcdef"[(h >> shift) & 15]);
    htif_putc('\n');
  }
}
#else
#if !MERLIN_OUT_IS_F32 && !MERLIN_OUT_IS_I64 && !MERLIN_OUT_IS_I1
#error "bare-metal model output supports only f32, i64 or i1"
#endif
#endif
#ifdef MERLIN_GROUP_PROFILE
/* Cycle brackets around every device group call (merlin.runtime.whole_model_readback). */
uint64_t merlin_group_profile_now(void);
void merlin_group_profile_dump(uint64_t forward_start, uint64_t forward_end);
#endif
#if MERLIN_OUT_IS_F32
#define OUT ((float *)MERLIN_OUTPUT_PTR[0])
#endif
static merlin_descriptor_t DESCS[MERLIN_N_ARGS];

int main(int hart) {
  if (hart != 0) {
    for (;;)
      ;
  }
  /* Before the first character. On a hosted substrate this is a no-op; on real silicon it programs
   * the console UART's clocks and baud divisor, without which printing hangs the core. */
  console_init();
  uint64_t c0;
  __asm__ volatile("csrr %0, mcycle" : "=r"(c0));

  merlin_reset_session();
  merlin_prepare_step(0);
#ifdef MERLIN_GROUP_PROFILE
  uint64_t merlin_forward_start = merlin_group_profile_now();
#endif
  merlin_run_multi(MERLIN_ARGS, MERLIN_N_ARGS, (const void *)MERLIN_WEIGHTS_BASE_ADDR,
                   MERLIN_INPUT_PTR, MERLIN_OUTPUT_PTR, DESCS);
#ifdef MERLIN_GROUP_PROFILE
  uint64_t merlin_forward_end = merlin_group_profile_now();
#endif
#if MERLIN_N_STATE_PAIRS > 0
  if (merlin_commit_state(MERLIN_ARGS, MERLIN_N_ARGS, MERLIN_INPUT_PTR,
                          MERLIN_OUTPUT_PTR, MERLIN_N_STATE_PAIRS,
                          MERLIN_STATE_INPUT_ARGS, MERLIN_STATE_OUTPUT_INDICES) != 0) {
    htif_puts("FAIL state ABI mismatch\n"); htif_exit(1);
  }
#endif

  uint64_t c1;
  __asm__ volatile("csrr %0, mcycle" : "=r"(c1));

  /* Output protocol:
   *   OUT <k> <bits...>     : the first k = min(N, 4096) raw values (exact prefix).
   *   OUT_I64 <k> <lo hi...>: i64 values as unsigned 32-bit halves, low first.
   *   OUT_I1 <k> <bytes...>: actual one-byte Boolean storage; host requires 0/1.
   * For large outputs (e.g. LM logits) additionally a digest the host can gate on:
   *   ARGMAX <rows> <idx...>: argmax over the last dim per row (token predictions).
   *   SUM <bits>            : f32 sum of all outputs (loose-tol checksum). */
#ifdef MERLIN_FULL_READBACK
  merlin_emit_full_outputs();
#else
#ifndef MERLIN_DUMP_CAP
#define MERLIN_DUMP_CAP 4096
#endif
  int k = MERLIN_OUT_ELEMS < MERLIN_DUMP_CAP ? MERLIN_OUT_ELEMS : MERLIN_DUMP_CAP;
#if MERLIN_OUT_IS_I64
  htif_puts("OUT_I64 ");
#elif MERLIN_OUT_IS_I1
  htif_puts("OUT_I1 ");
#else
  htif_puts("OUT ");
#endif
  htif_putd((long)k);
  for (int i = 0; i < k; i++) {
#if MERLIN_OUT_IS_I64
    uint64_t bits;
    memcpy(&bits, (const unsigned char *)MERLIN_OUTPUT_PTR[0] + (long)i * 8, 8);
    htif_putc(' ');
    htif_putd((long)(bits & UINT32_MAX));
    htif_putc(' ');
    htif_putd((long)(bits >> 32));
#elif MERLIN_OUT_IS_I1
    htif_putc(' ');
    htif_putd((long)((const uint8_t *)MERLIN_OUTPUT_PTR[0])[i]);
#else
    uint32_t bits;
    memcpy(&bits, &OUT[i], 4);
    htif_putc(' ');
    htif_putd((long)(uint64_t)bits);
#endif
  }
  htif_putc('\n');

#if MERLIN_OUT_IS_F32
  if (MERLIN_OUT_ELEMS > MERLIN_DUMP_CAP) {
    int rows = MERLIN_OUT_ELEMS / MERLIN_OUT_LASTDIM;
    htif_puts("ARGMAX ");
    htif_putd((long)rows);
    for (int r = 0; r < rows; r++) {
      const float *row = &OUT[(long)r * MERLIN_OUT_LASTDIM];
      int best = 0;
      float bv = row[0];
      for (int j = 1; j < MERLIN_OUT_LASTDIM; j++)
        if (row[j] > bv) { bv = row[j]; best = j; }
      htif_putc(' ');
      htif_putd((long)best);
    }
    htif_putc('\n');
    float s = 0.0f;
    for (int i = 0; i < MERLIN_OUT_ELEMS; i++)
      s += OUT[i];
    uint32_t sb;
    memcpy(&sb, &s, 4);
    htif_puts("SUM ");
    htif_putd((long)(uint64_t)sb);
    htif_putc('\n');
  }
#ifdef MERLIN_OUTPUT_SHA256
  /* Exact full first-output evidence; excluded from c1-c0 above. */
  uint8_t digest[32];
  merlin_output_sha256(OUT, (size_t)MERLIN_OUT_ELEMS, digest);
  htif_puts("OUT_SHA256 f32le ");
  htif_putd((long)MERLIN_OUT_ELEMS);
  htif_putc(' ');
  htif_putd((long)((uint64_t)MERLIN_OUT_ELEMS * 4));
  htif_putc(' ');
  for (unsigned i = 0; i < 32; i++) {
    static const char hex[] = "0123456789abcdef";
    htif_putc(hex[digest[i] >> 4]);
    htif_putc(hex[digest[i] & 15]);
  }
  htif_putc('\n');
#endif
#endif
#endif /* MERLIN_FULL_READBACK */
  htif_puts("METRIC cycles ");
  htif_putd((long)(c1 - c0));
  htif_putc('\n');
  /* Build identity, so a console log mailed back from someone else's board can be tied to a specific
     binary instead of being unattributable. Absent unless the builder defines it -> byte-identical. */
#ifdef MERLIN_BUILD_HASH
  htif_puts("METRIC build_hash " MERLIN_BUILD_HASH "\n");
#endif
  /* Which channel this log came out of, and the clock `cycles` above was counted against -- a cycle
     count is uninterpretable as time without it, and someone reading a mailed-back log has no other
     way to tell a 50 MHz reset-clock run from a PLL-raised one. */
#ifdef MERLIN_CONSOLE_NAME
  htif_puts("METRIC console " MERLIN_CONSOLE_NAME "\n");
#endif
#ifdef MERLIN_CHIP_FREQ_HZ
  htif_puts("METRIC chip_freq_hz ");
  htif_putd((long)(uint64_t)MERLIN_CHIP_FREQ_HZ);
  htif_putc('\n');
#endif
  /* What the runtime REFUSED to do. memrefCopy declines a copy whose two descriptors disagree on rank,
     because it cannot be performed and computing through it stores outside any mapping. A refusal is still
     a wrong answer -- the copy did not happen -- so a run that hit one has to say so, or it grades badly
     with no reason given. Reported unconditionally: zero is the common case, and a metric that appears only
     when things break is one nobody knows to look for. */
  htif_puts("METRIC memref_rank_mismatch ");
  htif_putd((long)merlin_memref_rank_mismatches());
  htif_putc('\n');
#ifdef MERLIN_GROUP_PROFILE
  merlin_group_profile_dump(merlin_forward_start, merlin_forward_end);
#endif
#ifdef MERLIN_PROF_BAREMETAL
  /* Per-op ticks, emitted only by a build that instrumented the IR to produce them. Placed after the
     output and the cycle metric so a profiled run is a superset of a normal one -- the same grade, the
     same whole-model cycle count, plus the breakdown -- rather than a different run that has to be
     compared across images. */
  merlin_prof_dump();
#endif
  htif_puts("DONE\n");
  htif_exit(0);
  return 0;
}
