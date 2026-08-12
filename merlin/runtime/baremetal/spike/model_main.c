/* Bare-metal whole-model driver for spike.
 *
 * Weights are linked in as a binary blob (objcopy/ld -b binary -> _binary_weights_bin_*);
 * the Merlin C runtime (merlin_model.c) builds memref descriptors from the generated arg
 * table and invokes the compiled forward(). The f32 output is emitted over HTIF as raw
 * 32-bit bit patterns (exact, deterministic) for the host to reinterpret and compare.
 *
 * Bit-exact reproducibility is the point: the host x86 build and this rv64gcv build share
 * the same LLVM IR, so the harness gates `spike == host`.
 */
#include <stdint.h>
#include <string.h>

#include "merlin_model.h"
#include "model_gen.h"
#include "model_io.h"

void console_init(void);
void htif_puts(const char *);
unsigned long long merlin_memref_rank_mismatches(void);
void htif_putd(long);
void htif_putc(char);
void htif_exit(int);

/* Weights are loaded at a fixed absolute address (a separate ELF section, see
 * model_link.ld) and addressed by literal constant — with multi-GB blobs they sit
 * beyond medany's ±2GB PC-relative reach, so a symbol reference would truncate. */
#ifndef MERLIN_WEIGHTS_BASE_ADDR
#define MERLIN_WEIGHTS_BASE_ADDR 0x200000000ULL
#endif

static float OUT[MERLIN_OUT_ELEMS];
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

  merlin_run(MERLIN_ARGS, MERLIN_N_ARGS, (const void *)MERLIN_WEIGHTS_BASE_ADDR,
             MERLIN_INPUT_PTR, OUT, DESCS);

  uint64_t c1;
  __asm__ volatile("csrr %0, mcycle" : "=r"(c1));

  /* Output protocol (all f32 emitted as exact 32-bit patterns):
   *   OUT <k> <bits...>     : the first k = min(N, 4096) raw values (exact prefix).
   * For large outputs (e.g. LM logits) additionally a digest the host can gate on:
   *   ARGMAX <rows> <idx...>: argmax over the last dim per row (token predictions).
   *   SUM <bits>            : f32 sum of all outputs (loose-tol checksum). */
#define MERLIN_DUMP_CAP 4096
  int k = MERLIN_OUT_ELEMS < MERLIN_DUMP_CAP ? MERLIN_OUT_ELEMS : MERLIN_DUMP_CAP;
  htif_puts("OUT ");
  htif_putd((long)k);
  for (int i = 0; i < k; i++) {
    uint32_t bits;
    memcpy(&bits, &OUT[i], 4);
    htif_putc(' ');
    htif_putd((long)(uint64_t)bits);
  }
  htif_putc('\n');

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
  htif_puts("DONE\n");
  htif_exit(0);
  return 0;
}
