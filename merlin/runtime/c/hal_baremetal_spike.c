/* Merlin HAL — spike bare-metal (RV64) implementation. libc-free + heap-free: the static arena is
 * the fixed absolute-addressed region (same convention as merlin_malloc.c), bound directly by the
 * compiler's static memory plan (no bump needed — offsets are pre-planned); parallel_for is serial
 * (single hart in v1); ticks = rdcycle; log = HTIF. This is the proof that the lean runtime drops to
 * bare metal with NO OS, NO libc, NO threads — only this seam changes vs Linux. */
#include "merlin_hal.h"

#ifndef MERLIN_ARENA_BASE_ADDR
#define MERLIN_ARENA_BASE_ADDR 0xC0000000ULL
#endif
#ifndef MERLIN_ARENA_SIZE_BYTES
#define MERLIN_ARENA_SIZE_BYTES 0x10000000ULL /* 256 MB default; grow for big models */
#endif

/* HTIF console (provided by the spike harness htif.c). */
extern void htif_puts(const char *s);

void   merlin_hal_arena_init(size_t bytes) { (void)bytes; /* region is fixed/absolute */ }
uint8_t *merlin_hal_arena_base(void) { return (uint8_t *)(uintptr_t)MERLIN_ARENA_BASE_ADDR; }
size_t merlin_hal_arena_size(void) { return (size_t)MERLIN_ARENA_SIZE_BYTES; }
void   merlin_hal_arena_reset(void) { /* one planned arena; nothing to reclaim mid-inference */ }

void *merlin_hal_memcpy(void *d, const void *s, size_t n) {
  unsigned char *dd = (unsigned char *)d; const unsigned char *ss = (const unsigned char *)s;
  for (size_t i = 0; i < n; i++) dd[i] = ss[i];
  return d;
}
void *merlin_hal_memset(void *d, int c, size_t n) {
  unsigned char *dd = (unsigned char *)d;
  for (size_t i = 0; i < n; i++) dd[i] = (unsigned char)c;
  return d;
}

uint64_t merlin_hal_now_ticks(void) {
  uint64_t c;
  __asm__ volatile("rdcycle %0" : "=r"(c));
  return c;
}
void merlin_hal_log(const char *msg) { htif_puts(msg); htif_puts("\n"); }

/* single-hart bare metal: parallel_for runs serially (correct; multi-hart is a phase-3 HAL). */
void merlin_hal_parallel_for(int64_t lo, int64_t hi, merlin_hal_body_fn body, void *ctx,
                             int n_workers) {
  (void)n_workers;
  for (int64_t i = lo; i < hi; i++) body(i, ctx);
}

void merlin_hal_launch(int64_t gx, int64_t gy, int64_t gz,
                       int64_t bx, int64_t by, int64_t bz,
                       merlin_hal_lane_fn lane, void *ctx) {
  for (int64_t a = 0; a < (gx > 0 ? gx : 1); a++)
   for (int64_t b = 0; b < (gy > 0 ? gy : 1); b++)
    for (int64_t c = 0; c < (gz > 0 ? gz : 1); c++)
     for (int64_t i = 0; i < (bx > 0 ? bx : 1); i++)
      for (int64_t j = 0; j < (by > 0 ? by : 1); j++)
       for (int64_t k = 0; k < (bz > 0 ? bz : 1); k++)
         lane(a, b, c, i, j, k, ctx);
}

uint32_t merlin_hal_capabilities(void) { return MERLIN_CAP_SCALAR | MERLIN_CAP_RVV; }
int merlin_hal_has(uint32_t cap) { return (merlin_hal_capabilities() & cap) == cap; }
