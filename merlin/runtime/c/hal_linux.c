/* Merlin HAL — Linux / hosted (glibc) implementation. Used for K1 Linux + the x86 host.
 * pthread parallel_for (genuinely concurrent); clock_gettime ticks; malloc'd arena. This is the
 * ONLY place the lean runtime touches libc/pthread on Linux — the engine + generated code stay
 * platform-agnostic behind merlin_hal.h. */
#include "merlin_hal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>

static uint8_t *g_arena = NULL;
static size_t   g_arena_bytes = 0;

void merlin_hal_arena_init(size_t bytes) {
  if (g_arena && g_arena_bytes >= bytes) return;
  free(g_arena);
  g_arena = (uint8_t *)malloc(bytes ? bytes : 1);
  g_arena_bytes = bytes;
}
uint8_t *merlin_hal_arena_base(void) { return g_arena; }
size_t   merlin_hal_arena_size(void) { return g_arena_bytes; }
void     merlin_hal_arena_reset(void) { /* one-arena, lifetime-planned: nothing to reclaim */ }

void *merlin_hal_memcpy(void *d, const void *s, size_t n) { return memcpy(d, s, n); }
void *merlin_hal_memset(void *d, int c, size_t n) { return memset(d, c, n); }

uint64_t merlin_hal_now_ticks(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}
void merlin_hal_log(const char *msg) { fputs(msg, stderr); fputc('\n', stderr); }

/* ---- pthread parallel_for: split [lo,hi) into n_workers contiguous chunks. n_workers<=1 -> serial. */
typedef struct { int64_t lo, hi; merlin_hal_body_fn body; void *ctx; } chunk_t;
static void *worker(void *arg) {
  chunk_t *c = (chunk_t *)arg;
  for (int64_t i = c->lo; i < c->hi; i++) c->body(i, c->ctx);
  return NULL;
}
void merlin_hal_parallel_for(int64_t lo, int64_t hi, merlin_hal_body_fn body, void *ctx,
                             int n_workers) {
  int64_t n = hi - lo;
  if (n <= 0) return;
  if (n_workers <= 1 || n == 1) { for (int64_t i = lo; i < hi; i++) body(i, ctx); return; }
  if ((int64_t)n_workers > n) n_workers = (int)n;
  pthread_t th[64];
  chunk_t ck[64];
  if (n_workers > 64) n_workers = 64;
  int64_t per = (n + n_workers - 1) / n_workers;
  int spawned = 0;
  for (int w = 0; w < n_workers; w++) {
    int64_t a = lo + (int64_t)w * per, b = a + per;
    if (a >= hi) break;
    if (b > hi) b = hi;
    ck[w] = (chunk_t){a, b, body, ctx};
    if (pthread_create(&th[w], NULL, worker, &ck[w]) != 0) { worker(&ck[w]); continue; }
    spawned++;
  }
  for (int w = 0; w < spawned; w++) pthread_join(th[w], NULL);
}

/* SIMT launch on a host without SIMT: lower to a serial walk over the flattened grid×block so the
 * engine is uniform (correctness; not parallel). Real SIMT targets override in hal_simt.c. */
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

uint32_t merlin_hal_capabilities(void) {
  return MERLIN_CAP_SCALAR | MERLIN_CAP_RVV | MERLIN_CAP_THREADS;
}
int merlin_hal_has(uint32_t cap) { return (merlin_hal_capabilities() & cap) == cap; }
