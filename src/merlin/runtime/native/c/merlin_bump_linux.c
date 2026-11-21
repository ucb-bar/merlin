/* Linux bump allocator that OVERRIDES glibc malloc/free/calloc/realloc when linked into the K1
 * binary (env-gated build flag). This is a MEASUREMENT PROXY for the lean runtime's static arena:
 * the current whole-model path does ~4391 per-op malloc/free per inference (one per tensor.empty);
 * replacing them with O(1) bump-from-a-preallocated-arena (free = no-op) isolates how much of the
 * dispatch/runtime overhead is the per-op allocator itself — i.e. the cost the lean arena removes.
 * It is NOT the lean runtime; it just lets us measure the allocator's share of the wall before
 * building the full program+replay engine. cos is unaffected (same bytes, just a cheaper allocator).
 */
#include <stddef.h>
#include <stdint.h>
#include <sys/mman.h>

/* one big arena, mmap'd lazily on first use; bump pointer; free is a no-op (whole-inference lifetime). */
#ifndef MERLIN_BUMP_ARENA_BYTES
#define MERLIN_BUMP_ARENA_BYTES (4ULL * 1024 * 1024 * 1024)   /* 4 GiB virtual; pages fault in lazily */
#endif

static uint8_t *g_base = 0;
static uint8_t *g_brk = 0;
static uint8_t *g_end = 0;

static void bump_init(void) {
  void *p = mmap(0, (size_t)MERLIN_BUMP_ARENA_BYTES, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  if (p == MAP_FAILED) { g_base = g_brk = g_end = 0; return; }
  g_base = g_brk = (uint8_t *)p;
  g_end = g_base + (size_t)MERLIN_BUMP_ARENA_BYTES;
}

static void *bump(size_t n, size_t align) {
  if (!g_base) bump_init();
  uintptr_t p = ((uintptr_t)g_brk + (align - 1)) & ~(uintptr_t)(align - 1);
  uint8_t *next = (uint8_t *)p + n;
  if (!g_base || next > g_end) return 0;   /* arena exhausted -> null (surfaces as a fault, honest) */
  g_brk = next;
  return (void *)p;
}

void *malloc(size_t n) { return bump(n, 64); }
void *calloc(size_t nm, size_t sz) {
  size_t n = nm * sz;
  uint8_t *p = (uint8_t *)bump(n, 64);
  if (p) for (size_t i = 0; i < n; i++) p[i] = 0;   /* MAP_ANONYMOUS is already zero, but be safe */
  return p;
}
void *aligned_alloc(size_t align, size_t sz) { return bump(sz, align < 64 ? 64 : align); }
void *realloc(void *old, size_t n) {
  /* bump can't grow in place; allocate fresh (callers in the lowered model rarely realloc). */
  (void)old;
  return bump(n, 64);
}
void free(void *p) { (void)p; }   /* whole-inference lifetime; reclaimed by process exit */
