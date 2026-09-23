// Bare-metal support for OURS' standalone ceiling driver: a bump allocator so
// the lowered model's `tensor.empty`/bufferization `malloc`s (and merlin_run's
// large-arity heap path, unused at n_args=3) link under -nostdlib, where no libc
// heap exists. The arena is sized for the kernel-shape GEMM workloads (<=128^3
// fp32: a few hundred KB of activation/intermediate buffers). Allocation happens
// OUTSIDE the timed compute region in practice (forward() allocates its result/
// intermediates as it runs — but the bump allocator is O(1) and the dominant cost
// remains the matmul; this matches the experts whose pack is also hoisted).
//
// free() is a no-op (bump allocator); fine for a single forward() invocation.

#include <stddef.h>
#include <stdint.h>

#define MERLIN_BM_ARENA_BYTES (16 * 1024 * 1024)   // 16 MB: ample for <=128^3 fp32

static unsigned char merlin_bm_arena[MERLIN_BM_ARENA_BYTES] __attribute__((aligned(64)));
static size_t merlin_bm_off = 0;

void *malloc(size_t n) {
  // 64-byte align every allocation (RVV-friendly; MLIR expects aligned memref data).
  size_t a = (merlin_bm_off + 63u) & ~((size_t)63u);
  if (a + n > MERLIN_BM_ARENA_BYTES) return (void *)0;   // OOM -> null -> surfaces as a fault
  merlin_bm_off = a + n;
  return &merlin_bm_arena[a];
}

void *calloc(size_t nmemb, size_t size) {
  size_t n = nmemb * size;
  unsigned char *p = (unsigned char *)malloc(n);
  if (p) for (size_t i = 0; i < n; i++) p[i] = 0;
  return p;
}

void *aligned_alloc(size_t alignment, size_t n) {
  (void)alignment;   // malloc already 64-byte aligns
  return malloc(n);
}

void free(void *p) { (void)p; }   // bump allocator: no per-block free
