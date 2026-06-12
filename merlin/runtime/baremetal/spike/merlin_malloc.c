/* Bump allocator for the bare-metal model runtime.
 *
 * MLIR bufferization emits malloc/free for activation buffers. We serve them from a
 * fixed, absolute-addressed arena (default 0xC0000000), NOT a linker-reserved symbol:
 * with multi-GB models the arena sits >2GB from the code, beyond `-mcmodel=medany`'s
 * PC-relative reach, so it must be addressed by a literal constant (compiled to `li`).
 * The arena lives in spike's -m memory; it is not a loaded ELF section.
 *
 * free() is a no-op; merlin_arena_reset reclaims the whole arena between inferences.
 * calloc/aligned_alloc zero/align what they hand out. Scales by MERLIN_ARENA_SIZE_BYTES.
 */
#include <stddef.h>
#include <stdint.h>

#ifndef MERLIN_ARENA_BASE_ADDR
#define MERLIN_ARENA_BASE_ADDR 0xC0000000ULL
#endif
#ifndef MERLIN_ARENA_SIZE_BYTES
#define MERLIN_ARENA_SIZE_BYTES 0x10000000ULL /* 256 MB default */
#endif

static uintptr_t brk = 0;

static inline uintptr_t arena_base(void) { return (uintptr_t)MERLIN_ARENA_BASE_ADDR; }
static inline uintptr_t arena_end(void) {
  return (uintptr_t)MERLIN_ARENA_BASE_ADDR + (uintptr_t)MERLIN_ARENA_SIZE_BYTES;
}

void merlin_arena_reset(void) { brk = arena_base(); }

static void *bump(size_t n, uintptr_t align) {
  if (!brk) brk = arena_base();
  uintptr_t p = (brk + (align - 1)) & ~(align - 1);
  uintptr_t next = p + n;
  if (next > arena_end())
    return 0; /* out of arena — grow MERLIN_ARENA_SIZE_BYTES */
  brk = next;
  return (void *)p;
}

void *malloc(size_t n) { return bump(n, 64); }

void *aligned_alloc(size_t alignment, size_t size) {
  return bump(size, alignment < 64 ? 64 : alignment);
}

void *calloc(size_t nmemb, size_t size) {
  size_t n = nmemb * size;
  char *p = (char *)bump(n, 64);
  if (p)
    for (size_t i = 0; i < n; i++)
      p[i] = 0;
  return p;
}

void free(void *p) { (void)p; }
