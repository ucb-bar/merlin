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

#include "htif.h"

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

/* Exhaustion FAILS LOUDLY rather than returning NULL.
 *
 * Nothing in the generated model checks a malloc result -- bufferization emits the allocation and then
 * stores through it -- so a NULL return does not stop the run, it defers the failure to the first store,
 * which surfaces as a store access fault at address 0 somewhere deep in `forward` with no indication
 * that an allocation ever failed. That is a materially harder bug than the one that actually happened:
 * it was investigated as descriptor corruption for a long time before `mtval == 0` identified it.
 *
 * Reporting the request and the shortfall turns it into a one-line diagnosis. Sizes and pointers go
 * through the console's UNSIGNED printer: a corrupt length is exactly the case where the signed one
 * reports a negative number, which reads as a different bug than the one that happened. */
static size_t largest = 0;
static unsigned long long seq = 0, largest_seq = 0;

static void arena_exhausted(const char *why, size_t n, uintptr_t align) {
  htif_puts("\nFATAL: model arena -- ");
  htif_puts(why);
  htif_puts("\n  requested ");
  htif_puthex((unsigned long long)n);
  htif_puts(" bytes (align ");
  htif_putd((long)align);
  htif_puts(")\n  arena ");
  htif_puthex((unsigned long long)arena_base());
  htif_puts(" .. ");
  htif_puthex((unsigned long long)arena_end());
  htif_puts(", brk ");
  htif_puthex((unsigned long long)brk);
  htif_puts("\n  largest prior request ");
  htif_puthex((unsigned long long)largest);
  htif_puts(" at call #");
  htif_putd((long)largest_seq);
  htif_puts(" of ");
  htif_putd((long)seq);
  htif_putc('\n');
  htif_exit(0x900);
}

/* An allocation LARGER THAN THE WHOLE ARENA can never be served, and serving it is not the failure --
 * believing it is. `brk` is a bump pointer, so one request with a corrupt length carries it past the
 * arena and every later allocation fails; the report then names whichever innocent allocation happened
 * to be next, and the number it prints as "used" is the corrupt length. MEASURED on deepjscc: the
 * console blamed a 320-byte request and reported 29.8 TB used, which is not a size any model asks for.
 * Rejecting the impossible request AT the call that makes it names the real one. */
static void *bump(size_t n, uintptr_t align) {
  if (!brk) brk = arena_base();
  ++seq;
  if (n > (size_t)(uintptr_t)MERLIN_ARENA_SIZE_BYTES)
    arena_exhausted("request exceeds the whole arena (corrupt length, not a sizing failure)", n, align);
  uintptr_t p = (brk + (align - 1)) & ~(align - 1);
  uintptr_t next = p + n;
  if (next > arena_end())
    arena_exhausted("exhausted -- raise it with arena_mb=", n, align);   /* noreturn */
  if (n > largest) { largest = n; largest_seq = seq; }
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
