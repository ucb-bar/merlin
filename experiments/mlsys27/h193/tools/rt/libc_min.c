/* Minimal freestanding libc for the Merlin bare-metal harness.
 *
 * Even with -ffreestanding, GCC idiom recognition may lower copy/clear loops to
 * memcpy/memset calls (e.g. an empty COMMIT epilogue becomes a plain copy loop),
 * so these must exist.
 */
#include <stddef.h>
#include <stdint.h>
#include "htif.h"

/* Upstream MLIR assertion lowering uses these libc symbols. Keep failure
 * paths linkable even when host optimization does not eliminate assertions.
 * This console ABI also supports the non-HTIF bare-metal backends. */
int puts(const char *s) {
  htif_puts(s);
  htif_putc('\n');
  return 0;
}

__attribute__((noreturn)) void abort(void) {
  htif_puts("FATAL: model assertion abort\n");
  htif_exit(0x901);
}

/* may_alias permits moving arbitrary tensor bytes through aligned word loads.
 * Misaligned pointers keep the byte path, avoiding alignment traps. */
typedef uintptr_t copy_word __attribute__((__may_alias__));

#ifdef MERLIN_WORD_MEMOPS
/* Word-wide copies and fills, for a program whose own code copies large tensors (a whole model's host
 * code bufferizes into memcpy/memset calls: MB per call). Opt-in, so every other harness build keeps the
 * byte loops below and its cycle counts. Scalar on purpose: the program's harts need not share an ISA.
 * Byte-identical results; the word path is taken only when both pointers share their 8-byte alignment.
 * Build with -fno-tree-loop-distribute-patterns, or GCC may turn these loops back into calls to
 * themselves. */
typedef uint64_t __attribute__((may_alias)) merlin_word_t;

void *memcpy(void *dst, const void *src, size_t n) {
  uint8_t *d = dst;
  const uint8_t *s = src;
  if ((((uintptr_t)d ^ (uintptr_t)s) & 7) == 0) {
    while (n && ((uintptr_t)d & 7)) {
      *d++ = *s++;
      n--;
    }
    merlin_word_t *dw = (merlin_word_t *)d;
    const merlin_word_t *sw = (const merlin_word_t *)s;
    for (; n >= 32; n -= 32, dw += 4, sw += 4) {
      merlin_word_t a = sw[0], b = sw[1], c = sw[2], e = sw[3];
      dw[0] = a;
      dw[1] = b;
      dw[2] = c;
      dw[3] = e;
    }
    for (; n >= 8; n -= 8)
      *dw++ = *sw++;
    d = (uint8_t *)dw;
    s = (const uint8_t *)sw;
  }
  while (n--)
    *d++ = *s++;
  return dst;
}

void *memset(void *dst, int c, size_t n) {
  uint8_t *d = dst;
  while (n && ((uintptr_t)d & 7)) {
    *d++ = (uint8_t)c;
    n--;
  }
  merlin_word_t w = (uint8_t)c;
  w |= w << 8;
  w |= w << 16;
  w |= w << 32;
  merlin_word_t *dw = (merlin_word_t *)d;
  for (; n >= 32; n -= 32, dw += 4) {
    dw[0] = w;
    dw[1] = w;
    dw[2] = w;
    dw[3] = w;
  }
  for (; n >= 8; n -= 8)
    *dw++ = w;
  d = (uint8_t *)dw;
  while (n--)
    *d++ = (uint8_t)c;
  return dst;
}
#else
void *memcpy(void *dst, const void *src, size_t n) {
  uint8_t *d = dst;
  const uint8_t *s = src;
  const size_t width = sizeof(copy_word);
  if ((uintptr_t)d % width == (uintptr_t)s % width) {
    while (n && (uintptr_t)d % width) {
      *d++ = *s++;
      --n;
    }
    while (n >= width) {
      *(copy_word *)d = *(const copy_word *)s;
      d += width;
      s += width;
      n -= width;
    }
  }
  while (n--)
    *d++ = *s++;
  return dst;
}

void *memset(void *dst, int c, size_t n) {
  uint8_t *d = dst;
  const size_t width = sizeof(copy_word);
  const copy_word pattern = (~(copy_word)0 / 255) * (uint8_t)c;
  while (n && (uintptr_t)d % width) {
    *d++ = (uint8_t)c;
    --n;
  }
  while (n >= width) {
    *(copy_word *)d = pattern;
    d += width;
    n -= width;
  }
  while (n--)
    *d++ = (uint8_t)c;
  return dst;
}
#endif

/* newlib libm (powf/sqrtf/...) calls __errno() for its errno pointer; provide one. */
static int _merlin_errno;
int *__errno(void) { return &_merlin_errno; }

void *memmove(void *dst, const void *src, size_t n) {
  uint8_t *d = dst;
  const uint8_t *s = src;
  if (d < s) {
    while (n--)
      *d++ = *s++;
  } else {
    d += n;
    s += n;
    while (n--)
      *--d = *--s;
  }
  return dst;
}
