/* Minimal freestanding libc for the Merlin bare-metal harness.
 *
 * Even with -ffreestanding, GCC idiom recognition may lower copy/clear loops to
 * memcpy/memset calls (e.g. an empty COMMIT epilogue becomes a plain copy loop),
 * so these must exist.
 */
#include <stddef.h>
#include <stdint.h>

void *memcpy(void *dst, const void *src, size_t n) {
  uint8_t *d = dst;
  const uint8_t *s = src;
  while (n--)
    *d++ = *s++;
  return dst;
}

void *memset(void *dst, int c, size_t n) {
  uint8_t *d = dst;
  while (n--)
    *d++ = (uint8_t)c;
  return dst;
}

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
