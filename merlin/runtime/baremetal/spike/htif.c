/* Merlin HTIF console/exit for spike bare-metal runs.
 *
 * tohost/fromhost are located by symbol name by spike's fesvr. Console output
 * uses the HTIF blocking character device (device 1, command 1); exit uses the
 * canonical (code << 1) | 1 protocol.
 */
#include <stdint.h>

#include "htif.h"

volatile uint64_t tohost __attribute__((section(".htif")));
volatile uint64_t fromhost __attribute__((section(".htif")));

void htif_putc(char c) {
  while (tohost)
    ;
  tohost = (1ULL << 56) | (1ULL << 48) | (uint8_t)c;
}

void htif_puts(const char *s) {
  while (*s)
    htif_putc(*s++);
}

void htif_putd(long v) {
  char buf[24];
  int i = 0;
  unsigned long u = (unsigned long)v;
  if (v < 0) {
    htif_putc('-');
    u = -(unsigned long)v;
  }
  do {
    buf[i++] = '0' + (u % 10);
    u /= 10;
  } while (u);
  while (i)
    htif_putc(buf[--i]);
}

void htif_exit(int code) {
  while (tohost)
    ;
  tohost = ((uint64_t)(uint32_t)code << 1) | 1;
  for (;;)
    ;
}
