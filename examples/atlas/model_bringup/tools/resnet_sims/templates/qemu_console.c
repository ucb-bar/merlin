/*
 * qemu_console.c - validation build only (-DRESNET50_QEMU=ON); never link this for F2.
 *
 * QEMU virt has no HTIF host, so glossy's HTIF write would wait forever for fromhost. This replaces
 * glossy's weak _write with the NS16550 UART that QEMU virt provides at 0x10000000.
 */
#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>

#define QEMU_NS16550_BASE 0x10000000UL

ssize_t _write(int fd, const void *ptr, size_t len) {
  (void)fd;
  volatile uint8_t *regs = (volatile uint8_t *)QEMU_NS16550_BASE;
  const uint8_t *bytes = (const uint8_t *)ptr;
  for (size_t i = 0; i < len; ++i) {
    while (!(regs[5] & 0x20)) {
    }
    regs[0] = bytes[i];
  }
  return (ssize_t)len;
}
