/* Merlin HTIF console/exit for bare-metal runs on a HOSTED substrate.
 *
 * `tohost`/`fromhost` live in a `.htif` section that the host locates by scanning the ELF. There are
 * two ways to put a character through HTIF, and which one a host implements is not a detail:
 *
 *   1. the CHARACTER DEVICE — `tohost = (dev 1 << 56) | (cmd 1 << 48) | c`, one round-trip per char;
 *   2. the SYSCALL PROXY — `tohost = &block`, where block is `{SYS_write, fd, ptr, len}` and the host
 *      copies the whole buffer out in one round-trip.
 *
 * This file used to use (1). spike's fesvr implements both, so it worked here — and the loader for a
 * real board may implement only (2). Measured: `pyuartsi --fesvr` reads `tohost` as a request POINTER
 * and dispatches on the syscall id, so a character-device word (`0x0101_0000_0000_00xx`) is read as a
 * pointer, dereferenced at a nonexistent address, and reported as an invalid syscall. The image
 * produces no readable output and looks like it hung. That is what happened to a probe whose entire
 * job was to print four CSR values, on a chip where the same protocol was carrying Zephyr's console
 * output perfectly — because Zephyr's HTIF driver uses (2) under CONFIG_UART_HTIF_SYSCALL_PRINT.
 *
 * So: use (2). It is what libgloss/htif_nano's `_write` does, it is what every fesvr implements, and
 * it is also an order of magnitude faster — one round-trip per BUFFER instead of per character, which
 * matters when the protocol dumps thousands of values over a serial link.
 */
#include <stdint.h>

#include "htif.h"

volatile uint64_t tohost __attribute__((section(".htif")));
volatile uint64_t fromhost __attribute__((section(".htif")));

/* fesvr syscall numbers (a RISC-V newlib/proxy-kernel ABI, not the host's). */
#define FESVR_SYS_write 64

/* Output is buffered so one host round-trip carries many characters. 256 bytes matches the size
 * Zephyr's HTIF driver uses for the same reason. */
static char htif_buf[256];
static unsigned htif_len;

/* The request block. Static rather than on the stack so its address is stable and 8-byte aligned:
 * the host reads 4 doublewords from wherever `tohost` points, and an under-aligned or transient
 * address is read as garbage. */
static volatile uint64_t htif_req[4] __attribute__((aligned(8)));

static void htif_flush(void) {
  if (htif_len == 0) {
    return;
  }
  htif_req[0] = FESVR_SYS_write;
  htif_req[1] = 1;                                  /* fd 1 = stdout. fd 0 is stdin: a host may write
                                                     * it somewhere useless, or refuse outright. */
  htif_req[2] = (uint64_t)(uintptr_t)htif_buf;
  htif_req[3] = (uint64_t)htif_len;
  /* Order the buffer and the block ahead of the doorbell: the host reads them as soon as it observes
   * tohost, and a store still sitting in a write buffer would be read as stale bytes. */
  __asm__ volatile("fence rw, rw" ::: "memory");
  /* Device 0, command 0 -> the word IS the pointer to the request block. */
  tohost = (uint64_t)(uintptr_t)htif_req;
  /* The host clears tohost and raises fromhost when the request has been serviced; we clear fromhost.
   * (Same handshake as the vendor Zephyr driver, so a host that drives one drives the other.) */
  while (fromhost == 0)
    ;
  fromhost = 0;
  __asm__ volatile("fence rw, rw" ::: "memory");
  htif_len = 0;
}

void console_init(void) {
  /* Nothing to bring up: the host is already listening on tohost/fromhost before the core starts.
   * Defined so callers can invoke it unconditionally and the console backend stays a link-time
   * choice. NOTE this backend REQUIRES that host -- with nothing servicing tohost (i.e. on real
   * silicon with no debugger attached) the first flush spins forever. Boards without a host must link
   * `console_uart.c` instead; see boards.Board.console. */
  htif_len = 0;
}

void htif_putc(char c) {
  htif_buf[htif_len++] = c;
  /* Flush on a full buffer, and on a newline so a line is readable as soon as it is complete — a log
   * that only appears at exit is indistinguishable from a hang while the model is still running. */
  if (htif_len == sizeof(htif_buf) || c == '\n') {
    htif_flush();
  }
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
  /* Flush FIRST: the exit request is terminal, and anything still buffered would be lost — including
   * the DONE line the host parser gates on. */
  htif_flush();
  tohost = ((uint64_t)(uint32_t)code << 1) | 1;
  for (;;)
    ;
}
