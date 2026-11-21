/* Merlin console over a chip's own UART -- the backend for runs on real silicon.
 *
 * This implements the same four-symbol console ABI as `htif.c`, so `model_main.c` and the probe are
 * unchanged and the choice of channel is a link-time decision (see `boards.Board.console`).
 *
 * Why a second backend exists at all. HTIF is **host-assisted**: the program writes `tohost` and
 * spins until a host (fesvr, inside spike / FireSim / uart_tsi) clears it. Every substrate we can
 * test on provides that host, so HTIF is correct there -- and on bare silicon, where nothing clears
 * `tohost`, the SECOND character spins forever inside the first print, before any model work runs.
 * That looks exactly like a core that never booted, and it is what happened to the first binaries
 * shipped to a board we cannot reach.
 *
 * The chip's own console needs bring-up before its first character, and the vendor SDK says so in as
 * many words: their `init_test()` "MUST run before any printf on silicon -- the console UART is not
 * usable until then (a printf to it would hang the core)". `console_init()` below is that sequence,
 * in their order:
 *
 *   1. enable RX/TX, set the stop bits, and program the baud divisor against the RESET clock;
 *   2. park every clock domain on the slow source, program the PLL, switch the domains to it;
 *   3. **re-program the baud divisor**, because step 2 changed the clock it is relative to.
 *
 * Step 3 is the subtle one: skip it and the UART keeps a divisor for a clock that is now 10x faster,
 * so the console emits garbage rather than nothing -- a failure that reads as a corrupt program.
 *
 * Not one address, offset, bit position or clock rate is written down here. They arrive as macros
 * derived from the target SDK's own headers (`runtime/sdk_facts.py`), because a literal MMIO address
 * in a shared harness is a fact about one tapeout that would be silently wrong for the next chip.
 * If a macro is missing the build FAILS rather than assuming a default: a wrong console address
 * produces no output, which is the one failure mode nobody on the far end can debug.
 */
#include <stdint.h>

#if !defined(MERLIN_UART_BASE) || !defined(MERLIN_UART_TXDATA_OFF) || \
    !defined(MERLIN_UART_TXCTRL_OFF) || !defined(MERLIN_UART_RXCTRL_OFF) || \
    !defined(MERLIN_UART_DIV_OFF) || !defined(MERLIN_UART_TX_FULL_BIT) || \
    !defined(MERLIN_UART_TXEN_BIT) || !defined(MERLIN_UART_RXEN_BIT) || \
    !defined(MERLIN_UART_NSTOP_BIT) || !defined(MERLIN_UART_STOPBITS) || \
    !defined(MERLIN_UART_BAUD) || !defined(MERLIN_SYS_CLK_HZ)
#error "console_uart.c needs the UART facts derived from the target SDK (runtime/sdk_facts.py)"
#endif

#define REG32(addr) (*(volatile uint32_t *)(uintptr_t)(addr))
#define UART_REG(off) REG32(MERLIN_UART_BASE + (off))

/* Baud divisor for a given clock: f_baud = f_clk / (div + 1), as the vendor driver computes it. */
static uint32_t baud_div(uint64_t clk_hz) {
  return (uint32_t)((clk_hz / (uint64_t)MERLIN_UART_BAUD) - 1u);
}

void console_init(void) {
  /* 1. Enable the transmitter and receiver and set the stop bits. Written as a read-modify-write of
   *    the control registers so bits this harness does not manage (interrupt watermarks, FIFO
   *    counts) survive -- clobbering the whole word would silently change the framing. */
  uint32_t txctrl = UART_REG(MERLIN_UART_TXCTRL_OFF);
  txctrl |= (1u << MERLIN_UART_TXEN_BIT);
#if MERLIN_UART_STOPBITS == 2
  txctrl |= (1u << MERLIN_UART_NSTOP_BIT);
#else
  txctrl &= ~(1u << MERLIN_UART_NSTOP_BIT);
#endif
  UART_REG(MERLIN_UART_TXCTRL_OFF) = txctrl;
  UART_REG(MERLIN_UART_RXCTRL_OFF) =
      UART_REG(MERLIN_UART_RXCTRL_OFF) | (1u << MERLIN_UART_RXEN_BIT);

  /* The divisor for the clock we are running at RIGHT NOW. If the PLL is left alone this is the
   * final value, and the console works at the chip's reset frequency. */
  UART_REG(MERLIN_UART_DIV_OFF) = baud_div((uint64_t)MERLIN_SYS_CLK_HZ);

#ifdef MERLIN_CHIP_FREQ_HZ
  /* 2. Raise the PLL, exactly as the SDK's init_test() does: park the clock domains on the slow
   *    source first, program the PLL, then move the domains onto it. Reordering this switches a
   *    running core onto an unlocked PLL. */
  uint32_t ratio = (uint32_t)((uint64_t)MERLIN_CHIP_FREQ_HZ / (uint64_t)MERLIN_SYS_CLK_HZ);
  if (ratio == 0u) {
    ratio = 1u;
  }
  for (int i = 0; i < MERLIN_CLKSEL_N; i++) {
    REG32(MERLIN_CLKSEL_BASE + 4 * i) = (uint32_t)MERLIN_CLKSEL_SLOW;
  }
  REG32(MERLIN_PLL_BASE + MERLIN_PLL_PLLEN_OFF) = 0u;
  REG32(MERLIN_PLL_BASE + MERLIN_PLL_MDIV_RATIO_OFF) = 1u;
  REG32(MERLIN_PLL_BASE + MERLIN_PLL_RATIO_OFF) = ratio;
  REG32(MERLIN_PLL_BASE + MERLIN_PLL_FRACTION_OFF) = 0u;
  REG32(MERLIN_PLL_BASE + MERLIN_PLL_ZDIV0_RATIO_OFF) = 1u;
  REG32(MERLIN_PLL_BASE + MERLIN_PLL_ZDIV1_RATIO_OFF) = 1u;
  REG32(MERLIN_PLL_BASE + MERLIN_PLL_LDO_ENABLE_OFF) = 1u;
  REG32(MERLIN_PLL_BASE + MERLIN_PLL_PLLEN_OFF) = 1u;
  REG32(MERLIN_PLL_BASE + MERLIN_PLL_POWERGOOD_VNN_OFF) = 1u;
  REG32(MERLIN_PLL_BASE + MERLIN_PLL_PLLFWEN_B_OFF) = 1u;
  for (int i = 0; i < MERLIN_CLKSEL_N; i++) {
    REG32(MERLIN_CLKSEL_BASE + 4 * i) = (uint32_t)MERLIN_CLKSEL_PLL;
  }

  /* 3. The divisor is relative to a clock that just changed. Without this the console emits garbage
   *    instead of nothing, which reads as a corrupt program rather than a misconfigured UART. */
  UART_REG(MERLIN_UART_DIV_OFF) = baud_div((uint64_t)MERLIN_CHIP_FREQ_HZ);
#endif
}

void htif_putc(char c) {
  while (UART_REG(MERLIN_UART_TXDATA_OFF) & (1u << MERLIN_UART_TX_FULL_BIT))
    ;
  UART_REG(MERLIN_UART_TXDATA_OFF) = (uint32_t)(uint8_t)c;
}

void htif_line_flush(int enable) {
  /* Nothing to hold back: this backend writes each byte straight to the UART's TX register, so there is
   * no buffer whose flush policy could be changed. Defined anyway because it is part of the console ABI
   * and exactly one backend is linked per image -- a caller must not have to know which. */
  (void)enable;
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

void htif_puthex(unsigned long long v) {
  char buf[16];
  int i = 0;
  htif_putc('0');
  htif_putc('x');
  do {
    buf[i++] = "0123456789abcdef"[v & 0xf];
    v >>= 4;
  } while (v);
  while (i)
    htif_putc(buf[--i]);
}

void htif_exit(int code) {
  /* There is no host to tell, and no reset to perform that the operator has not already got a button
   * for. Report the code on the console the caller is already reading, then idle -- a `wfi` loop
   * rather than a spin so the chip is not burning power while someone reads the log. */
  htif_puts("EXIT ");
  htif_putd((long)code);
  htif_putc('\n');
  for (;;) {
    __asm__ volatile("wfi");
  }
}
