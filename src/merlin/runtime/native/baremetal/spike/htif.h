/* Merlin bare-metal console/exit ABI.
 *
 * These five symbols ARE the harness console interface; HTIF is one backend for them (`htif.c`, for
 * hosted substrates: spike, FireSim, uart_tsi) and a chip's own UART is another (`console_uart.c`,
 * for real silicon). The `htif_` prefix is historical -- it predates there being a choice -- and is
 * kept because generated code emits calls to these names. Exactly one backend is linked per image.
 */
#ifndef MERLIN_BAREMETAL_HTIF_H
#define MERLIN_BAREMETAL_HTIF_H

/* Bring the console up. MUST be called before the first character: on a real chip the UART is not
 * usable until its clocks and baud divisor are programmed, and printing to it before that hangs the
 * core. A no-op for host-assisted backends, so callers can invoke it unconditionally. */
void console_init(void);

void htif_putc(char c);
void htif_puts(const char *s);
void htif_putd(long v);          /* signed decimal */
/* Unsigned hex, `0x`-prefixed, no leading zeros. Sizes and addresses are unsigned, and putting one
 * through the signed printer reports a large pointer or a corrupt length as a negative number --
 * which reads as a different kind of bug than the one that happened. */
void htif_puthex(unsigned long long v);
void htif_exit(int code) __attribute__((noreturn));

/* Suspend the flush-per-newline policy for a BULK dump, then restore it (which flushes what is held).
 *
 * The default -- flush whenever a line completes -- exists so a log appears while the model is still
 * running rather than only at exit. It is the right default and stays the default. But it costs one
 * host round-trip per LINE, and a dump of thousands of short lines then pays that per ~25 bytes instead
 * of per 256-byte buffer. Measured on FireSim: the per-op profiler's dump crawled at ~6.6 B/s where the
 * single long OUT line ran at ~100 B/s, which would have made a whole-model profiled run take about
 * five hours of FPGA instead of minutes.
 *
 * A backend with no batching (a chip UART writing a byte at a time to MMIO) implements this as a no-op.
 */
void htif_line_flush(int enable);

#endif
