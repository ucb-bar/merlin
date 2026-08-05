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
void htif_exit(int code) __attribute__((noreturn));

#endif
