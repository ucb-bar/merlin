/* Merlin HTIF console/exit for spike bare-metal runs. */
#ifndef MERLIN_BAREMETAL_HTIF_H
#define MERLIN_BAREMETAL_HTIF_H

void htif_putc(char c);
void htif_puts(const char *s);
void htif_putd(long v);          /* signed decimal */
void htif_exit(int code) __attribute__((noreturn));

#endif
