/* A minimal printf/exit/abort for code linked into the bare-metal model image.
 *
 * The whole-model image has no libc stdio, while code it links from a target's own software
 * environment (a vendor library header, a generated dispatch file) prints its reports and its failures
 * with printf and stops with exit. This implements exactly the conversions that code uses -- %d %i %u
 * %x %s %c %% with the l/ll/z length modifiers -- over the console ABI (htif.h). A conversion it does
 * not know is printed as written, never silently dropped, so an unexpected one shows up in the log.
 */
#include <stdarg.h>
#include <stddef.h>

#include "htif.h"
#ifdef MERLIN_EFFECT_OBSERVATION
#include "effect_observation.h"
#include "fixed_decimal.h"
#include <string.h>
#define htif_putc merlin_effect_putc
#define htif_puts merlin_effect_puts
#endif

static void put_unsigned(unsigned long long v, unsigned base) {
  char digits[24];
  int n = 0;
  do {
    unsigned d = (unsigned)(v % base);
    digits[n++] = (char)(d < 10 ? '0' + d : 'a' + d - 10);
    v /= base;
  } while (v);
  while (n) htif_putc(digits[--n]);
}

int vprintf(const char *fmt, va_list ap) {
  for (const char *p = fmt; *p; p++) {
    if (*p != '%') {
      htif_putc(*p);
      continue;
    }
    const char *start = p++;
    int longs = 0;
    while (*p == 'l' || *p == 'z') {
      longs += (*p == 'z') ? 2 : 1;
      p++;
    }
    switch (*p) {
      case 'd':
      case 'i': {
        long long v = longs >= 2 ? va_arg(ap, long long) : longs == 1 ? va_arg(ap, long) : va_arg(ap, int);
        if (v < 0) {
          htif_putc('-');
          put_unsigned((unsigned long long)(-(v + 1)) + 1ULL, 10);
        } else {
          put_unsigned((unsigned long long)v, 10);
        }
        break;
      }
      case 'u':
      case 'x': {
        unsigned long long v = longs >= 2   ? va_arg(ap, unsigned long long)
                               : longs == 1 ? va_arg(ap, unsigned long)
                                            : va_arg(ap, unsigned int);
        put_unsigned(v, *p == 'x' ? 16 : 10);
        break;
      }
      case 's': {
        const char *s = va_arg(ap, const char *);
        htif_puts(s ? s : "(null)");
        break;
      }
      case 'c':
        htif_putc((char)va_arg(ap, int));
        break;
#ifdef MERLIN_EFFECT_OBSERVATION
      case 'f': {
        double value = va_arg(ap, double);
        uint64_t bits;
        char text[320];
        memcpy(&bits, &value, sizeof(bits));
        merlin_binary64_fixed6(text, bits);
        htif_puts(text);
        break;
      }
#endif
      case '%':
        htif_putc('%');
        break;
      default:
#ifdef MERLIN_EFFECT_OBSERVATION
        merlin_effect_format_error();
#else
        for (const char *q = start; q <= p && *q; q++) htif_putc(*q);
        if (!*p) return 0;
#endif
        break;
    }
  }
  return 0;
}

int printf(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vprintf(fmt, ap);
  va_end(ap);
  return 0;
}

#ifndef MERLIN_PRINTF_FORMAT_ONLY
int puts(const char *s) {
  htif_puts(s);
  htif_putc('\n');
  return 0;
}
#endif

int putchar(int c) {
  htif_putc((char)c);
  return c;
}

#ifndef MERLIN_PRINTF_FORMAT_ONLY
void exit(int code) { htif_exit(code); }

void abort(void) { htif_exit(134); }
#endif
