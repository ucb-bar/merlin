/* Preserve arbitrary diagnostic bytes separately from driver/result framing.
 * The caller supplies the input-readback callback; no expected values enter.
 */
#include <stddef.h>
#include <stdint.h>
#include "htif.h"
#include "effect_observation.h"
#ifndef MERLIN_EFFECT_BYTES
#error "the caller must select a bounded diagnostic byte capacity"
#endif
static unsigned char source_stdout[MERLIN_EFFECT_BYTES];
static size_t used;
static int active;
static int started;

static void decimal(size_t value) {
  char digits[24];
  unsigned n = 0;
  do { digits[n++] = (char)('0' + value % 10); value /= 10; } while (value);
  while (n) htif_putc(digits[--n]);
}

static void invalid(const char *reason) __attribute__((noreturn));
static void invalid(const char *reason) {
  htif_puts("MERLIN_EFFECT_ERROR v1 "); htif_puts(reason); htif_putc('\n');
  htif_exit(125);
}

void merlin_effect_begin(void) {
  if (started) invalid("repeated-invocation");
  started = 1;
  active = 1;
  htif_puts("MERLIN_EFFECT_BEGIN v1\n");
}

void merlin_effect_putc(char value) {
  if (!active) invalid("stdio-outside-invocation");
  if (used == sizeof(source_stdout)) invalid("diagnostic-capacity");
  source_stdout[used++] = (unsigned char)value;
}

void merlin_effect_puts(const char *value) {
  while (*value) merlin_effect_putc(*value++);
}

static void finish(const char *outcome) {
  if (!active) invalid("outcome-outside-invocation");
  active = 0;
  htif_puts("OUT_BIN_BEGIN v1 source_stdout 1 "); decimal(used);
  htif_puts(" 1 u "); decimal(used); htif_putc('\n');
  uint64_t hash = 0xcbf29ce484222325ULL;
  htif_line_flush(0);
  for (size_t i = 0; i < used; i++) {
    htif_putc((char)source_stdout[i]);
    hash = (hash ^ source_stdout[i]) * 0x100000001b3ULL;
  }
  htif_line_flush(1);
  htif_puts("OUT_BIN_END v1 ");
  for (int shift = 60; shift >= 0; shift -= 4)
    htif_putc("0123456789abcdef"[(hash >> shift) & 15]);
  htif_puts("\nMERLIN_EFFECT_END v1 "); htif_puts(outcome); htif_putc('\n');
}

void merlin_effect_return(void) { finish("return"); }

void merlin_effect_abort(void) {
  finish("assertion_failure");
  merlin_effect_failure_inputs();
  htif_exit(134);
}

void merlin_effect_format_error(void) { invalid("unsupported-format"); }
