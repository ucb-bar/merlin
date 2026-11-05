/* Dense sweep of the portable transcendentals: one FNV-1a hash of the result bits per function.
 *
 * Linked against the MLIR runtime object (mlir_runtime.c, which defines merlin_erff & co.) exactly as
 * a host build or a device image links it, so two builds agree on every hash iff they compute the
 * same bits on every swept input. Built for the host with SWEEP_HOST (printf), otherwise for the
 * bare-metal spike harness (htif).
 *
 * The sweep: every 4099th bit pattern of all 2^32 (every exponent, both signs, subnormals, infinities
 * and NaNs), plus every 1021st float of magnitude below 8 in each sign (the range GELU, softmax and
 * sigmoid arguments occupy). */
#include <stdint.h>

float merlin_erff(float);
float merlin_expf(float);
float merlin_tanhf(float);
float merlin_logf(float);

#ifdef SWEEP_HOST
#include <stdio.h>
static void emit(const char *name, uint64_t h) { printf("HASH %s %016llx\n", name, (unsigned long long)h); }
#else
#include "htif.h"
static void emit(const char *name, uint64_t h) {
  htif_puts("HASH ");
  htif_puts(name);
  htif_putc(' ');
  htif_puthex(h);
  htif_putc('\n');
}
#endif

static uint64_t fold(uint64_t h, float y) {
  uint32_t b;
  __builtin_memcpy(&b, &y, 4);
  for (int k = 0; k < 4; ++k) {
    h ^= (b >> (8 * k)) & 0xffu;
    h *= 1099511628211ull;
  }
  return h;
}

static uint64_t sweep(float (*fn)(float)) {
  uint64_t h = 1469598103934665603ull;
  for (uint64_t i = 0; i < 1047800u; ++i) {
    uint32_t xb = (uint32_t)(i * 4099u);
    float x;
    __builtin_memcpy(&x, &xb, 4);
    h = fold(h, fn(x));
  }
  for (uint32_t sign = 0; sign < 2; ++sign) {
    for (uint32_t xb = 0; xb < 0x41000000u; xb += 1021u) { /* |x| < 8 */
      uint32_t sb = xb | (sign << 31);
      float x;
      __builtin_memcpy(&x, &sb, 4);
      h = fold(h, fn(x));
    }
  }
  return h;
}

int main(void) {
#ifndef SWEEP_HOST
  console_init();
#endif
  emit("erf", sweep(merlin_erff));
  emit("exp", sweep(merlin_expf));
  emit("tanh", sweep(merlin_tanhf));
  emit("log", sweep(merlin_logf));
#ifndef SWEEP_HOST
  htif_puts("DONE\n");
  htif_exit(0);
#endif
  return 0;
}
