/* Native observation only. Indirect volatile libm calls prevent pow(x,2) folding to x*x. */
#include <errno.h>
#include <fenv.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

static int last_errno, last_exceptions;
int ss_round_nearest(void) { return fesetround(FE_TONEAREST); }
int ss_last_errno(void) { return last_errno; }
int ss_last_exceptions(void) { return last_exceptions; }

#define PROBE(NAME, UINT, FLOAT, EXPRESSION) \
UINT NAME(UINT bits) { \
  FLOAT x, copy; volatile FLOAT result; UINT output; memcpy(&x, &bits, sizeof(x)); \
  errno = 0; feclearexcept(FE_ALL_EXCEPT); \
  result = (EXPRESSION); \
  last_errno = errno; last_exceptions = fetestexcept(FE_ALL_EXCEPT); \
  copy = result; memcpy(&output, &copy, sizeof(output)); return output; \
}

static float (*volatile call_powf)(float, float) = powf;
static double (*volatile call_pow)(double, double) = pow;
PROBE(ss_pow32, uint32_t, float, call_powf(x, 2.0f))
PROBE(ss_mul32, uint32_t, float, x * x)
PROBE(ss_pow64, uint64_t, double, call_pow(x, 2.0))
PROBE(ss_mul64, uint64_t, double, x * x)
