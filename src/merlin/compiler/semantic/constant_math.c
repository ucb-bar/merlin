/* Compiler-only positive-power bounds. MPFR/GMP remain external dependencies. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <limits.h>
#include <stdint.h>
#include <mpfr.h>

_Static_assert(CHAR_BIT == 8 && ULONG_MAX >= UINT64_C(0x1fffffffffffff), "requires 64-bit unsigned long");

unsigned merlin_math_abi(void) { return 1; }
unsigned merlin_math_tls(void) { return mpfr_buildopt_tls_p(); }
const char *merlin_math_version(unsigned which) {
  return which == 0 ? mpfr_get_version() : which == 1 ? gmp_version : MPFR_VERSION_STRING;
}
const char *merlin_math_dependency(unsigned which) {
  Dl_info info;
  const void *address = which == 0 ? (const void *)mpfr_pow : (const void *)__gmpz_init;
  return dladdr(address, &info) ? info.dli_fname : 0;
}

static void set_bits(mpfr_ptr out, uint64_t bits, unsigned width) {
  unsigned fraction = width == 32 ? 23 : 52, bias = width == 32 ? 127 : 1023;
  uint64_t mask = (UINT64_C(1) << fraction) - 1;
  unsigned exponent = (unsigned)((bits >> fraction) & (width == 32 ? 255u : 2047u));
  uint64_t significand = (bits & mask) | (exponent ? UINT64_C(1) << fraction : 0);
  mpfr_set_ui_2exp(out, (unsigned long)significand, (long)(exponent ? exponent : 1) - bias - fraction, MPFR_RNDN);
  if (bits >> (width - 1)) mpfr_neg(out, out, MPFR_RNDN);
}

/* Round an already bounded positive MPFR endpoint without host FP arithmetic. */
static uint64_t get_bits(mpfr_srcptr value, mpfr_ptr scratch, unsigned width) {
  unsigned fraction = width == 32 ? 23 : 52, bias = width == 32 ? 127 : 1023;
  uint64_t infinity = ((UINT64_C(1) << (width - fraction - 1)) - 1) << fraction;
  if (mpfr_inf_p(value)) return infinity;
  if (mpfr_zero_p(value)) return 0;
  mpfr_exp_t highest = mpfr_get_exp(value) - 1;
  long minimum = 1 - (long)bias - fraction;
  if (highest > (long)bias) return infinity;
  if (highest < minimum - 1) return 0;
  long quantum = highest - fraction > minimum ? highest - fraction : minimum;
  mpfr_mul_2si(scratch, value, -quantum, MPFR_RNDN); /* Exact binary shift. */
  uint64_t rounded = mpfr_get_ui(scratch, MPFR_RNDN);
  if (rounded == (UINT64_C(1) << (fraction + 1))) { rounded >>= 1; ++quantum; }
  if (rounded < (UINT64_C(1) << fraction)) return rounded;
  long exponent = quantum + fraction + bias;
  if (exponent >= (width == 32 ? 255 : 2047)) return infinity;
  return ((uint64_t)exponent << fraction) | (rounded - (UINT64_C(1) << fraction));
}

/* 0: matching rounded bounds; 1: inconclusive; negative: invalid/environment. */
int merlin_math_pow(unsigned width, uint64_t base, uint64_t exponent, unsigned precision,
                    uint64_t *lower_bits, uint64_t *upper_bits) {
  if ((width != 32 && width != 64) || precision < 64 || precision > 4096 ||
      !lower_bits || !upper_bits || (width == 32 && ((base | exponent) >> 32))) return -1;
  unsigned fraction = width == 32 ? 23 : 52;
  uint64_t sign = UINT64_C(1) << (width - 1), magnitude = sign - 1;
  uint64_t infinity = magnitude - ((UINT64_C(1) << fraction) - 1);
  if (base & sign || !(base & magnitude) || (base & magnitude) >= infinity ||
      (exponent & magnitude) >= infinity) return -2;
  if (mpfr_get_emin() > -4096 || mpfr_get_emax() < 4096) return -3;
  mpfr_flags_t flags = mpfr_flags_save();
  mpfr_t a, b, lower, upper, scratch;
  mpfr_inits2(precision, a, b, lower, upper, scratch, (mpfr_ptr)0);
  set_bits(a, base, width); set_bits(b, exponent, width);
  mpfr_pow(lower, a, b, MPFR_RNDD); mpfr_pow(upper, a, b, MPFR_RNDU);
  int status = -4;
  if (!mpfr_nan_p(lower) && !mpfr_nan_p(upper)) {
    *lower_bits = get_bits(lower, scratch, width);
    *upper_bits = get_bits(upper, scratch, width);
    status = *lower_bits == *upper_bits ? 0 : 1;
  }
  mpfr_clears(a, b, lower, upper, scratch, (mpfr_ptr)0);
  mpfr_flags_restore(flags, MPFR_FLAGS_ALL);
  return status;
}

/* Verification-only sine/cosine bounds; no runtime tensor enters compilation. */
int merlin_math_unary(unsigned width, uint64_t bits, unsigned operation, unsigned precision,
                      uint64_t *lower_bits, uint64_t *upper_bits) {
  if ((width != 32 && width != 64) || operation>1 || precision<64 || precision>4096 ||
      !lower_bits || !upper_bits || (width==32 && (bits>>32))) return -1;
  unsigned fraction=width==32 ? 23 : 52;
  uint64_t magnitude=(UINT64_C(1)<<(width-1))-1;
  uint64_t infinity=magnitude-((UINT64_C(1)<<fraction)-1);
  if ((bits&magnitude)>=infinity) return -2;
  if (mpfr_get_emin()>-4096 || mpfr_get_emax()<4096) return -3;
  mpfr_flags_t flags=mpfr_flags_save();
  mpfr_t input,lower,upper,scratch;
  mpfr_inits2(precision,input,lower,upper,scratch,(mpfr_ptr)0);
  set_bits(input,bits,width);
  if (operation==0) { mpfr_sin(lower,input,MPFR_RNDD); mpfr_sin(upper,input,MPFR_RNDU); }
  else { mpfr_cos(lower,input,MPFR_RNDD); mpfr_cos(upper,input,MPFR_RNDU); }
  int status=-4;
  if (!mpfr_nan_p(lower) && !mpfr_nan_p(upper)) {
    uint64_t sign=mpfr_signbit(lower) ? UINT64_C(1)<<(width-1) : 0;
    mpfr_abs(scratch,lower,MPFR_RNDN); *lower_bits=get_bits(scratch,scratch,width)|sign;
    sign=mpfr_signbit(upper) ? UINT64_C(1)<<(width-1) : 0;
    mpfr_abs(scratch,upper,MPFR_RNDN); *upper_bits=get_bits(scratch,scratch,width)|sign;
    status=*lower_bits==*upper_bits ? 0 : 1;
  }
  mpfr_clears(input,lower,upper,scratch,(mpfr_ptr)0);
  mpfr_flags_restore(flags,MPFR_FLAGS_ALL);
  return status;
}
