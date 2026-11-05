/* Merlin's portable transcendentals: one float erf/exp/tanh/log for every target.
 *
 * WHY. `convert-math-to-libm` lowers math.erf/exp/tanh/log to the C library's erff/expf/tanhf/logf,
 * so the same model computed a different function on each target: glibc on the host that writes the
 * W8A8 golden, picolibc/newlib in a RISC-V image. Measured: erff differs by 1 ulp between the two on
 * 76 of 8001 inputs in GELU's range, and requantization turns that ulp into whole int8 steps, which
 * tripped the W8A8 gate's per-element bound on a model that was otherwise bit-exact. Under the
 * `portable-transcendentals` numeric policy (merlin.llvmlower.portable_math) the lowered model calls
 * merlin_erff/merlin_expf/merlin_tanhf/merlin_logf instead, defined here and compiled into the same
 * runtime object (mlir_runtime.c) on the host and on the device, so the golden and the image
 * evaluate the same function bit for bit.
 *
 * HOW THE BITS ARE PINNED. Each function converts its float argument to double (exact), evaluates
 * the function in double with a fixed sequence of IEEE-754 operations, and rounds once to float.
 * IEEE-754 defines every one of those operations (+, -, *, /, and fused multiply-add) exactly, so
 * the result is a function of this source alone. The one freedom a conforming C compiler still has
 * is CONTRACTION: fusing a multiply whose result feeds an add into one fma, which rounds once instead
 * of twice. GCC does it across statements by default for GNU C (-ffp-contract=fast; on RISC-V that
 * is fmadd.d), clang within an expression, and neither pragma nor attribute switches it off in every
 * compiler mode. So this file leaves the compiler nothing to fuse: every multiply-add is written as
 * an explicit __builtin_fma, and no other product feeds an addition or subtraction. (The
 * multiplications that remain feed a division, a conversion to float, an fma operand, or a further
 * multiplication.) __builtin_fma is a single correctly rounded operation everywhere: fmadd.d on
 * RISC-V with D, vfmadd on x86 with FMA, and otherwise the C library's fma, which IEEE-754 and C99
 * require to be correctly rounded.
 *
 * The build refuses -ffast-math and excess-precision evaluation outright, since both change what an
 * operation means. Default rounding (round to nearest, ties to even) is assumed. No libm
 * transcendental is called.
 *
 * ACCURACY. The double evaluation is accurate to a few units in the last place of a double (about
 * 1e-15 relative), so the float result is the correctly rounded value except where the true value
 * lies within that distance of a float rounding boundary. The coefficients are generated, not
 * tuned: build_tools/scripts/gen_portable_math_tables.py. The measured accuracy (exhaustive over
 * the float inputs) is in docs/design/portable_transcendentals.md.
 *
 * Included exactly once, by mlir_runtime.c, with MERLIN_PORTABLE_MATH_IMPLEMENTATION defined, so
 * every link path that already carries the MLIR runtime carries these definitions too. */
#ifndef MERLIN_PORTABLE_MATH_H
#define MERLIN_PORTABLE_MATH_H

float merlin_erff(float x);
float merlin_expf(float x);
float merlin_tanhf(float x);
float merlin_logf(float x);

#ifdef MERLIN_PORTABLE_MATH_IMPLEMENTATION

#include <stdint.h>

#include "portable_math_tables.h"

#if defined(__FAST_MATH__)
#error "portable_math.h must not be compiled with -ffast-math: its results are defined by strict IEEE-754 evaluation"
#endif
#if defined(__FLT_EVAL_METHOD__) && __FLT_EVAL_METHOD__ != 0
#error "portable_math.h needs FLT_EVAL_METHOD == 0 (no excess precision, e.g. SSE2 rather than x87)"
#endif

/* clang turns __builtin_fma into a call to the C library's fma whenever math-errno is on (the
 * default for a freestanding RISC-V target); its elementwise builtin is always the instruction.
 * Either is the same correctly rounded operation; the instruction is just cheaper. */
#if defined(__clang__) && defined(__has_builtin)
#if __has_builtin(__builtin_elementwise_fma)
#define MPM_FMA(a, b, c) __builtin_elementwise_fma((a), (b), (c))
#endif
#endif
#ifndef MPM_FMA
#define MPM_FMA(a, b, c) __builtin_fma((a), (b), (c))
#endif

/* The quiet NaN every function returns for a NaN argument. Propagating the argument's payload is
 * not portable (RISC-V returns the canonical NaN from arithmetic, x86 keeps the payload), so the
 * payload is not propagated. */
#define MPM_NAN_BITS 0x7fc00000u

/* 1.5 * 2^52: adding it and subtracting it again rounds a double of magnitude < 2^51 to an integer. */
#define MPM_ROUND_SHIFT 0x1.8p52
/* ln(2) / 2, the bound of the reduced argument of exp and of expm1's direct series. */
#define MPM_HALF_LN2 0x1.62e42fefa39efp-2

static uint32_t mpm_f32_bits(float x) {
  uint32_t b;
  __builtin_memcpy(&b, &x, 4);
  return b;
}

static float mpm_f32_from_bits(uint32_t b) {
  float x;
  __builtin_memcpy(&x, &b, 4);
  return x;
}

static uint64_t mpm_f64_bits(double x) {
  uint64_t b;
  __builtin_memcpy(&b, &x, 8);
  return b;
}

static double mpm_f64_from_bits(uint64_t b) {
  double x;
  __builtin_memcpy(&x, &b, 8);
  return x;
}

/* sum_{n<=degree} c[n] t^n, Horner from the highest degree, one fma per step. */
static double mpm_poly(const double *c, int degree, double t) {
  double p = c[degree];
  for (int n = degree - 1; n >= 0; --n)
    p = MPM_FMA(p, t, c[n]);
  return p;
}

/* exp(x) = mantissa * *scale for |x| <= 700, with *scale = 2^k exact: x = k ln2 + r, |r| <= ln2/2
 * (Cody-Waite, k * LN2_HI exact), exp(r) by its Taylor series. Split so a caller that adds to
 * exp(x) can do so in one fma rather than leave the compiler a product feeding an add. */
static double mpm_exp_split(double x, double *scale) {
  double kd = MPM_FMA(x, MPM_LOG2E, MPM_ROUND_SHIFT) - MPM_ROUND_SHIFT;
  double r = MPM_FMA(-kd, MPM_LN2_HI, x);
  r = MPM_FMA(-kd, MPM_LN2_LO, r);
  *scale = mpm_f64_from_bits((uint64_t)(int64_t)((int)kd + 1023) << 52);
  return mpm_poly(MPM_EXP_TAYLOR, MPM_EXP_DEGREE, r);
}

float merlin_expf(float xf) {
  double x = (double)xf;
  double scale;
  if (x != x)
    return mpm_f32_from_bits(MPM_NAN_BITS);
  /* Outside [-104, 89] the float result is 0 or +inf; clamping keeps 2^k a normal double and the
   * final rounding produces exactly that 0 or inf (and every subnormal result in between). */
  if (x < -104.0)
    x = -104.0;
  if (x > 89.0)
    x = 89.0;
  double m = mpm_exp_split(x, &scale);
  return (float)(m * scale);
}

float merlin_erff(float xf) {
  uint32_t bits = mpm_f32_bits(xf);
  double a = (double)mpm_f32_from_bits(bits & 0x7fffffffu);
  double r;
  if (a != a)
    return mpm_f32_from_bits(MPM_NAN_BITS);
  if (a < 0.5) {
    /* erf(a) = a * sum c_n a^(2n); a*a is exact in double for a float a. */
    r = a * mpm_poly(MPM_ERF_SMALL, MPM_ERF_SMALL_TERMS - 1, a * a);
  } else if (a < 4.0) {
    /* erf(a) = 1 - exp(-a^2) g(a), with g(a) = erfc(a) exp(a^2) fitted on half-unit intervals.
     * i and the interval midpoint are exact, and so is t (Sterbenz). */
    int i = (int)((a - 0.5) * 2.0);
    double t = a - (0.75 + 0.5 * (double)i);
    double g = mpm_poly(MPM_ERFC_SCALED[i], MPM_ERFC_DEGREE, t);
    double scale;
    double m = mpm_exp_split(-(a * a), &scale);
    r = MPM_FMA(-(m * scale), g, 1.0); /* m * scale is exact: scale is a power of two */
  } else {
    r = 1.0; /* erfc(4) < 2^-25: every float a >= 4 rounds to 1 */
  }
  return (float)((bits >> 31) ? -r : r);
}

float merlin_tanhf(float xf) {
  uint32_t bits = mpm_f32_bits(xf);
  double a = (double)mpm_f32_from_bits(bits & 0x7fffffffu);
  double r;
  if (a != a)
    return mpm_f32_from_bits(MPM_NAN_BITS);
  if (a < 10.0) {
    /* tanh(a) = expm1(2a) / (expm1(2a) + 2), expm1 without cancellation near 0. */
    double y = a + a;
    double num, den;
    if (y < MPM_HALF_LN2) {
      double p = mpm_poly(MPM_EXPM1_TAYLOR, MPM_EXPM1_DEGREE, y);
      num = y * p;
      den = MPM_FMA(y, p, 2.0);
    } else {
      double scale;
      double m = mpm_exp_split(y, &scale);
      num = MPM_FMA(m, scale, -1.0);
      den = MPM_FMA(m, scale, 1.0);
    }
    r = num / den;
  } else {
    r = 1.0; /* 1 - tanh(10) < 2^-25 */
  }
  return (float)((bits >> 31) ? -r : r);
}

float merlin_logf(float xf) {
  uint32_t bits = mpm_f32_bits(xf);
  if ((bits & 0x7fffffffu) > 0x7f800000u || ((bits >> 31) && (bits & 0x7fffffffu) != 0u))
    return mpm_f32_from_bits(MPM_NAN_BITS); /* NaN, and log(negative) */
  if ((bits & 0x7fffffffu) == 0u)
    return mpm_f32_from_bits(0xff800000u); /* log(+-0) = -inf */
  if (bits == 0x7f800000u)
    return xf; /* log(+inf) = +inf */
  /* A float (subnormals included) is a normal double: split it as m * 2^e, m in [sqrt(1/2), sqrt(2)). */
  uint64_t db = mpm_f64_bits((double)xf);
  int e = (int)((db >> 52) & 0x7ffu) - 1023;
  double m = mpm_f64_from_bits((db & 0x000fffffffffffffull) | 0x3ff0000000000000ull);
  if (m > 0x1.6a09e667f3bcdp+0) { /* sqrt(2) */
    m = m * 0.5;
    e = e + 1;
  }
  /* log(m) = 2 atanh(s), s = (m - 1)/(m + 1). f is exact (Sterbenz), and so is 2 + f, since f has
   * at most 25 significant bits. */
  double f = m - 1.0;
  double s = f / (2.0 + f);
  double z = s * s;
  double hs = s + s;
  double logm = MPM_FMA(hs * z, mpm_poly(MPM_LOG_ATANH, MPM_LOG_TERMS - 1, z), hs);
  double ed = (double)e;
  return (float)MPM_FMA(ed, MPM_LN2_HI, MPM_FMA(ed, MPM_LN2_LO, logm));
}

#endif /* MERLIN_PORTABLE_MATH_IMPLEMENTATION */
#endif /* MERLIN_PORTABLE_MATH_H */
