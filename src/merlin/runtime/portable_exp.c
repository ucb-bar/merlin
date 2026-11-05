/* Portable binary32 exponential for Merlin's explicit host-math policy.
 * Adapted from Arm Optimized Routines commit
 * 97362b2eab91d35d30c4ac2c8c79dcd4dae4331b: expf.c, exp2f_data.c,
 * math_config.h and math_errf.c. Original notices remain below.
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 * See portable_trig.LICENSE and portable_math.provenance.json.
 * Selected configuration: EXP2F_TABLE_BITS=5, TOINT_INTRINSICS=0,
 * WANT_ERRNO=0, WANT_ERRNO_UFLOW=0, USE_GLIBC_ABI=0.
 * Returned values assume round-to-nearest, ties-to-even. errno and floating
 * exception flags are outside this policy. Build without FMA contraction,
 * excess precision or fast-math; same source is used by the host oracle.
 */
#include <float.h>
#include <math.h>
#include <stdint.h>
#if FLT_RADIX != 2 || FLT_MANT_DIG != 24 || DBL_MANT_DIG != 53 || FLT_EVAL_METHOD != 0
#error "portable host math requires binary32/binary64 without excess precision"
#endif
#define MERLIN_EXP2F_TABLE_BITS 5
#define MERLIN_EXP2F_POLY_ORDER 3
#define MERLIN_EXP_N (1 << MERLIN_EXP2F_TABLE_BITS)
static inline uint32_t merlin_exp_asuint(float f) { union { float f; uint32_t i; } u = {f}; return u.i; }
static inline uint64_t merlin_exp_asuint64(double f) { union { double f; uint64_t i; } u = {f}; return u.i; }
static inline double merlin_exp_asdouble(uint64_t i) { union { uint64_t i; double f; } u = {i}; return u.f; }
static inline float merlin_exp_eval_as_float(float x) { return x; }
static inline double merlin_exp_eval_as_double(double x) { return x; }
static inline float merlin_exp_opt_barrier_float(float x) { volatile float y = x; return y; }
/* math_errf.c, Copyright (c) 2017-2020, Arm Limited. errno disabled. */
static __attribute__((noinline)) float merlin_exp_xflowf(uint32_t sign, float y) {
  y = merlin_exp_eval_as_float(merlin_exp_opt_barrier_float(sign ? -y : y) * y);
  return y;
}
static float merlin_exp_uflowf(uint32_t sign) { return merlin_exp_xflowf(sign, 0x1p-95f); }
static float merlin_exp_oflowf(uint32_t sign) { return merlin_exp_xflowf(sign, 0x1p97f); }
struct merlin_exp2f_data {
  uint64_t tab[MERLIN_EXP_N]; double shift_scaled;
  double poly[MERLIN_EXP2F_POLY_ORDER]; double invln2_scaled;
  double poly_scaled[MERLIN_EXP2F_POLY_ORDER]; double shift;
};

/*
 * Shared data between expf, exp2f and powf.
 *
 * Copyright (c) 2017-2018, Arm Limited.
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */





static const struct merlin_exp2f_data merlin_exp2f_data = {
  /* tab[i] = uint(2^(i/MERLIN_EXP_N)) - (i << 52-BITS)
     used for computing 2^(k/MERLIN_EXP_N) for an int |k| < 150 MERLIN_EXP_N as
     double(tab[k%MERLIN_EXP_N] + (k << 52-BITS)) */
  .tab = {
#if MERLIN_EXP_N == 8
0x3ff0000000000000, 0x3fef72b83c7d517b, 0x3fef06fe0a31b715, 0x3feebfdad5362a27,
0x3feea09e667f3bcd, 0x3feeace5422aa0db, 0x3feee89f995ad3ad, 0x3fef5818dcfba487,
#elif MERLIN_EXP_N == 16
0x3ff0000000000000, 0x3fefb5586cf9890f, 0x3fef72b83c7d517b, 0x3fef387a6e756238,
0x3fef06fe0a31b715, 0x3feedea64c123422, 0x3feebfdad5362a27, 0x3feeab07dd485429,
0x3feea09e667f3bcd, 0x3feea11473eb0187, 0x3feeace5422aa0db, 0x3feec49182a3f090,
0x3feee89f995ad3ad, 0x3fef199bdd85529c, 0x3fef5818dcfba487, 0x3fefa4afa2a490da,
#elif MERLIN_EXP_N == 32
0x3ff0000000000000, 0x3fefd9b0d3158574, 0x3fefb5586cf9890f, 0x3fef9301d0125b51,
0x3fef72b83c7d517b, 0x3fef54873168b9aa, 0x3fef387a6e756238, 0x3fef1e9df51fdee1,
0x3fef06fe0a31b715, 0x3feef1a7373aa9cb, 0x3feedea64c123422, 0x3feece086061892d,
0x3feebfdad5362a27, 0x3feeb42b569d4f82, 0x3feeab07dd485429, 0x3feea47eb03a5585,
0x3feea09e667f3bcd, 0x3fee9f75e8ec5f74, 0x3feea11473eb0187, 0x3feea589994cce13,
0x3feeace5422aa0db, 0x3feeb737b0cdc5e5, 0x3feec49182a3f090, 0x3feed503b23e255d,
0x3feee89f995ad3ad, 0x3feeff76f2fb5e47, 0x3fef199bdd85529c, 0x3fef3720dcef9069,
0x3fef5818dcfba487, 0x3fef7c97337b9b5f, 0x3fefa4afa2a490da, 0x3fefd0765b6e4540,
#elif MERLIN_EXP_N == 64
0x3ff0000000000000, 0x3fefec9a3e778061, 0x3fefd9b0d3158574, 0x3fefc74518759bc8,
0x3fefb5586cf9890f, 0x3fefa3ec32d3d1a2, 0x3fef9301d0125b51, 0x3fef829aaea92de0,
0x3fef72b83c7d517b, 0x3fef635beb6fcb75, 0x3fef54873168b9aa, 0x3fef463b88628cd6,
0x3fef387a6e756238, 0x3fef2b4565e27cdd, 0x3fef1e9df51fdee1, 0x3fef1285a6e4030b,
0x3fef06fe0a31b715, 0x3feefc08b26416ff, 0x3feef1a7373aa9cb, 0x3feee7db34e59ff7,
0x3feedea64c123422, 0x3feed60a21f72e2a, 0x3feece086061892d, 0x3feec6a2b5c13cd0,
0x3feebfdad5362a27, 0x3feeb9b2769d2ca7, 0x3feeb42b569d4f82, 0x3feeaf4736b527da,
0x3feeab07dd485429, 0x3feea76f15ad2148, 0x3feea47eb03a5585, 0x3feea23882552225,
0x3feea09e667f3bcd, 0x3fee9fb23c651a2f, 0x3fee9f75e8ec5f74, 0x3fee9feb564267c9,
0x3feea11473eb0187, 0x3feea2f336cf4e62, 0x3feea589994cce13, 0x3feea8d99b4492ed,
0x3feeace5422aa0db, 0x3feeb1ae99157736, 0x3feeb737b0cdc5e5, 0x3feebd829fde4e50,
0x3feec49182a3f090, 0x3feecc667b5de565, 0x3feed503b23e255d, 0x3feede6b5579fdbf,
0x3feee89f995ad3ad, 0x3feef3a2b84f15fb, 0x3feeff76f2fb5e47, 0x3fef0c1e904bc1d2,
0x3fef199bdd85529c, 0x3fef27f12e57d14b, 0x3fef3720dcef9069, 0x3fef472d4a07897c,
0x3fef5818dcfba487, 0x3fef69e603db3285, 0x3fef7c97337b9b5f, 0x3fef902ee78b3ff6,
0x3fefa4afa2a490da, 0x3fefba1bee615a27, 0x3fefd0765b6e4540, 0x3fefe7c1819e90d8,
#endif
  },
  .shift_scaled = 0x1.8p+52 / MERLIN_EXP_N,
  .poly = {
#if MERLIN_EXP_N == 8
  0x1.c6a00335106e2p-5, 0x1.ec0c313449f55p-3, 0x1.62e431111f69fp-1,
#elif MERLIN_EXP_N == 16
  0x1.c6ac6aa313963p-5, 0x1.ebfff4532d9bap-3, 0x1.62e43001bc49fp-1,
#elif MERLIN_EXP_N == 32
  0x1.c6af84b912394p-5, 0x1.ebfce50fac4f3p-3, 0x1.62e42ff0c52d6p-1,
#elif MERLIN_EXP_N == 64
  0x1.c6b04b4221b2ap-5, 0x1.ebfc213e184d7p-3, 0x1.62e42fefb5b7fp-1,
#endif
  },
  .shift = 0x1.8p+52,
  .invln2_scaled = 0x1.71547652b82fep+0 * MERLIN_EXP_N,
  .poly_scaled = {
#if MERLIN_EXP_N == 8
  0x1.c6a00335106e2p-5/MERLIN_EXP_N/MERLIN_EXP_N/MERLIN_EXP_N, 0x1.ec0c313449f55p-3/MERLIN_EXP_N/MERLIN_EXP_N, 0x1.62e431111f69fp-1/MERLIN_EXP_N,
#elif MERLIN_EXP_N == 16
  0x1.c6ac6aa313963p-5/MERLIN_EXP_N/MERLIN_EXP_N/MERLIN_EXP_N, 0x1.ebfff4532d9bap-3/MERLIN_EXP_N/MERLIN_EXP_N, 0x1.62e43001bc49fp-1/MERLIN_EXP_N,
#elif MERLIN_EXP_N == 32
  0x1.c6af84b912394p-5/MERLIN_EXP_N/MERLIN_EXP_N/MERLIN_EXP_N, 0x1.ebfce50fac4f3p-3/MERLIN_EXP_N/MERLIN_EXP_N, 0x1.62e42ff0c52d6p-1/MERLIN_EXP_N,
#elif MERLIN_EXP_N == 64
  0x1.c6b04b4221b2ap-5/MERLIN_EXP_N/MERLIN_EXP_N/MERLIN_EXP_N, 0x1.ebfc213e184d7p-3/MERLIN_EXP_N/MERLIN_EXP_N, 0x1.62e42fefb5b7fp-1/MERLIN_EXP_N,
#endif
  },
};

/*
 * Single-precision e^x function.
 *
 * Copyright (c) 2017-2025, Arm Limited.
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */


/*
MERLIN_EXP2F_TABLE_BITS = 5
EXP2F_POLY_ORDER = 3

ULP error: 0.502 (nearest rounding.)
Relative error: 1.69 * 2^-34 in [-ln2/64, ln2/64] (before rounding.)
Wrong count: 170635 (all nearest rounding wrong results with fma.)
Non-nearest ULP error: 1 (rounded ULP error)
*/


#define MERLIN_EXP_INV_LN2_N merlin_exp2f_data.invln2_scaled
#define MERLIN_EXP_T merlin_exp2f_data.tab
#define MERLIN_EXP_C merlin_exp2f_data.poly_scaled

static inline uint32_t
merlin_exp_top12 (float x)
{
  return merlin_exp_asuint (x) >> 20;
}

float
__wrap_expf (float x)
{
  uint32_t abstop;
  uint64_t ki, t;
  double kd, xd, z, r, r2, y, s;

  xd = x;
  abstop = merlin_exp_top12 (x) & 0x7ff;
  if (__builtin_expect (abstop >= merlin_exp_top12 (88.0f), 0))
    {
      /* |x| >= 88 or x is nan.  */
      if (merlin_exp_asuint (x) == merlin_exp_asuint (-INFINITY))
	return 0.0f;
      if (abstop >= merlin_exp_top12 (INFINITY))
	return x + x;
      if (x > 0x1.62e42ep6f) /* x > log(0x1p128) ~= 88.72 */
	return merlin_exp_oflowf (0);
      if (x < -0x1.9fe368p6f) /* x < log(0x1p-150) ~= -103.97 */
	return merlin_exp_uflowf (0);
#if 0
      if (x < -0x1.9d1d9ep6f) /* x < log(0x1p-149) ~= -103.28 */
	return __math_may_uflowf (0);
#endif
    }

  /* x*MERLIN_EXP_N/Ln2 = k + r with r in [-1/2, 1/2] and int k.  */
  z = MERLIN_EXP_INV_LN2_N * xd;

  /* Round and convert z to int, the result is in [-150*MERLIN_EXP_N, 128*MERLIN_EXP_N] and
     ideally nearest int is used, otherwise the magnitude of r can be
     bigger which gives larger approximation error.  */
#if 0
  kd = roundtoint (z);
  ki = converttoint (z);
#else
# define MERLIN_EXP_SHIFT merlin_exp2f_data.shift
  kd = merlin_exp_eval_as_double (z + MERLIN_EXP_SHIFT);
  ki = merlin_exp_asuint64 (kd);
  kd -= MERLIN_EXP_SHIFT;
#endif
  r = z - kd;

  /* exp(x) = 2^(k/MERLIN_EXP_N) * 2^(r/MERLIN_EXP_N) ~= s * (C0*r^3 + C1*r^2 + C2*r + 1) */
  t = MERLIN_EXP_T[ki % MERLIN_EXP_N];
  t += ki << (52 - MERLIN_EXP2F_TABLE_BITS);
  s = merlin_exp_asdouble (t);
  z = MERLIN_EXP_C[0] * r + MERLIN_EXP_C[1];
  r2 = r * r;
  y = MERLIN_EXP_C[2] * r + 1;
  y = z * r2 + y;
  y = y * s;
  return merlin_exp_eval_as_float (y);
}
