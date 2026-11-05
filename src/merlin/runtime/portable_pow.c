/* Shared binary32 pow implementation for portable_neural_f32.
 * Arm Optimized Routines commit97362b2eab91d35d30c4ac2c8c79dcd4dae4331b.
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 * See portable_trig.LICENSE and portable_math.provenance.json.
 * Included after portable_exp.c: it reuses the SAME pinned exp2 table/config.
 * Fixed TOINT_INTRINSICS=0, WANT_ERRNO=0, WANT_ERRNO_UFLOW=0, WANT_ROUNDING=1.
 * No contraction/fast-math/excess precision; round-to-nearest ties-even.
 */
#define MERLIN_POW_LOG2_TABLE_BITS 4
#define MERLIN_POW_LOG2_POLY_ORDER 5
#define MERLIN_POW_SCALE_BITS 0
#define MERLIN_POW_SCALE 1.0
#define MERLIN_POW_WANT_ROUNDING 1
static inline float merlin_pow_asfloat(uint32_t i) { union { uint32_t i; float f; } u = {i}; return u.f; }
static inline int merlin_pow_issignaling(float x) {
  uint32_t ix = merlin_exp_asuint(x);
  return 2 * (ix ^ 0x00400000) > 2u * 0x7fc00000;
}
static float merlin_pow_invalid(float x) { return (x-x)/(x-x); }
struct merlin_pow_log2_data { struct { double invc, logc; } tab[1 << MERLIN_POW_LOG2_TABLE_BITS]; double poly[MERLIN_POW_LOG2_POLY_ORDER]; };
#define MERLIN_POW_UNLIKELY(x) __builtin_expect(!!(x),0)

/*
 * Data definition for powf.
 *
 * Copyright (c) 2017-2019, Arm Limited.
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */



static const struct merlin_pow_log2_data merlin_pow_log2_data = {
  .tab = {
  { 0x1.661ec79f8f3bep+0, -0x1.efec65b963019p-2 * MERLIN_POW_SCALE },
  { 0x1.571ed4aaf883dp+0, -0x1.b0b6832d4fca4p-2 * MERLIN_POW_SCALE },
  { 0x1.49539f0f010bp+0, -0x1.7418b0a1fb77bp-2 * MERLIN_POW_SCALE },
  { 0x1.3c995b0b80385p+0, -0x1.39de91a6dcf7bp-2 * MERLIN_POW_SCALE },
  { 0x1.30d190c8864a5p+0, -0x1.01d9bf3f2b631p-2 * MERLIN_POW_SCALE },
  { 0x1.25e227b0b8eap+0, -0x1.97c1d1b3b7afp-3 * MERLIN_POW_SCALE },
  { 0x1.1bb4a4a1a343fp+0, -0x1.2f9e393af3c9fp-3 * MERLIN_POW_SCALE },
  { 0x1.12358f08ae5bap+0, -0x1.960cbbf788d5cp-4 * MERLIN_POW_SCALE },
  { 0x1.0953f419900a7p+0, -0x1.a6f9db6475fcep-5 * MERLIN_POW_SCALE },
  { 0x1p+0, 0x0p+0 * MERLIN_POW_SCALE },
  { 0x1.e608cfd9a47acp-1, 0x1.338ca9f24f53dp-4 * MERLIN_POW_SCALE },
  { 0x1.ca4b31f026aap-1, 0x1.476a9543891bap-3 * MERLIN_POW_SCALE },
  { 0x1.b2036576afce6p-1, 0x1.e840b4ac4e4d2p-3 * MERLIN_POW_SCALE },
  { 0x1.9c2d163a1aa2dp-1, 0x1.40645f0c6651cp-2 * MERLIN_POW_SCALE },
  { 0x1.886e6037841edp-1, 0x1.88e9c2c1b9ff8p-2 * MERLIN_POW_SCALE },
  { 0x1.767dcf5534862p-1, 0x1.ce0a44eb17bccp-2 * MERLIN_POW_SCALE },
  },
  .poly = {
  0x1.27616c9496e0bp-2 * MERLIN_POW_SCALE, -0x1.71969a075c67ap-2 * MERLIN_POW_SCALE,
  0x1.ec70a6ca7baddp-2 * MERLIN_POW_SCALE, -0x1.7154748bef6c8p-1 * MERLIN_POW_SCALE,
  0x1.71547652ab82bp0 * MERLIN_POW_SCALE,
  }
};

/* Returns 0 if not int, 1 if odd int, 2 if even int.  The argument is
   the bit representation of a non-zero finite floating-point value.  */
static inline int
merlin_pow_checkint (uint32_t iy)
{
  int e = iy >> 23 & 0xff;
  if (e < 0x7f)
    return 0;
  if (e > 0x7f + 23)
    return 2;
  if (iy & ((1 << (0x7f + 23 - e)) - 1))
    return 0;
  if (iy & (1 << (0x7f + 23 - e)))
    return 1;
  return 2;
}

/* Returns 1 if input is the bit representation of 0, infinity or nan.  */
static inline int
merlin_pow_zeroinfnan (uint32_t ix)
{
  return 2 * ix - 1 >= 2u * 0x7f800000 - 1;
}


/*
 * Single-precision pow function.
 *
 * Copyright (c) 2017-2026, Arm Limited.
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */


/*
MERLIN_POW_LOG2_POLY_ORDER = 5
MERLIN_EXP2F_TABLE_BITS = 5

ULP error: 0.82 (~ 0.5 + relerr*2^24)
relerr: 1.27 * 2^-26 (Relative error ~= 128*Ln2*relerr_log2 + relerr_exp2)
relerr_log2: 1.83 * 2^-33 (Relative error of logx.)
relerr_exp2: 1.69 * 2^-34 (Relative error of exp2(ylogx).).  */

#define MERLIN_POW_N (1 << MERLIN_POW_LOG2_TABLE_BITS)
#define MERLIN_POW_T merlin_pow_log2_data.tab
#define MERLIN_POW_A merlin_pow_log2_data.poly
#define MERLIN_POW_OFF 0x3f330000

/* Subnormal input is normalized so ix has negative biased exponent.
   Output is multiplied by MERLIN_POW_N (MERLIN_POW_SCALE) if TOINT_INTRINICS is set.  */
static inline double
merlin_pow_log2_inline (uint32_t ix)
{
  double z, r, r2, r4, p, q, y, y0, invc, logc;
  uint32_t iz, top, tmp;
  int k, i;

  /* x = 2^k z; where z is in range [MERLIN_POW_OFF,2*MERLIN_POW_OFF] and exact.
     The range is split into MERLIN_POW_N subintervals.
     The ith subinterval contains z and c is near its center.  */
  tmp = ix - MERLIN_POW_OFF;
  i = (tmp >> (23 - MERLIN_POW_LOG2_TABLE_BITS)) % MERLIN_POW_N;
  top = tmp & 0xff800000;
  iz = ix - top;
  k = (int32_t) top >> (23 - MERLIN_POW_SCALE_BITS); /* arithmetic shift.  */
  invc = MERLIN_POW_T[i].invc;
  logc = MERLIN_POW_T[i].logc;
  z = merlin_pow_asfloat (iz);

  /* log2(x) = log1p(z/c-1)/ln2 + log2(c) + k.  */
  r = z * invc - 1;
  y0 = logc + (double) k;

  /* Pipelined polynomial evaluation to approximate log1p(r)/ln2.  */
  r2 = r * r;
  y = MERLIN_POW_A[0] * r + MERLIN_POW_A[1];
  p = MERLIN_POW_A[2] * r + MERLIN_POW_A[3];
  r4 = r2 * r2;
  q = MERLIN_POW_A[4] * r + y0;
  q = p * r2 + q;
  y = y * r4 + q;
  return y;
}

#undef MERLIN_POW_N
#undef MERLIN_POW_T
#define MERLIN_POW_N (1 << MERLIN_EXP2F_TABLE_BITS)
#define MERLIN_POW_T merlin_exp2f_data.tab
#define MERLIN_POW_SIGN_BIAS (1 << (MERLIN_EXP2F_TABLE_BITS + 11))

/* The output of log2 and thus the input of exp2 is either scaled by MERLIN_POW_N
   (in case of fast toint intrinsics) or not.  The unscaled xd must be
   in [-1021,1023], sign_bias sets the sign of the result.  */
static inline float
merlin_pow_exp2_inline (double xd, uint32_t sign_bias)
{
  uint64_t ki, ski, t;
  double kd, z, r, r2, y, s;

#if 0
# define MERLIN_POW_C merlin_exp2f_data.poly_scaled
  /* MERLIN_POW_N*x = k + r with r in [-1/2, 1/2].  */
  kd = roundtoint (xd); /* k.  */
  ki = converttoint (xd);
#else
# define MERLIN_POW_C merlin_exp2f_data.poly
# define MERLIN_POW_SHIFT merlin_exp2f_data.shift_scaled
  /* x = k/MERLIN_POW_N + r with r in [-1/(2N), 1/(2N)].  */
  kd = merlin_exp_eval_as_double (xd + MERLIN_POW_SHIFT);
  ki = merlin_exp_asuint64 (kd);
  kd -= MERLIN_POW_SHIFT; /* k/MERLIN_POW_N.  */
#endif
  r = xd - kd;

  /* exp2(x) = 2^(k/MERLIN_POW_N) * 2^r ~= s * (C0*r^3 + C1*r^2 + C2*r + 1).  */
  t = MERLIN_POW_T[ki % MERLIN_POW_N];
  ski = ki + sign_bias;
  t += ski << (52 - MERLIN_EXP2F_TABLE_BITS);
  s = merlin_exp_asdouble (t);
  z = MERLIN_POW_C[0] * r + MERLIN_POW_C[1];
  r2 = r * r;
  y = MERLIN_POW_C[2] * r + 1;
  y = z * r2 + y;
  y = y * s;
  return merlin_exp_eval_as_float (y);
}

float
__wrap_powf (float x, float y)
{
  uint32_t sign_bias = 0;
  uint32_t ix, iy;

  ix = merlin_exp_asuint (x);
  iy = merlin_exp_asuint (y);
  if (MERLIN_POW_UNLIKELY (ix - 0x00800000 >= 0x7f800000 - 0x00800000 || merlin_pow_zeroinfnan (iy)))
    {
      /* Either (x < 0x1p-126 or inf or nan) or (y is 0 or inf or nan).  */
      if (MERLIN_POW_UNLIKELY (merlin_pow_zeroinfnan (iy)))
	{
	  if (2 * iy == 0)
	    return merlin_pow_issignaling (x) ? x + y : 1.0f;
	  if (ix == 0x3f800000)
	    return merlin_pow_issignaling (y) ? x + y : 1.0f;
	  if (2 * ix > 2u * 0x7f800000 || 2 * iy > 2u * 0x7f800000)
	    return x + y;
	  if (2 * ix == 2 * 0x3f800000)
	    return 1.0f;
	  if ((2 * ix < 2 * 0x3f800000) == !(iy & 0x80000000))
	    return 0.0f; /* |x|<1 && y==inf or |x|>1 && y==-inf.  */
	  return y * y;
	}
      if (MERLIN_POW_UNLIKELY (merlin_pow_zeroinfnan (ix)))
	{
	  float x2 = x * x;
	  if (ix & 0x80000000 && merlin_pow_checkint (iy) == 1)
	    {
	      x2 = -x2;
	      sign_bias = 1;
	    }
#if 0
	  if (2 * ix == 0 && iy & 0x80000000)
	    return __math_divzerof (sign_bias);
#endif
	  /* Without the barrier some versions of clang hoist the 1/x2 and
	     thus division by zero exception can be signaled spuriously.  */
	  return iy & 0x80000000 ? merlin_exp_opt_barrier_float (1 / x2) : x2;
	}
      /* x and y are non-zero finite.  */
      if (ix & 0x80000000)
	{
	  /* Finite x < 0.  */
	  int yint = merlin_pow_checkint (iy);
	  if (yint == 0)
	    return merlin_pow_invalid (x);
	  if (yint == 1)
	    sign_bias = MERLIN_POW_SIGN_BIAS;
	  ix &= 0x7fffffff;
	}
      if (ix < 0x00800000)
	{
	  /* Normalize subnormal x so exponent becomes negative.  */
	  ix = merlin_exp_asuint (x * 0x1p23f);
	  ix &= 0x7fffffff;
	  ix -= 23 << 23;
	}
    }
  /* y * log2(x) cannot overflow since y is single precision.  */
  double ylogx = (double) y * merlin_pow_log2_inline (ix);

  /* Check whether |y*log(x)| >= 126.  */
  if (MERLIN_POW_UNLIKELY ((merlin_exp_asuint64 (ylogx) >> 47 & 0xffff)
		 >= merlin_exp_asuint64 (126.0 * MERLIN_POW_SCALE) >> 47))
    {
      if (ylogx <= -150.0 * MERLIN_POW_SCALE)
	return merlin_exp_uflowf (sign_bias);


      /* |x^y| > 0x1.ffffffp127.  */
      if (!MERLIN_POW_WANT_ROUNDING && ylogx > 0x1.fffffffd1d571p+6 * MERLIN_POW_SCALE)
	return merlin_exp_oflowf (sign_bias);

      if (MERLIN_POW_WANT_ROUNDING && ylogx > 0x1.fffffffa3aae2p+6 * MERLIN_POW_SCALE)
	{
	  if (ylogx > 0x1.fffffffd1d571p+6 * MERLIN_POW_SCALE)
	    return merlin_exp_oflowf (sign_bias);

	  /* |x^y| > 0x1.fffffep127, check if we round away from 0.  */
	  if (ylogx != 0x1.fffffffa3aae3p+6 * MERLIN_POW_SCALE)
	    {
	      float x = merlin_exp_opt_barrier_float (0x1p-25f);
	      if ((!sign_bias && merlin_exp_eval_as_float (1.0f + x) != 1.0f)
		  || (sign_bias && merlin_exp_eval_as_float (-1.0f - x) != -1.0f))
		return merlin_exp_oflowf (sign_bias);
	    }

	  return sign_bias ? -0x1.fffffep127 : 0x1.fffffep127;
	}
    }
  return merlin_pow_exp2_inline (ylogx, sign_bias);
}
