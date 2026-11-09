#ifndef MERLIN_ORDERED_FMA_BOUNDS_H
#define MERLIN_ORDERED_FMA_BOUNDS_H

/* A certificate for zero-seeded, increasing-K binary32 fused multiply-adds.
 * The caller supplies outward binary64 summaries of consecutive reconstructed
 * product chunks.  No target, tensor shape, or approximate-answer selector is
 * part of this contract.  A summary must enclose the EXACT reconstructed sum,
 * its absolute product sum, and sum(abs(original_product-reconstructed_product)).
 * This header checks arithmetic eligibility, not the origin of those summaries.
 * Source and certificate execution require IEEE RNE, gradual underflow, and no
 * fast-math/reassociation.  Failure must select exact source replay.
 */
#include "source_f32_math.h"
#include <float.h>
#include <fenv.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

typedef struct {
  double sum_lower, sum_upper;
  double absolute_upper, representation_error_upper;
  size_t length;
} merlin_fma_chunk;

typedef struct {
  double prefix_lower, prefix_upper;
  double representation_error, rounding_error;
  int valid;
} merlin_fma_bound;

/* IEEE binary64 adjacency. Unlike a math-library nextafter call, these helpers
 * preserve values without setting floating exception/errno flags. Certificates
 * observe values and eligibility, not those diagnostic side effects. */
static inline double merlin_fma_next_up(double value) {
  uint64_t raw; MERLIN_SOURCE_BITCAST_COPY(&raw, &value, sizeof(raw));
  const uint64_t magnitude = raw & UINT64_C(0x7fffffffffffffff);
  if (magnitude > UINT64_C(0x7ff0000000000000) || raw == UINT64_C(0x7ff0000000000000)) return value;
  if (!magnitude) raw = 1;
  else if (raw >> 63) --raw;
  else ++raw;
  MERLIN_SOURCE_BITCAST_COPY(&value, &raw, sizeof(raw)); return value;
}
static inline double merlin_fma_next_down(double value) {
  uint64_t raw; MERLIN_SOURCE_BITCAST_COPY(&raw, &value, sizeof(raw));
  const uint64_t magnitude = raw & UINT64_C(0x7fffffffffffffff);
  if (magnitude > UINT64_C(0x7ff0000000000000) || raw == UINT64_C(0xfff0000000000000)) return value;
  if (!magnitude) raw = UINT64_C(0x8000000000000001);
  else if (raw >> 63) ++raw;
  else --raw;
  MERLIN_SOURCE_BITCAST_COPY(&value, &raw, sizeof(raw)); return value;
}
static inline float merlin_fma_next_up_f32(float value) {
  uint32_t raw; MERLIN_SOURCE_BITCAST_COPY(&raw, &value, sizeof(raw));
  const uint32_t magnitude = raw & UINT32_C(0x7fffffff);
  if (magnitude > UINT32_C(0x7f800000) || raw == UINT32_C(0x7f800000)) return value;
  if (!magnitude) raw = 1;
  else if (raw >> 31) --raw;
  else ++raw;
  MERLIN_SOURCE_BITCAST_COPY(&value, &raw, sizeof(raw)); return value;
}
static inline float merlin_fma_next_down_f32(float value) {
  uint32_t raw; MERLIN_SOURCE_BITCAST_COPY(&raw, &value, sizeof(raw));
  const uint32_t magnitude = raw & UINT32_C(0x7fffffff);
  if (magnitude > UINT32_C(0x7f800000) || raw == UINT32_C(0xff800000)) return value;
  if (!magnitude) raw = UINT32_C(0x80000001);
  else if (raw >> 31) ++raw;
  else --raw;
  MERLIN_SOURCE_BITCAST_COPY(&value, &raw, sizeof(raw)); return value;
}
static inline double merlin_fma_half_ulp(double magnitude) {
  if (magnitude < 0x1p-126) return 0x1p-150;
  uint64_t raw; MERLIN_SOURCE_BITCAST_COPY(&raw, &magnitude, sizeof(raw));
  const uint64_t exponent = (raw >> 52) & UINT64_C(0x7ff);
  raw = (exponent - 24) << 52;
  double result; MERLIN_SOURCE_BITCAST_COPY(&result, &raw, sizeof(raw)); return result;
}

/* Optional explicitly qualified scalar capabilities. An upward result must
 * enclose the exact real operation from above; a downward result from below.
 * Source RNE and gradual underflow remain eligible at every source operation.
 * Implementations may use fixed directed rounding or a restored local scope;
 * nontrapping arithmetic and unobserved exception flags are caller obligations.
 * Defining a capability does not establish its theorem, ISA legality or cost.
 * With no capability, the original adjacency implementation is unchanged. */
static inline double merlin_fma_up_add(double a, double b) {
#ifdef MERLIN_F64_OUTWARD_ADD_UP
  return MERLIN_F64_OUTWARD_ADD_UP(a, b);
#else
  return merlin_fma_next_up(a + b);
#endif
}
static inline double merlin_fma_down_add(double a, double b) {
#ifdef MERLIN_F64_OUTWARD_ADD_DOWN
  return MERLIN_F64_OUTWARD_ADD_DOWN(a, b);
#else
  return merlin_fma_next_down(a + b);
#endif
}
static inline double merlin_fma_up_mul(double a, double b) {
#ifdef MERLIN_F64_OUTWARD_MUL_UP
  return MERLIN_F64_OUTWARD_MUL_UP(a, b);
#else
  return merlin_fma_next_up(a * b);
#endif
}

/* Optional directed narrowing of exact binary64 inputs. A lower result must
 * be <= the exact input and an upper result >= it. The default RNE cast plus
 * adjacency remains unchanged. Source arithmetic never uses these bounds as
 * an implicit rounding policy; platform and exception proofs remain external. */
static inline float merlin_fma_down_cast_f32(double value) {
#ifdef MERLIN_F32_OUTWARD_FROM_F64_DOWN
  return MERLIN_F32_OUTWARD_FROM_F64_DOWN(value);
#else
  return merlin_fma_next_down_f32((float)value);
#endif
}
static inline float merlin_fma_up_cast_f32(double value) {
#ifdef MERLIN_F32_OUTWARD_FROM_F64_UP
  return MERLIN_F32_OUTWARD_FROM_F64_UP(value);
#else
  return merlin_fma_next_up_f32((float)value);
#endif
}

static inline merlin_fma_bound merlin_fma_bound_begin(void) {
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__)
  return (merlin_fma_bound){0.0, 0.0, 0.0, 0.0, 0};
#else
  if (sizeof(double) != sizeof(uint64_t) || sizeof(float) != sizeof(uint32_t))
    return (merlin_fma_bound){0.0, 0.0, 0.0, 0.0, 0};
  const double one = 1.0;
  uint64_t one_bits; MERLIN_SOURCE_BITCAST_COPY(&one_bits, &one, sizeof(one_bits));
  const float fone = 1.0f;
  uint32_t fone_bits; MERLIN_SOURCE_BITCAST_COPY(&fone_bits, &fone, sizeof(fone_bits));
  volatile float tiny = 0x1p-149f;
  volatile double dtiny = 0x1p-1074;
  volatile float twice_tiny = tiny + tiny;
  volatile double twice_dtiny = dtiny + dtiny;
  const float observed_float = twice_tiny;
  const double observed_double = twice_dtiny;
  uint32_t observed_float_bits; uint64_t observed_double_bits;
  MERLIN_SOURCE_BITCAST_COPY(&observed_float_bits,&observed_float,sizeof(observed_float_bits));
  MERLIN_SOURCE_BITCAST_COPY(&observed_double_bits,&observed_double,sizeof(observed_double_bits));
  const int eligible = FLT_RADIX == 2 && FLT_MANT_DIG == 24 &&
      FLT_MAX_EXP == 128 && FLT_MIN_EXP == -125 && DBL_MANT_DIG == 53 &&
      DBL_MAX_EXP == 1024 && DBL_MIN_EXP == -1021 &&
      one_bits == UINT64_C(0x3ff0000000000000) &&
      fone_bits == UINT32_C(0x3f800000) &&
      fegetround() == FE_TONEAREST && observed_float_bits == UINT32_C(2) &&
      observed_double_bits == UINT64_C(2);
  return (merlin_fma_bound){0.0, 0.0, 0.0, 0.0, eligible};

#endif
}

static inline int merlin_fma_bound_push(merlin_fma_bound *state,
                                      merlin_fma_chunk chunk) {
  if (!state->valid || chunk.length == 0 || chunk.length >= ((size_t)1 << 24) ||
      !MERLIN_SOURCE_ISFINITE(chunk.sum_lower) || !MERLIN_SOURCE_ISFINITE(chunk.sum_upper) ||
      !MERLIN_SOURCE_ISFINITE(chunk.absolute_upper) || !MERLIN_SOURCE_ISFINITE(chunk.representation_error_upper) ||
      chunk.sum_lower > chunk.sum_upper || chunk.absolute_upper < 0.0 ||
      chunk.representation_error_upper < 0.0 ||
      chunk.sum_lower > chunk.absolute_upper ||
      chunk.sum_upper < -chunk.absolute_upper) {
    state->valid = 0;
    return 0;
  }

  /* A partial sum of this chunk lies between its total negative and positive
   * variations: (T-A)/2 <= partial <= (T+A)/2.  Include the uncertain incoming
   * reconstructed prefix and all representation error accumulated so far. */
  const int first = state->prefix_lower == 0.0 && state->prefix_upper == 0.0 &&
      state->representation_error == 0.0 && state->rounding_error == 0.0;
  double repr, magnitude;
  if (first) {
    repr = chunk.representation_error_upper;
    const double signed_magnitude = MERLIN_SOURCE_F64_MAX(MERLIN_SOURCE_F64_ABS(chunk.sum_lower),MERLIN_SOURCE_F64_ABS(chunk.sum_upper));
    magnitude = merlin_fma_up_add(merlin_fma_up_mul(0.5,
        merlin_fma_up_add(signed_magnitude,chunk.absolute_upper)),repr);
  } else {
    const double middle_lower = merlin_fma_down_add(state->prefix_lower,
        merlin_fma_next_down(chunk.sum_lower * 0.5));
    const double middle_upper = merlin_fma_up_add(state->prefix_upper,
        merlin_fma_next_up(chunk.sum_upper * 0.5));
    repr = merlin_fma_up_add(state->representation_error,
                           chunk.representation_error_upper);
    magnitude = merlin_fma_up_add(
        merlin_fma_up_add(MERLIN_SOURCE_F64_MAX(MERLIN_SOURCE_F64_ABS(middle_lower), MERLIN_SOURCE_F64_ABS(middle_upper)),
                          merlin_fma_up_mul(0.5, chunk.absolute_upper)), repr);
  }

  /* e[j] <= (1+u)e[j-1] + u*|exact_prefix[j]| + half_min_subnormal.
   * gamma_l >= (1+u)^l-1 gives an initial safe radius for every step. */
  const double lu = (double)chunk.length * 0x1p-24;
  const double gamma = merlin_fma_next_up(lu / (1.0 - lu));
  const double eta = merlin_fma_next_up((double)chunk.length * 0x1p-150 / (1.0 - lu));
  const double coarse = merlin_fma_up_add(
      merlin_fma_up_mul(merlin_fma_up_add(1.0, gamma), state->rounding_error),
      merlin_fma_up_add(merlin_fma_up_mul(gamma, magnitude), eta));
  const double rounded_input_max = merlin_fma_up_add(magnitude, coarse);
  if (!MERLIN_SOURCE_ISFINITE(rounded_input_max) || rounded_input_max >= (double)FLT_MAX) {
    state->valid = 0;
    return 0;
  }

  /* The coarse radius encloses every exact FMA input.  Its binade gives an
   * absolute half-ulp limit for every rounding, often tighter than u*|x|. */
  const double half_ulp = merlin_fma_half_ulp(rounded_input_max);
  const double sharp = merlin_fma_up_add(state->rounding_error,
      merlin_fma_up_mul((double)chunk.length, half_ulp));
  state->rounding_error = MERLIN_SOURCE_F64_MIN(coarse, sharp);
  state->representation_error = repr;
  state->prefix_lower = first ? chunk.sum_lower : merlin_fma_down_add(state->prefix_lower, chunk.sum_lower);
  state->prefix_upper = first ? chunk.sum_upper : merlin_fma_up_add(state->prefix_upper, chunk.sum_upper);
  if (!MERLIN_SOURCE_ISFINITE(state->prefix_lower) || !MERLIN_SOURCE_ISFINITE(state->prefix_upper)) {
    state->valid = 0;
    return 0;
  }
  return 1;
}

static inline int merlin_fma_bound_finish(const merlin_fma_bound *state,
                                        float *lower, float *upper) {
  if (!state->valid) return 0;
  const double radius = merlin_fma_up_add(state->representation_error,
                                        state->rounding_error);
  const double lo = merlin_fma_down_add(state->prefix_lower, -radius);
  const double hi = merlin_fma_up_add(state->prefix_upper, radius);
  if (!MERLIN_SOURCE_ISFINITE(lo) || !MERLIN_SOURCE_ISFINITE(hi) || lo <= -(double)FLT_MAX || hi >= (double)FLT_MAX)
    return 0;
#if defined(MERLIN_ENABLE_EXACT_BOUND_CONVERSION)
 *lower=MERLIN_F32_EXACT_FLOOR_FROM_F64(lo);
 *upper=MERLIN_F32_EXACT_CEIL_FROM_F64(hi);
#else
  *lower = (float)lo;
  *upper = (float)hi;
  if ((double)*lower > lo) *lower = merlin_fma_next_down_f32(*lower);
  if ((double)*upper < hi) *upper = merlin_fma_next_up_f32(*upper);
#endif
  return 1;
}

/* Cheaper, coarser alternative for ONE zero-seeded chunk. Eligibility is the
 * untouched result of bound_begin; no incoming prefix/error is accepted. */
static inline int merlin_fma_zero_chunk_gamma(
    const merlin_fma_bound *eligibility, merlin_fma_chunk chunk,
    float *lower, float *upper) {
  if (!eligibility->valid || eligibility->prefix_lower != 0.0 ||
      eligibility->prefix_upper != 0.0 || eligibility->rounding_error != 0.0 ||
      eligibility->representation_error != 0.0 || chunk.length == 0 ||
      chunk.length >= ((size_t)1 << 24) ||
      !MERLIN_SOURCE_ISFINITE(chunk.sum_lower) || !MERLIN_SOURCE_ISFINITE(chunk.sum_upper) ||
      !MERLIN_SOURCE_ISFINITE(chunk.absolute_upper) || !MERLIN_SOURCE_ISFINITE(chunk.representation_error_upper) ||
      chunk.sum_lower > chunk.sum_upper || chunk.absolute_upper < 0.0 ||
      chunk.representation_error_upper < 0.0 ||
      chunk.sum_lower > chunk.absolute_upper || chunk.sum_upper < -chunk.absolute_upper)
    return 0;
  const double magnitude = merlin_fma_up_add(merlin_fma_up_mul(0.5,
      merlin_fma_up_add(MERLIN_SOURCE_F64_MAX(MERLIN_SOURCE_F64_ABS(chunk.sum_lower),MERLIN_SOURCE_F64_ABS(chunk.sum_upper)),
                        chunk.absolute_upper)),chunk.representation_error_upper);
  const double lu = (double)chunk.length * 0x1p-24;
  const double gamma = merlin_fma_next_up(lu/(1.0-lu));
  const double eta = merlin_fma_next_up((double)chunk.length*0x1p-150/(1.0-lu));
  const double error = merlin_fma_up_add(merlin_fma_up_mul(gamma,magnitude),eta);
  const double maximum = merlin_fma_up_add(magnitude,error);
  if (!MERLIN_SOURCE_ISFINITE(maximum) || maximum >= (double)FLT_MAX) return 0;
  const merlin_fma_bound result = {chunk.sum_lower,chunk.sum_upper,
      chunk.representation_error_upper,error,1};
  return merlin_fma_bound_finish(&result,lower,upper);
}

/* Length-dependent constants may be prepared once for a tensor reduction.
 * The caller must pass the unmodified result of prepare; as with eligibility
 * and chunk summaries, this is a semantic contract rather than an opaque seal.
 * Preparation/apply require stable RNE/gradual-underflow eligibility. Floating
 * exception/errno side effects of repeated constant evaluation are unobserved.
 */
typedef struct {
  size_t length;
  double gamma_upper, subnormal_error_upper;
  int valid;
} merlin_fma_zero_gamma_plan;

static inline merlin_fma_zero_gamma_plan merlin_fma_zero_gamma_prepare(
    const merlin_fma_bound *eligibility, size_t length) {
  if (!eligibility || !eligibility->valid || eligibility->prefix_lower != 0.0 ||
      eligibility->prefix_upper != 0.0 || eligibility->rounding_error != 0.0 ||
      eligibility->representation_error != 0.0 || !length ||
      length >= ((size_t)1 << 24)) return (merlin_fma_zero_gamma_plan){0,0,0,0};
  const double lu = (double)length * 0x1p-24;
  return (merlin_fma_zero_gamma_plan){length,
      merlin_fma_next_up(lu/(1.0-lu)),
      merlin_fma_next_up((double)length*0x1p-150/(1.0-lu)),1};
}

static inline int merlin_fma_zero_gamma_apply(
    const merlin_fma_zero_gamma_plan *plan, merlin_fma_chunk chunk,
    float *lower, float *upper) {
  if (!plan || !plan->valid || !plan->length ||
      plan->length >= ((size_t)1 << 24) || chunk.length != plan->length ||
      !MERLIN_SOURCE_ISFINITE(plan->gamma_upper) || !MERLIN_SOURCE_ISFINITE(plan->subnormal_error_upper) ||
      plan->gamma_upper <= 0.0 || plan->subnormal_error_upper <= 0.0 ||
      !MERLIN_SOURCE_ISFINITE(chunk.sum_lower) || !MERLIN_SOURCE_ISFINITE(chunk.sum_upper) ||
      !MERLIN_SOURCE_ISFINITE(chunk.absolute_upper) || !MERLIN_SOURCE_ISFINITE(chunk.representation_error_upper) ||
      chunk.sum_lower > chunk.sum_upper || chunk.absolute_upper < 0.0 ||
      chunk.representation_error_upper < 0.0 ||
      chunk.sum_lower > chunk.absolute_upper || chunk.sum_upper < -chunk.absolute_upper)
    return 0;
  const double magnitude = merlin_fma_up_add(merlin_fma_up_mul(0.5,
      merlin_fma_up_add(MERLIN_SOURCE_F64_MAX(MERLIN_SOURCE_F64_ABS(chunk.sum_lower),MERLIN_SOURCE_F64_ABS(chunk.sum_upper)),
                        chunk.absolute_upper)),chunk.representation_error_upper);
  const double error = merlin_fma_up_add(
      merlin_fma_up_mul(plan->gamma_upper,magnitude),plan->subnormal_error_upper);
  const double maximum = merlin_fma_up_add(magnitude,error);
  if (!MERLIN_SOURCE_ISFINITE(maximum) || maximum >= (double)FLT_MAX) return 0;
  const merlin_fma_bound result = {chunk.sum_lower,chunk.sum_upper,
      chunk.representation_error_upper,error,1};
  return merlin_fma_bound_finish(&result,lower,upper);
}

/* Explicit batch admission hoists only immutable plan validation. The caller
 * passes the unmodified result and preserves the admitted floating environment.
 * Every dynamic chunk, overflow and output-enclosure check remains active. */
typedef struct {
 size_t length;
 double gamma_upper, subnormal_error_upper;
 int valid;
} merlin_fma_zero_gamma_batch;
static inline merlin_fma_zero_gamma_batch merlin_fma_zero_gamma_batch_prepare(
 const merlin_fma_zero_gamma_plan *plan) {
 if(!plan||!plan->valid||!plan->length||plan->length>=((size_t)1<<24)||
    !MERLIN_SOURCE_ISFINITE(plan->gamma_upper)||!MERLIN_SOURCE_ISFINITE(plan->subnormal_error_upper)||
    plan->gamma_upper<=0||plan->subnormal_error_upper<=0)
   return (merlin_fma_zero_gamma_batch){0,0,0,0};
 return (merlin_fma_zero_gamma_batch){plan->length,plan->gamma_upper,
   plan->subnormal_error_upper,1};
}
static inline int merlin_fma_zero_gamma_batch_apply(
 const merlin_fma_zero_gamma_batch *plan,merlin_fma_chunk chunk,
 float *lower,float *upper) {
 if(!plan||!plan->valid||chunk.length!=plan->length||
      !MERLIN_SOURCE_ISFINITE(chunk.sum_lower) || !MERLIN_SOURCE_ISFINITE(chunk.sum_upper) ||
      !MERLIN_SOURCE_ISFINITE(chunk.absolute_upper) || !MERLIN_SOURCE_ISFINITE(chunk.representation_error_upper) ||
      chunk.sum_lower > chunk.sum_upper || chunk.absolute_upper < 0.0 ||
      chunk.representation_error_upper < 0.0 ||
      chunk.sum_lower > chunk.absolute_upper || chunk.sum_upper < -chunk.absolute_upper)
    return 0;
  const double magnitude = merlin_fma_up_add(merlin_fma_up_mul(0.5,
      merlin_fma_up_add(MERLIN_SOURCE_F64_MAX(MERLIN_SOURCE_F64_ABS(chunk.sum_lower),MERLIN_SOURCE_F64_ABS(chunk.sum_upper)),
                        chunk.absolute_upper)),chunk.representation_error_upper);
  const double error = merlin_fma_up_add(
      merlin_fma_up_mul(plan->gamma_upper,magnitude),plan->subnormal_error_upper);
  const double maximum = merlin_fma_up_add(magnitude,error);
  if (!MERLIN_SOURCE_ISFINITE(maximum) || maximum >= (double)FLT_MAX) return 0;
  const merlin_fma_bound result = {chunk.sum_lower,chunk.sum_upper,
      chunk.representation_error_upper,error,1};
  return merlin_fma_bound_finish(&result,lower,upper);
}


/* Division-free alternative for ONE zero-seeded chunk. If h bounds the half
 * ulp of every input whose magnitude is <= M+n*h, induction gives cumulative
 * rounding error <= n*h. Start at h(M), then check the enlarged binade. A
 * second enlarged-radius check is mandatory; failure selects source replay.
 * This omits the min(gamma,half-ulp) refinement, so may certify fewer outputs.
 */
static inline int merlin_fma_zero_chunk_half_ulp(
    const merlin_fma_bound *eligibility, merlin_fma_chunk chunk,
    float *lower, float *upper) {
  if (!eligibility->valid || eligibility->prefix_lower != 0.0 ||
      eligibility->prefix_upper != 0.0 || eligibility->rounding_error != 0.0 ||
      eligibility->representation_error != 0.0 || chunk.length == 0 ||
      chunk.length >= ((size_t)1 << 24) ||
      !MERLIN_SOURCE_ISFINITE(chunk.sum_lower) || !MERLIN_SOURCE_ISFINITE(chunk.sum_upper) ||
      !MERLIN_SOURCE_ISFINITE(chunk.absolute_upper) || !MERLIN_SOURCE_ISFINITE(chunk.representation_error_upper) ||
      chunk.sum_lower > chunk.sum_upper || chunk.absolute_upper < 0.0 ||
      chunk.representation_error_upper < 0.0 ||
      chunk.sum_lower > chunk.absolute_upper || chunk.sum_upper < -chunk.absolute_upper)
    return 0;
  const double magnitude = merlin_fma_up_add(merlin_fma_up_mul(0.5,
      merlin_fma_up_add(MERLIN_SOURCE_F64_MAX(MERLIN_SOURCE_F64_ABS(chunk.sum_lower),MERLIN_SOURCE_F64_ABS(chunk.sum_upper)),
                        chunk.absolute_upper)),chunk.representation_error_upper);
  if (!MERLIN_SOURCE_ISFINITE(magnitude) || magnitude >= (double)FLT_MAX) return 0;
  const double provisional = merlin_fma_up_add(magnitude,
      merlin_fma_up_mul((double)chunk.length,merlin_fma_half_ulp(magnitude)));
  if (!MERLIN_SOURCE_ISFINITE(provisional) || provisional >= (double)FLT_MAX) return 0;
  const double half_ulp = merlin_fma_half_ulp(provisional);
  const double error = merlin_fma_up_mul((double)chunk.length,half_ulp);
  const double maximum = merlin_fma_up_add(magnitude,error);
  if (!MERLIN_SOURCE_ISFINITE(maximum) || maximum >= (double)FLT_MAX ||
      merlin_fma_half_ulp(maximum) > half_ulp) return 0;
  const merlin_fma_bound result = {chunk.sum_lower,chunk.sum_upper,
      chunk.representation_error_upper,error,1};
  return merlin_fma_bound_finish(&result,lower,upper);
}


#endif
