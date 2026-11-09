#ifndef MERLIN_FMA_PRODUCT_NORMS_H
#define MERLIN_FMA_PRODUCT_NORMS_H

/* Outward metadata for product-sum certificates. These row/column summaries
 * cost O(M*K+N*K); forming all pair bounds costs O(M*N), never O(M*N*K).
 * IEEE binary32 inputs, binary64 metadata, strict RNE and correctly rounded
 * binary64 sqrt are required. Callers use ordered_fma_bounds.h eligibility.
 */
#include "source_f32_math.h"
#include "ordered_fma_bounds.h"

typedef struct {
  double absolute_sum, absolute_max, square_sum;
  double original_max, error_sum, error_max;
  double euclidean_upper;
} merlin_fma_operand_norm;

static inline int merlin_fma_operand_summarize(
    const float *original, const float *reconstructed, size_t length,
    size_t original_stride, size_t reconstructed_stride,
    merlin_fma_operand_norm *out) {
  *out = (merlin_fma_operand_norm){0, 0, 0, 0, 0, 0, 0};
  if (!length || length >= ((size_t)1 << 24)) return 0;
  for (size_t z = 0; z < length; ++z) {
    const double a = original[z * original_stride];
    const double ar = reconstructed[z * reconstructed_stride];
    if (!MERLIN_SOURCE_ISFINITE(a) || !MERLIN_SOURCE_ISFINITE(ar)) return 0;
    const double error = merlin_fma_next_up(MERLIN_SOURCE_F64_ABS(a - ar));
    out->absolute_sum = merlin_fma_up_add(out->absolute_sum, MERLIN_SOURCE_F64_ABS(ar));
    out->square_sum = merlin_fma_up_add(out->square_sum, merlin_fma_up_mul(ar, ar));
    out->absolute_max = MERLIN_SOURCE_F64_MAX(out->absolute_max, MERLIN_SOURCE_F64_ABS(ar));
    out->original_max = MERLIN_SOURCE_F64_MAX(out->original_max, MERLIN_SOURCE_F64_ABS(a));
    out->error_sum = merlin_fma_up_add(out->error_sum, error);
    out->error_max = MERLIN_SOURCE_F64_MAX(out->error_max, error);
  }
  /* Factor Cauchy-Schwarz into row/column metadata: sqrt is O(M+N),
   * never recomputed for each of the M*N output pairs. */
  out->euclidean_upper = merlin_fma_next_up(sqrt(out->square_sum));
  return 1;
}

/* Internal adjacency for proved nonnegative, finite binary64 values. The
 * caller below validates finite binary32 operands and length < 2^24. Thus
 * absolute/error sums remain below 2^154 and square sums below 2^282, including
 * their upward rounding increments. Products, sqrt and every input to this
 * helper are nonnegative and finite; +0 maps to the least positive subnormal.
 * No signed-zero, NaN or infinity shortcut is granted to arbitrary callers. */
static inline double merlin_fma_norm_next_up_nonnegative(double value) {
  uint64_t raw; MERLIN_SOURCE_BITCAST_COPY(&raw, &value, sizeof(raw));
  ++raw; MERLIN_SOURCE_BITCAST_COPY(&value, &raw, sizeof(raw)); return value;
}

/* Explicit alternative to the original summary. All seven returned fields
 * match its bit patterns, including upward increments on exact zero. The
 * caller supplies a valid IEEE/RNE eligibility token and keeps the floating
 * environment stable through the call/batch. Input storage remains valid and
 * nonoverlapping with the output. Floating-point exception flags are unobserved
 * and arithmetic is nontrapping; stable rounding alone does not establish that
 * contract. Finite checks and failure behavior remain;
 * this API changes no default caller or certificate/numeric policy. */
static inline int merlin_fma_operand_summarize_finite(
    const merlin_fma_bound *eligibility,
    const float *original, const float *reconstructed, size_t length,
    size_t original_stride, size_t reconstructed_stride,
    merlin_fma_operand_norm *out) {
  *out = (merlin_fma_operand_norm){0, 0, 0, 0, 0, 0, 0};
  if (!eligibility || !eligibility->valid || !length ||
      length >= ((size_t)1 << 24)) return 0;
  for (size_t z = 0; z < length; ++z) {
    const double a = original[z * original_stride];
    const double ar = reconstructed[z * reconstructed_stride];
    if (!MERLIN_SOURCE_ISFINITE(a) || !MERLIN_SOURCE_ISFINITE(ar)) return 0;
    const double error = merlin_fma_norm_next_up_nonnegative(MERLIN_SOURCE_F64_ABS(a - ar));
    out->absolute_sum = merlin_fma_norm_next_up_nonnegative(out->absolute_sum + MERLIN_SOURCE_F64_ABS(ar));
    out->square_sum = merlin_fma_norm_next_up_nonnegative(out->square_sum +
        merlin_fma_norm_next_up_nonnegative(ar * ar));
    out->absolute_max = MERLIN_SOURCE_F64_MAX(out->absolute_max, MERLIN_SOURCE_F64_ABS(ar));
    out->original_max = MERLIN_SOURCE_F64_MAX(out->original_max, MERLIN_SOURCE_F64_ABS(a));
    out->error_sum = merlin_fma_norm_next_up_nonnegative(out->error_sum + error);
    out->error_max = MERLIN_SOURCE_F64_MAX(out->error_max, error);
  }
  out->euclidean_upper = merlin_fma_norm_next_up_nonnegative(sqrt(out->square_sum));
  return 1;
}

/* Explicit representation-error-only requirements. This distinct type cannot
 * supply the absolute-product/Holder consumer: no square sum, reconstructed
 * maximum or Euclidean norm is computed. The four retained fields reproduce
 * the full summary bitwise under the finite-summary eligibility contract.
 * Use when an independent absolute-product bound is already available. Omitting
 * square/sqrt work may change exception flags: these must be unobserved, and
 * arithmetic must be nontrapping, as required by the finite-summary contract. */
typedef struct {
  double absolute_sum, original_max, error_sum, error_max;
} merlin_fma_representation_norm;

static inline int merlin_fma_representation_summarize_finite(
    const merlin_fma_bound *eligibility,
    const float *original, const float *reconstructed, size_t length,
    size_t original_stride, size_t reconstructed_stride,
    merlin_fma_representation_norm *out) {
  *out = (merlin_fma_representation_norm){0, 0, 0, 0};
  if (!eligibility || !eligibility->valid || !length ||
      length >= ((size_t)1 << 24)) return 0;
  for (size_t z = 0; z < length; ++z) {
    const double a = original[z * original_stride];
    const double ar = reconstructed[z * reconstructed_stride];
    if (!MERLIN_SOURCE_ISFINITE(a) || !MERLIN_SOURCE_ISFINITE(ar)) return 0;
    const double error = merlin_fma_norm_next_up_nonnegative(MERLIN_SOURCE_F64_ABS(a - ar));
    out->absolute_sum = merlin_fma_norm_next_up_nonnegative(out->absolute_sum + MERLIN_SOURCE_F64_ABS(ar));
    out->original_max = MERLIN_SOURCE_F64_MAX(out->original_max, MERLIN_SOURCE_F64_ABS(a));
    out->error_sum = merlin_fma_norm_next_up_nonnegative(out->error_sum + error);
    out->error_max = MERLIN_SOURCE_F64_MAX(out->error_max, error);
  }
  return 1;
}

static inline double merlin_fma_representation_pair_bound(
    merlin_fma_representation_norm a, merlin_fma_representation_norm b) {
  return merlin_fma_up_add(merlin_fma_up_mul(a.error_sum, b.original_max),
                         merlin_fma_up_mul(a.absolute_sum, b.error_max));
}

static inline double merlin_fma_representation_bound(
    merlin_fma_operand_norm a, merlin_fma_operand_norm b) {
  return merlin_fma_up_add(merlin_fma_up_mul(a.error_sum, b.original_max),
                           merlin_fma_up_mul(a.absolute_sum, b.error_max));
}

static inline void merlin_fma_product_summarize(
    merlin_fma_operand_norm a, merlin_fma_operand_norm b,
    double *absolute_upper, double *representation_error_upper) {
  /* |ab-ar*br| <= |a-ar|*|b| + |ar|*|b-br|. */
  *representation_error_upper = merlin_fma_representation_bound(a,b);
  /* Three independently valid Holder bounds on sum(abs(ar*br)). */
  const double cs = merlin_fma_up_mul(a.euclidean_upper,b.euclidean_upper);
  *absolute_upper = MERLIN_SOURCE_F64_MIN(cs, MERLIN_SOURCE_F64_MIN(
      merlin_fma_up_mul(a.absolute_sum, b.absolute_max),
      merlin_fma_up_mul(a.absolute_max, b.absolute_sum)));
}

#endif
