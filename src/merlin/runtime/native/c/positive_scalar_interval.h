#ifndef MERLIN_POSITIVE_SCALAR_INTERVAL_H
#define MERLIN_POSITIVE_SCALAR_INTERVAL_H
#include "source_f32_math.h"
#include "f32_interval_endpoint.h"

/* Optional source-operation specializations under the existing interval
 * environment contract. A positive finite scalar preserves endpoint order.
 * Scalar multiplication uses the original multiply, preserving point signed
 * zeros. Scalar FMA retains explicit fused source arithmetic. Invalid arithmetic
 * refuses. No source, numerical policy or target choice is inferred here. */
static inline merlin_f32_interval merlin_interval_positive_scale(
    merlin_f32_interval value, float factor) {
  if (!value.valid || !MERLIN_SOURCE_ISFINITE(factor) || !(factor > 0))
    return merlin_interval_bad();
  return merlin_interval(value.lo * factor, value.hi * factor);
}

/* Explicit source multiply also admits a zero factor. All returned endpoints
 * retain their own source zero sign; no multiply is changed into FMA(x,c,+0).
 * The mathematical zero interval may contain either source signed zero. */
static inline merlin_f32_interval merlin_interval_nonnegative_scale(
    merlin_f32_interval value, float factor) {
  if (!value.valid || !MERLIN_SOURCE_ISFINITE(factor) || !(factor >= 0))
    return merlin_interval_bad();
  return merlin_interval(value.lo * factor, value.hi * factor);
}

/* A finite strictly positive RHS interval needs two sign-selected SOURCE
 * products. The source operation is multiplication, not an FMA with +0. */
static inline merlin_f32_interval merlin_interval_positive_rhs_product(
    merlin_f32_interval value, merlin_f32_interval factor) {
  if (!value.valid || !factor.valid || !(factor.lo > 0))
    return merlin_interval_bad();
  return merlin_interval(value.lo * (value.lo < 0 ? factor.hi : factor.lo),
                         value.hi * (value.hi < 0 ? factor.lo : factor.hi));
}

static inline merlin_f32_interval merlin_interval_scalar_fma(
    float factor, merlin_f32_interval value, merlin_f32_interval addend) {
  if (!MERLIN_SOURCE_ISFINITE(factor) || !(factor > 0))
    return merlin_interval_fma(merlin_interval_point(factor), value, addend);
  if (!value.valid || !addend.valid) return merlin_interval_bad();
  return merlin_interval(MERLIN_SOURCE_F32_FMA(factor, value.lo, addend.lo),
                         MERLIN_SOURCE_F32_FMA(factor, value.hi, addend.hi));
}
#endif
