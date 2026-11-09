#ifndef MERLIN_PREPARED_BF16_INTERVAL_H
#define MERLIN_PREPARED_BF16_INTERVAL_H
#include "f32_interval_endpoint.h"
/* Private value snapshot, not an alias to mutable interval storage. The caller
 * refreshes this snapshot after replacing/refining either source endpoint.
 * Equal rounded endpoint bits prove the rounded source midpoint has that same
 * bit pattern: binary32/f64 rounding and finite BF16 RNE are monotone. Distinct
 * signed zeros, unordered/nonfinite endpoints retain the original midpoint.
 * Stable source rounding and unobserved nontrapping flags are required. */
typedef struct { float lo,hi; uint32_t low_bits,high_bits; } merlin_bf16_interval_bins;
static inline merlin_bf16_interval_bins merlin_bf16_interval_prepare(merlin_f32_interval x) {
 float lo=merlin_interval_bf16(x.lo),hi=merlin_interval_bf16(x.hi);
 return (merlin_bf16_interval_bins){lo,hi,merlin_interval_bits(lo),merlin_interval_bits(hi)};
}
static inline float merlin_bf16_interval_midpoint(merlin_f32_interval x,
    merlin_bf16_interval_bins bins) {
 if(bins.low_bits==bins.high_bits && MERLIN_SOURCE_ISFINITE(x.lo) &&
    MERLIN_SOURCE_ISFINITE(x.hi) && x.lo<=x.hi)return bins.lo;
 return merlin_interval_bf16((float)(((double)x.lo+x.hi)*.5));
}
/* Private producer result: x.valid comes from this runtime's finite ordered
 * interval constructors or an admitted finite source polynomial domain; bins
 * is the snapshot of exactly x, refreshed after replay. Distinct signed zeros,
 * nonfinite BF16 outputs and unknown/ambiguous bins refuse. No source-FMA
 * exactness is inferred from equal reconstructed values. */
typedef struct { float value; int valid; } merlin_bf16_exact_point;
static inline merlin_bf16_exact_point merlin_bf16_exact_point_finish(
    merlin_f32_interval x, merlin_bf16_interval_bins bins) {
 if(!x.valid || bins.low_bits!=bins.high_bits ||
    (bins.low_bits & UINT32_C(0x7f800000))==UINT32_C(0x7f800000))
  return (merlin_bf16_exact_point){0,0};
 return (merlin_bf16_exact_point){bins.lo,1};
}
/* Constructed only by the compiler-proved complete private producer/copy.
 * The epoch lives through exactly one synchronous product call; source is
 * immutable, source/output/plane storage is disjoint, and no handle escapes.
 * This proves source lower=value=upper, not exact radix reconstruction. */
typedef struct { const float *source; size_t count; const void *epoch; }
 merlin_source_point_span;
static inline int merlin_source_point_span_matches(
    const merlin_source_point_span *p,const float *source,size_t count,
    const void *epoch) {
 return p && source && count && epoch && p->source==source &&
        p->count==count && p->epoch==epoch;
}
#endif
