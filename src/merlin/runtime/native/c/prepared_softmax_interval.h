#ifndef MERLIN_PREPARED_SOFTMAX_INTERVAL_H
#define MERLIN_PREPARED_SOFTMAX_INTERVAL_H
#include "monotone_bit_polynomial.h"

/* Private source-schedule capability, not an unchecked public interval flag.
 * The owner keeps the plan, endpoint arrays and RNE environment immutable.
 * Active dot spans must be admitted before use. Source replay still checks its
 * result against those spans. A nonfinite/unsupported plan uses checked code.
 * These positive-lane operations do not authorize signed PV arithmetic.
 */
typedef struct { float lo, hi; } merlin_soft_interval;
typedef struct {
  float scale, score_magnitude, probability_upper, denominator_upper;
  size_t chunk, lanes, chunks;
  int valid;
} merlin_prepared_softmax_domain;

static inline merlin_prepared_softmax_domain merlin_softmax_domain_prepare(
    const merlin_fma_bound *environment,
    const merlin_monotone_bit_polynomial *polynomial, float scale,
    size_t chunk, size_t lanes, size_t chunks) {
  merlin_prepared_softmax_domain out={0};
  if(!environment||!environment->valid||!polynomial||!polynomial->fast_valid||
     polynomial->upper!=0||!MERLIN_SOURCE_ISFINITE(scale)||!(scale>0)||
     !chunk||!lanes||(lanes&(lanes-1))||!chunks||chunk>SIZE_MAX-(lanes-1))return out;
  const merlin_bit_polynomial_plan *p=&polynomial->checked.source;
  if(!MERLIN_SOURCE_ISFINITE(p->cutoff)||p->cutoff>0)return out;
  /* RNE is monotone. This bounds every finite source endpoint, not samples. */
  float magnitude=FLT_MAX*scale;
  if(!MERLIN_SOURCE_ISFINITE(magnitude)||
     !MERLIN_SOURCE_ISFINITE(magnitude+magnitude))return out;
  /* Uniform source-q range: each original Horner argument lies in [0,1].
   * Cover the largest optional source-q outward budget, not merely the source
   * polynomial's real range. Thus every returned interval endpoint is covered,
   * including narrower intervals whose enclosure error differs. */
  merlin_f32_interval fraction=merlin_interval(0,1);
  merlin_f32_interval poly=merlin_interval_point(p->coefficients[0]);
  for(int i=1;i<4;i++)poly=merlin_interval_fma(fraction,poly,
                                           merlin_interval_point(p->coefficients[i]));
  float first=p->cutoff*p->scale;
  merlin_f32_interval q=merlin_interval_sub(merlin_interval(first,0),poly);
  if(!q.valid)return out;
  double error=0;
  for(int i=0;i<26;i++){
    if(!MERLIN_SOURCE_ISFINITE(polynomial->source_error[i])||polynomial->source_error[i]<0)return out;
    error=MERLIN_SOURCE_F64_MAX(error,polynomial->source_error[i]);
  }
  error=merlin_fma_up_mul(2,error);
  float lower=merlin_fma_next_down_f32((float)merlin_fma_next_down((double)q.lo-error));
  float upper=merlin_fma_next_up_f32((float)merlin_fma_next_up((double)q.hi+error));
  if(!MERLIN_SOURCE_ISFINITE(lower)||!MERLIN_SOURCE_ISFINITE(upper))return out;
  float encoded=MERLIN_SOURCE_F32_FMA(p->bit_multiplier,upper,p->bit_bias);
  if(!MERLIN_SOURCE_ISFINITE(encoded)||encoded<0)return out;
  uint32_t word=encoded>=2139095040.0f?UINT32_C(2139095039):(uint32_t)(int32_t)encoded;
  float probability=merlin_interval_float(word);
  if(!MERLIN_SOURCE_ISFINITE(probability)||probability<0)return out;
  /* Same positive lane additions/tree/FMA schedule on constant upper data.
   * Padding short lanes upward is sound by monotonicity, without reassociating
   * any runtime source operation. Every prefix is checked during preparation. */
  size_t visits=(chunk+lanes-1)/lanes;
  float lane=0;
  for(size_t i=0;i<visits;i++){
    lane=lane+probability;if(!MERLIN_SOURCE_ISFINITE(lane))return out;
  }
  for(size_t n=lanes/2;n;n/=2){
    lane=lane+lane;if(!MERLIN_SOURCE_ISFINITE(lane))return out;
  }
  float denominator=0;
  for(size_t i=0;i<chunks;i++){
    denominator=MERLIN_SOURCE_F32_FMA(1.0f,denominator,lane);
    if(!MERLIN_SOURCE_ISFINITE(denominator))return out;
  }
  return (merlin_prepared_softmax_domain){scale,magnitude,probability,denominator,
                                        chunk,lanes,chunks,1};
}

/* Distinct word-enclosure admission. The checked source-q probability cap
 * already encloses every original point word. apply_words can increase such
 * a word by at most word_budget (then clamps to finite positive binary32).
 * Add that budget in integer encoding space, and replay the original positive
 * lane/tree/chunk upper schedule. No source operation is reassociated. A
 * widened cap which overflows any prefix is refused; callers retain checked
 * interval execution. This does not authorize signed PV operations. */
static inline merlin_prepared_softmax_domain merlin_softmax_word_domain_prepare(
    const merlin_fma_bound *environment,
    const merlin_monotone_bit_polynomial *polynomial, float scale,
    size_t chunk, size_t lanes, size_t chunks) {
  merlin_prepared_softmax_domain out=merlin_softmax_domain_prepare(
      environment,polynomial,scale,chunk,lanes,chunks);
  if(!out.valid)return out;
  uint64_t word=(uint64_t)merlin_interval_bits(out.probability_upper)
      +polynomial->word_budget;
  if(word>UINT32_C(2139095039))word=UINT32_C(2139095039);
  float probability=merlin_interval_float((uint32_t)word);
  size_t visits=(chunk+lanes-1)/lanes;
  float lane=0;
  for(size_t i=0;i<visits;i++){
    lane=lane+probability;
    if(!MERLIN_SOURCE_ISFINITE(lane))return (merlin_prepared_softmax_domain){0};
  }
  for(size_t n=lanes/2;n;n/=2){
    lane=lane+lane;
    if(!MERLIN_SOURCE_ISFINITE(lane))return (merlin_prepared_softmax_domain){0};
  }
  float denominator=0;
  for(size_t i=0;i<chunks;i++){
    denominator=MERLIN_SOURCE_F32_FMA(1.0f,denominator,lane);
    if(!MERLIN_SOURCE_ISFINITE(denominator))return (merlin_prepared_softmax_domain){0};
  }
  out.probability_upper=probability;out.denominator_upper=denominator;
  return out;
}

static inline int merlin_softmax_admit_active_spans(
    const merlin_prepared_softmax_domain *p,const float *lo,const float *hi,
    const unsigned char *mask,size_t count) {
  if(!p||!p->valid||!lo||!hi||!mask)return 0;
  for(size_t i=0;i<count;i++){
    if(mask[i]>1)return 0;
    if(mask[i]&&(!MERLIN_SOURCE_ISFINITE(lo[i])||!MERLIN_SOURCE_ISFINITE(hi[i])||lo[i]>hi[i]))return 0;
  }
  return 1;
}

/* Consumers require the admitted private source schedule and immutable spans.
 * mx is the checked/replayed finite source maximum in [-score_magnitude,+...].
 * All-masked tiles never invoke this operation. Endpoint products remain two
 * original multiplies followed by two original subtracts, including signed 0. */
static inline merlin_soft_interval merlin_softmax_score(
    const merlin_prepared_softmax_domain *p,float lo,float hi,float mx) {
  float low=lo*p->scale,high=hi*p->scale;
  return (merlin_soft_interval){low-mx,high-mx};
}
static inline merlin_soft_interval merlin_softmax_lane_add(
    merlin_soft_interval a,merlin_soft_interval b) {
  return (merlin_soft_interval){a.lo+b.lo,a.hi+b.hi};
}
static inline int merlin_softmax_admit_alpha(float alpha) {
  return MERLIN_SOURCE_ISFINITE(alpha)&&alpha>=0&&alpha<=1;
}
#endif
