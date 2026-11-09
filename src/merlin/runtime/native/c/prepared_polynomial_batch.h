#ifndef MERLIN_PREPARED_POLYNOMIAL_BATCH_H
#define MERLIN_PREPARED_POLYNOMIAL_BATCH_H
#include "monotone_bit_polynomial.h"
/* Four independent word enclosures. Immutable prepared plan, stable source
 * RNE, nontrapping arithmetic and unobserved exception flags are required.
 * Each Horner recurrence retains its three original rounded FMAs. No lane
 * reduction occurs here. Unsupported admission uses the scalar checked path. */
static inline void merlin_polynomial_words_four(
    const merlin_f32_interval x[4], const merlin_monotone_bit_polynomial *p,
    merlin_f32_interval out[4]) {
  if(!p || !p->fast_valid) goto checked;
  for(int i=0;i<4;i++) if(!x[i].valid || x[i].hi>p->upper) goto checked;
  const merlin_bit_polynomial_plan *s=&p->checked.source;
  float scaled[8],fraction[8],poly[8];
  for(int i=0;i<4;i++) {
    float lo=MERLIN_SOURCE_F32_MAX(x[i].lo,s->cutoff);
    float hi=MERLIN_SOURCE_F32_MAX(x[i].hi,s->cutoff);
    scaled[2*i]=lo*s->scale;scaled[2*i+1]=hi*s->scale;
  }
  for(int i=0;i<8;i++) {
    fraction[i]=scaled[i]-MERLIN_MONOTONE_F32_FLOOR(scaled[i]);
    poly[i]=s->coefficients[0];
  }
  for(int c=1;c<4;c++) {
#if defined(MERLIN_ENABLE_SOURCE_FMA_BATCH_8)
    /* Separate private local arrays keep every other lane's operands stable. */
    MERLIN_SOURCE_F32_FMA_EIGHT(fraction,poly,s->coefficients[c]);
#else
    poly[0]=MERLIN_SOURCE_F32_FMA(fraction[0],poly[0],s->coefficients[c]);
    poly[1]=MERLIN_SOURCE_F32_FMA(fraction[1],poly[1],s->coefficients[c]);
    poly[2]=MERLIN_SOURCE_F32_FMA(fraction[2],poly[2],s->coefficients[c]);
    poly[3]=MERLIN_SOURCE_F32_FMA(fraction[3],poly[3],s->coefficients[c]);
    poly[4]=MERLIN_SOURCE_F32_FMA(fraction[4],poly[4],s->coefficients[c]);
    poly[5]=MERLIN_SOURCE_F32_FMA(fraction[5],poly[5],s->coefficients[c]);
    poly[6]=MERLIN_SOURCE_F32_FMA(fraction[6],poly[6],s->coefficients[c]);
    poly[7]=MERLIN_SOURCE_F32_FMA(fraction[7],poly[7],s->coefficients[c]);
#endif
  }
  for(int i=0;i<4;i++) {
    uint32_t first=(uint32_t)(int32_t)MERLIN_SOURCE_F32_FMA(s->bit_multiplier,scaled[2*i]-poly[2*i],s->bit_bias);
    uint32_t last=(uint32_t)(int32_t)MERLIN_SOURCE_F32_FMA(s->bit_multiplier,scaled[2*i+1]-poly[2*i+1],s->bit_bias);
    if(x[i].hi<s->cutoff) {out[i]=merlin_interval_point(0);continue;}
    int includes_zero=x[i].lo<s->cutoff;
    if(x[i].lo==x[i].hi&&!includes_zero) {out[i]=merlin_interval_point(merlin_interval_float(first));continue;}
    uint32_t low=includes_zero||first<p->word_budget?0:first-p->word_budget;
    uint64_t high=(uint64_t)last+p->word_budget;
    if(high>UINT32_C(2139095039))high=UINT32_C(2139095039);
    out[i]=merlin_interval(merlin_interval_float(low),merlin_interval_float((uint32_t)high));
  }
  return;
checked:
  for(int i=0;i<4;i++)out[i]=merlin_monotone_bit_polynomial_apply_words(x[i],p);
}
#endif
