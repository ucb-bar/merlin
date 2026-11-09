#ifndef MERLIN_PREPARED_BIT_POLYNOMIAL_H
#define MERLIN_PREPARED_BIT_POLYNOMIAL_H
#include "source_f32_math.h"
#include "f32_interval_endpoint.h"
/* Immutable prepared value. Preparation and application require the original
 * interval environment contract. Source coefficients are copied; callers must
 * not mutate the prepared value. Unknown stage signs retain the checked path.
 * This caches validation and sign facts, never changes the endpoint policy. */
typedef struct { merlin_bit_polynomial_plan source; signed char signs[3]; int valid; } merlin_prepared_bit_polynomial;
static inline merlin_prepared_bit_polynomial merlin_bit_polynomial_prepare(const merlin_bit_polynomial_plan *p){
 merlin_prepared_bit_polynomial out={0};
 if(!p||!MERLIN_SOURCE_ISFINITE(p->cutoff)||!MERLIN_SOURCE_ISFINITE(p->scale)||!(p->scale>0)||!MERLIN_SOURCE_ISFINITE(p->bit_multiplier)||!MERLIN_SOURCE_ISFINITE(p->bit_bias))return out;
 for(int i=0;i<4;i++)if(!MERLIN_SOURCE_ISFINITE(p->coefficients[i]))return out;
 out.source=*p;
 merlin_f32_interval f=merlin_interval(0,1),poly=merlin_interval_point(p->coefficients[0]);
 for(int i=0;i<3;i++){
  out.signs[i]=poly.valid?(poly.hi<0?-1:poly.lo>0?1:0):0;
  poly=merlin_interval_fma(f,poly,merlin_interval_point(p->coefficients[i+1]));
 }
 out.valid=1;return out;
}
static inline merlin_f32_interval merlin_prepared_polynomial_stage(merlin_f32_interval f,merlin_f32_interval p,float c,int sign){
 if(!f.valid||!p.valid||f.lo<0||f.hi>1||sign==0)return merlin_interval_fma(f,p,merlin_interval_point(c));
 float lo,hi;
 if(sign<0){lo=MERLIN_SOURCE_F32_FMA(f.hi,p.lo,c);hi=MERLIN_SOURCE_F32_FMA(f.lo,p.hi,c);}
 else{lo=MERLIN_SOURCE_F32_FMA(f.lo,p.lo,c);hi=MERLIN_SOURCE_F32_FMA(f.hi,p.hi,c);}
 /* Retain the original corner evaluation when signed zero is observable. */
 if(lo==0||hi==0)return merlin_interval_fma(f,p,merlin_interval_point(c));
 return merlin_interval(lo,hi);
}
static inline merlin_f32_interval merlin_prepared_bit_polynomial_apply(
    merlin_f32_interval x, const merlin_prepared_bit_polynomial *prepared) {
  if(!x.valid || !prepared || !prepared->valid)return merlin_interval_bad();
  const merlin_bit_polynomial_plan *p=&prepared->source;
  if(x.hi<p->cutoff)return merlin_interval_point(0);
  int includes_zero=x.lo<p->cutoff;
  x.lo=MERLIN_SOURCE_F32_MAX(x.lo,p->cutoff);
  merlin_f32_interval z=merlin_interval_mul(x,merlin_interval_point(p->scale));
  if(!z.valid)return merlin_interval_bad();
  double first=floor((double)z.lo),last=floor((double)z.hi);
  if(first < -16777216 || last > 16777215 || last-first>256)return merlin_interval_bad();
  float lo=includes_zero?0:INFINITY,hi=includes_zero?0:-INFINITY;
  for(int k=(int)first;k<=(int)last;k++) {
    float left=MERLIN_SOURCE_F32_MAX(z.lo,(float)k);
    float right=MERLIN_SOURCE_F32_MIN(z.hi,merlin_fma_next_down_f32((float)(k+1)));
    if(left>right)continue;
    merlin_f32_interval segment=merlin_interval(left,right);
    merlin_f32_interval f=merlin_interval_sub(segment,merlin_interval_point((float)k));
    merlin_f32_interval poly=merlin_interval_point(p->coefficients[0]);
    for(int i=1;i<4;i++)poly=merlin_prepared_polynomial_stage(f,poly,p->coefficients[i],prepared->signs[i-1]);
    merlin_f32_interval t=merlin_interval_fma(merlin_interval_point(p->bit_multiplier),merlin_interval_sub(segment,poly),merlin_interval_point(p->bit_bias));
    /* Positive finite IEEE encoding is monotone in its integer word. */
    if(!t.valid || t.lo<0 || t.hi>=2139095040.0f)return merlin_interval_bad();
    float l=merlin_interval_float((uint32_t)(int32_t)t.lo);
    float h=merlin_interval_float((uint32_t)(int32_t)t.hi);
    lo=MERLIN_SOURCE_F32_MIN(lo,l);hi=MERLIN_SOURCE_F32_MAX(hi,h);
  }
  return merlin_interval(lo,hi);
}

#endif
