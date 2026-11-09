#ifndef MERLIN_F32_INTERVAL_ENDPOINT_H
#define MERLIN_F32_INTERVAL_ENDPOINT_H

/* Optional value certificate, not an approximation policy. Source and bound
 * execution use binary32 RNE, gradual underflow, no reassociation/contraction
 * except explicit fmaf, stable FENV and unobserved nontrapping exception flags.
 * Callers validate that environment once with merlin_fma_bound_begin().
 * Every operation encloses the corresponding rounded SOURCE operation because
 * RNE is monotone and the extrema below include every sign corner. NaN,
 * infinity and unsupported integer conversion domains refuse to source replay.
 */
#include "source_f32_math.h"
#include "ordered_fma_bounds.h"

typedef struct { float lo, hi; int valid; } merlin_f32_interval;
static inline merlin_f32_interval merlin_interval(float lo, float hi) {
  return (merlin_f32_interval){lo, hi, MERLIN_SOURCE_ISFINITE(lo) && MERLIN_SOURCE_ISFINITE(hi) && lo <= hi};
}
static inline merlin_f32_interval merlin_interval_bad(void) {
  return (merlin_f32_interval){0, 0, 0};
}
static inline merlin_f32_interval merlin_interval_point(float value) {
  return merlin_interval(value, value);
}
static inline merlin_f32_interval merlin_interval_add(merlin_f32_interval a, merlin_f32_interval b) {
  if (!a.valid || !b.valid) return merlin_interval_bad();
  return merlin_interval(a.lo + b.lo, a.hi + b.hi);
}
static inline merlin_f32_interval merlin_interval_sub(merlin_f32_interval a, merlin_f32_interval b) {
  if (!a.valid || !b.valid) return merlin_interval_bad();
  return merlin_interval(a.lo - b.hi, a.hi - b.lo);
}
static inline merlin_f32_interval merlin_interval_fma(merlin_f32_interval a, merlin_f32_interval b, merlin_f32_interval c) {
  if (!a.valid || !b.valid || !c.valid) return merlin_interval_bad();
  float av[2]={a.lo,a.hi},bv[2]={b.lo,b.hi};
  float lo=INFINITY,hi=-INFINITY;
  for (int i=0;i<2;i++) for(int j=0;j<2;j++) {
    float l=MERLIN_SOURCE_F32_FMA(av[i],bv[j],c.lo),h=MERLIN_SOURCE_F32_FMA(av[i],bv[j],c.hi);
    if(!MERLIN_SOURCE_ISFINITE(l)||!MERLIN_SOURCE_ISFINITE(h))return merlin_interval_bad();
    lo=MERLIN_SOURCE_F32_MIN(lo,l);hi=MERLIN_SOURCE_F32_MAX(hi,h);
  }
  return merlin_interval(lo,hi);
}
static inline merlin_f32_interval merlin_interval_mul(merlin_f32_interval a, merlin_f32_interval b) {
  /* fma(x,y,+0) has the same nonzero rounded value as mul; signed zero may
   * differ. Endpoint certification refuses every interval touching zero. */
  return merlin_interval_fma(a,b,merlin_interval_point(0));
}
static inline merlin_f32_interval merlin_interval_recip_positive(merlin_f32_interval a) {
  if(!a.valid || !(a.lo>0))return merlin_interval_bad();
  return merlin_interval(1.0f/a.hi,1.0f/a.lo);
}
static inline uint32_t merlin_interval_bits(float value) {
  uint32_t raw;MERLIN_SOURCE_BITCAST_COPY(&raw,&value,sizeof(raw));return raw;
}
static inline float merlin_interval_float(uint32_t raw) {
  float value;MERLIN_SOURCE_BITCAST_COPY(&value,&raw,sizeof(value));return value;
}
static inline float merlin_interval_bf16(float value) {
  uint32_t raw=merlin_interval_bits(value);
  raw=(raw+UINT32_C(0x7fff)+((raw>>16)&1))&UINT32_C(0xffff0000);
  return merlin_interval_float(raw);
}
static inline int merlin_interval_bf16_endpoint(merlin_f32_interval value,float *output) {
  if(!value.valid || (value.lo<=0 && value.hi>=0))return 0;
  float lo=merlin_interval_bf16(value.lo),hi=merlin_interval_bf16(value.hi);
  if(!MERLIN_SOURCE_ISFINITE(lo)||!MERLIN_SOURCE_ISFINITE(hi)||merlin_interval_bits(lo)!=merlin_interval_bits(hi))return 0;
  *output=lo;return 1;
}

/* Explicit bounded policy: select the caller's reconstructed center only when
 * its rounded word is in the proved source endpoint bin interval. A budget of
 * one permits exactly two adjacent finite, same-sign BF16 values; larger
 * budgets are unsupported. No observation of a reference output is involved.
 */
static inline int merlin_interval_bf16_bounded_endpoint(
    merlin_f32_interval value, float center, unsigned max_steps, float *output) {
  if (max_steps > 1 || !MERLIN_SOURCE_ISFINITE(center) || !value.valid ||
      (value.lo <= 0 && value.hi >= 0)) return 0;
  float lo = merlin_interval_bf16(value.lo);
  float hi = merlin_interval_bf16(value.hi);
  float selected = merlin_interval_bf16(center);
  if (!MERLIN_SOURCE_ISFINITE(lo) || !MERLIN_SOURCE_ISFINITE(hi) || !MERLIN_SOURCE_ISFINITE(selected) ||
      selected < lo || selected > hi) return 0;
  uint32_t l = merlin_interval_bits(lo) >> 16;
  uint32_t h = merlin_interval_bits(hi) >> 16;
  uint32_t chosen = merlin_interval_bits(selected) >> 16;
  if ((l >> 15) != (h >> 15) || (chosen >> 15) != (l >> 15)) return 0;
  if (chosen < (l < h ? l : h) || chosen > (l > h ? l : h)) return 0;
  uint32_t distance = l > h ? l - h : h - l;
  if (distance > max_steps) return 0;
  *output = selected;
  return 1;
}

/* Parameterized source bit-polynomial expression:
 * z=RNE(x*scale); f=RNE(z-floor(z)); p=HornerFMA(f, coefficients);
 * t=FMA(bit_multiplier,RNE(z-p),bit_bias); result=bitcast_i32(trunc(t)).
 * x<cutoff selects +0. This is a structural enclosure of those operations,
 * not an assumption that the polynomial or emitted approximation is monotone.
 * Integer floor segments are split exactly before evaluating dependency bounds.
 */
typedef struct {
 float cutoff,scale,coefficients[4],bit_multiplier,bit_bias;
} merlin_bit_polynomial_plan;
static inline merlin_f32_interval merlin_interval_bit_polynomial(
    merlin_f32_interval x, const merlin_bit_polynomial_plan *p) {
  if(!x.valid || !p || !MERLIN_SOURCE_ISFINITE(p->cutoff) || !MERLIN_SOURCE_ISFINITE(p->scale) || !(p->scale>0) || !MERLIN_SOURCE_ISFINITE(p->bit_multiplier) || !MERLIN_SOURCE_ISFINITE(p->bit_bias))return merlin_interval_bad();
  for(int i=0;i<4;i++)if(!MERLIN_SOURCE_ISFINITE(p->coefficients[i]))return merlin_interval_bad();
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
    for(int i=1;i<4;i++)poly=merlin_interval_fma(f,poly,merlin_interval_point(p->coefficients[i]));
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
