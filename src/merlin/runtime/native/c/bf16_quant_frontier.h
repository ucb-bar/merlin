#ifndef MERLIN_BF16_QUANT_FRONTIER_H
#define MERLIN_BF16_QUANT_FRONTIER_H
#include "source_f32_math.h"
#include <math.h>
#include <fenv.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
/* Explicit source contract: finite f32 operands, RNE, BF16 RNE after scale,
 * reciprocal, product and rounding, nontrapping/unobserved exception flags.
 * Caller proves the complete typed consumer has no other observable uses.
 * Inputs contain finite ordered BF16 endpoints; candidate must lie inside.
 * No refinement, target instruction, or hidden storage is owned here. */
typedef struct { float divisor, epsilon; int lower, upper; } merlin_bf16_quant_plan;
static inline float merlin_frontier_bf16(float value) {
 uint32_t bits; MERLIN_SOURCE_BITCAST_COPY(&bits,&value,4); bits += 0x7fffu+((bits>>16)&1u); bits &= 0xffff0000u; MERLIN_SOURCE_BITCAST_COPY(&value,&bits,4); return value;
}
static inline int merlin_frontier_is_bf16(float value) {
 uint32_t bits;MERLIN_SOURCE_BITCAST_COPY(&bits,&value,4);return MERLIN_SOURCE_ISFINITE(value)&&!(bits&0xffffu);
}
static inline int merlin_frontier_plan_valid(merlin_bf16_quant_plan p) {
 return fegetround()==FE_TONEAREST&&MERLIN_SOURCE_ISFINITE(p.divisor)&&p.divisor>0&&merlin_frontier_is_bf16(p.epsilon)&&p.epsilon>0&&p.lower>=-128&&p.upper<=127&&p.lower<=p.upper;
}
static inline float merlin_frontier_scale(float peak,merlin_bf16_quant_plan p) {
 return MERLIN_SOURCE_F32_MAX(merlin_frontier_bf16(peak/p.divisor),p.epsilon);
}
static inline int merlin_frontier_quant_prepared(float x,float inverse,merlin_bf16_quant_plan p) {
 float product=merlin_frontier_bf16(x*inverse);
 /* nearbyintf implements source ties-even under the admitted RNE mode. */
 float rounded=merlin_frontier_bf16(nearbyintf(product));
 rounded=merlin_frontier_bf16(rounded+0.0f);
 return (int)MERLIN_SOURCE_F32_MAX((float)p.lower,MERLIN_SOURCE_F32_MIN((float)p.upper,rounded));
}
static inline int merlin_frontier_quant(float x,float scale,merlin_bf16_quant_plan p) {
 return merlin_frontier_quant_prepared(x,merlin_frontier_bf16(1.0f/scale),p);
}
/* Return -1 on unsupported input, 0 if refinement is needed, 1 if every
 * consumer observation is unique. pending points to caller-owned n bytes.
 * All eligibility checks precede writes; aliasing pending with input is not
 * admitted. Candidate output is not modified. */
static inline int merlin_frontier_row(const float *low,const float *high,
 const float *candidate,size_t n,merlin_bf16_quant_plan plan,
 uint8_t *pending,float *unique_scale,size_t *pending_count) {
 if(!n||!merlin_frontier_plan_valid(plan))return -1;
 float peak_low=0,peak_high=0;
 for(size_t i=0;i<n;i++) {
  float l=low[i],h=high[i],c=candidate[i];
  if(!merlin_frontier_is_bf16(l)||!merlin_frontier_is_bf16(h)||!merlin_frontier_is_bf16(c)||l>h||c<l||c>h)return -1;
  float a=l<=0&&h>=0?0:MERLIN_SOURCE_F32_MIN(MERLIN_SOURCE_F32_ABS(l),MERLIN_SOURCE_F32_ABS(h));
  peak_low=MERLIN_SOURCE_F32_MAX(peak_low,a);peak_high=MERLIN_SOURCE_F32_MAX(peak_high,MERLIN_SOURCE_F32_MAX(MERLIN_SOURCE_F32_ABS(l),MERLIN_SOURCE_F32_ABS(h)));
 }
 float slo=merlin_frontier_scale(peak_low,plan),shi=merlin_frontier_scale(peak_high,plan);
 if(!MERLIN_SOURCE_ISFINITE(slo)||!MERLIN_SOURCE_ISFINITE(shi)||!MERLIN_SOURCE_ISFINITE(merlin_frontier_bf16(1.0f/slo)))return -1;
 int stable=slo==shi;size_t count=0;
 float inverse=merlin_frontier_bf16(1.0f/slo);
 for(size_t i=0;i<n;i++) {
  int ambiguous=stable?merlin_frontier_quant_prepared(low[i],inverse,plan)!=merlin_frontier_quant_prepared(high[i],inverse,plan):MERLIN_SOURCE_F32_MAX(MERLIN_SOURCE_F32_ABS(low[i]),MERLIN_SOURCE_F32_ABS(high[i]))>=peak_low;
  pending[i]=(uint8_t)ambiguous;count+=ambiguous;
 }
 *pending_count=count;if(stable)*unique_scale=slo;
 return stable&&!count;
}
/* Explicit caller-owned coordination across [head,row,channel] intervals.
 * refresh and refine are mathematical-provider callbacks, never accelerator
 * calls selected by this helper. On failure the caller invokes its source
 * fallback. All scratch and row ownership is explicit, with no hidden cache.
 * A stable row may be reused only because refinement is row-local and cannot
 * change previously certified rows; this is a callback admission obligation. */
typedef struct {
 size_t heads, rows, channels;
 float *lower, *upper, *candidate;
 float *row_lower, *row_upper, *row_candidate;
 uint8_t *pending, *certified;
 int (*refresh)(void *context);
 int (*refine)(void *context,size_t head,size_t row,size_t channel,int exact);
 void *context;
} merlin_frontier_storage;
static inline int merlin_frontier_complete(merlin_frontier_storage *s,
 merlin_bf16_quant_plan plan,unsigned maximum_passes) {
 if(!s||!s->lower||!s->upper||!s->candidate||!s->row_lower||
    !s->row_upper||!s->row_candidate||!s->pending||!s->certified||
    !s->refresh||!s->refine||!s->heads||!s->rows||!s->channels||!maximum_passes||
    s->heads>SIZE_MAX/s->channels||s->rows>SIZE_MAX/(s->heads*s->channels)||
    !merlin_frontier_plan_valid(plan))return 0;
 size_t width=s->heads*s->channels;
 if(width>SIZE_MAX/sizeof(float)||s->rows>SIZE_MAX/(width*sizeof(float)))return 0;
 for(size_t row=0;row<s->rows;row++)s->certified[row]=0;
 for(unsigned pass=0;pass<maximum_passes;pass++) {
  if(!s->refresh(s->context))return 0;
  size_t remaining=0;
  for(size_t row=0;row<s->rows;row++) {
   if(s->certified[row])continue;
   for(size_t h=0;h<s->heads;h++)for(size_t c=0;c<s->channels;c++) {
    size_t local=h*s->channels+c,index=(h*s->rows+row)*s->channels+c;
    s->row_lower[local]=s->lower[index];s->row_upper[local]=s->upper[index];
    s->row_candidate[local]=s->candidate[index];
   }
   float scale;size_t count;
   int decision=merlin_frontier_row(s->row_lower,s->row_upper,
    s->row_candidate,width,plan,s->pending,&scale,&count);
   if(decision<0)return 0;
   if(decision){s->certified[row]=1;continue;}
   remaining++;
   for(size_t i=0;i<width;i++)if(s->pending[i]&&
    !s->refine(s->context,i/s->channels,row,i%s->channels,pass!=0))return 0;
  }
  if(!remaining)return 1;
 }
 return 0;
}
#endif
