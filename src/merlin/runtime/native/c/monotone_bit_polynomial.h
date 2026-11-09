#ifndef MERLIN_MONOTONE_BIT_POLYNOMIAL_H
#define MERLIN_MONOTONE_BIT_POLYNOMIAL_H
#include "source_f32_math.h"
#include "prepared_bit_polynomial.h"
#include "source_f32_pair.h"
#include <float.h>

/* An explicitly selected equivalent source-value floor capability must satisfy
 * the same IEEE encoding and unobserved, nontrapping exception contract. */
#ifndef MERLIN_MONOTONE_F32_FLOOR
#define MERLIN_MONOTONE_F32_FLOOR(value) MERLIN_SOURCE_F32_FLOOR(value)
#endif

/* Optional exact source enclosure, never a polynomial approximation. The
 * caller supplies the actual source plan/domain and existing IEEE/RNE token,
 * keeps rounding stable, and declares flags unobserved/nontrapping. Immutable
 * prepared data copies the source plan. Unsupported proofs use checked bounds.
 * For real E(s)=s-P(frac(s)), P'<1 and P(1)>=P(0) prove global monotonicity.
 * Local power-of-two budgets enclose original binary32 fraction, three Horner
 * FMAs and subtraction, including gradual underflow. Source endpoint q values
 * plus twice that budget enclose every interior q. The original final FMA and
 * positive integer encoding are monotone and retain their own rounding.
 */
typedef struct {
  merlin_prepared_bit_polynomial checked;
  float upper;
  uint32_t word_budget;
  double rounding_error;
  double source_error[26];
  int fast_valid;
} merlin_monotone_bit_polynomial;

typedef struct { double lo, hi; } merlin_polynomial_real_interval;

static inline merlin_polynomial_real_interval merlin_polynomial_real_product(
    merlin_polynomial_real_interval a, merlin_polynomial_real_interval b) {
  double v[4]={a.lo*b.lo,a.lo*b.hi,a.hi*b.lo,a.hi*b.hi};
  double lo=v[0],hi=v[0];
  for(int i=1;i<4;i++){lo=MERLIN_SOURCE_F64_MIN(lo,v[i]);hi=MERLIN_SOURCE_F64_MAX(hi,v[i]);}
  return (merlin_polynomial_real_interval){merlin_fma_next_down(lo),merlin_fma_next_up(hi)};
}

/* Preparation only: outward binary64 operations enclose exact real E(s).
 * Source s is binary32 and its floor is exactly represented in this domain. */
static inline merlin_polynomial_real_interval merlin_polynomial_real_e(
    float s, const merlin_bit_polynomial_plan *p) {
  double f=(double)s-floor((double)s);
  merlin_polynomial_real_interval fraction={MERLIN_SOURCE_F64_MAX(0,merlin_fma_next_down(f)),MERLIN_SOURCE_F64_MIN(1,merlin_fma_next_up(f))};
  merlin_polynomial_real_interval poly={p->coefficients[0],p->coefficients[0]};
  for(int i=1;i<4;i++){
    poly=merlin_polynomial_real_product(fraction,poly);
    poly.lo=merlin_fma_next_down(poly.lo+(double)p->coefficients[i]);
    poly.hi=merlin_fma_next_up(poly.hi+(double)p->coefficients[i]);
  }
  return (merlin_polynomial_real_interval){merlin_fma_next_down((double)s-poly.hi),merlin_fma_next_up((double)s-poly.lo)};
}

static inline double merlin_polynomial_f32_rounding_error(double magnitude) {
  int exponent;
  if(!MERLIN_SOURCE_ISFINITE(magnitude)||magnitude<0)return INFINITY;
  if(magnitude==0)return 0x1p-150;
  (void)frexp(magnitude,&exponent);
  return MERLIN_SOURCE_F64_MAX(ldexp(1.0,exponent-25),0x1p-150);
}

static inline merlin_monotone_bit_polynomial merlin_monotone_bit_polynomial_prepare(
    const merlin_fma_bound *environment, const merlin_bit_polynomial_plan *p,
    float upper) {
  merlin_monotone_bit_polynomial out={0};
  if(!environment||!environment->valid)return out;
  out.checked=merlin_bit_polynomial_prepare(p);out.upper=upper;
  if(!out.checked.valid||!MERLIN_SOURCE_ISFINITE(upper)||p->cutoff>upper||!(p->bit_multiplier>0))return out;
  /* These products are exact in binary64 for binary32 coefficients. */
  double d0=3.0*(double)p->coefficients[0],d1=2.0*(double)p->coefficients[1],d2=p->coefficients[2];
  double derivative_hi=merlin_fma_next_up(merlin_fma_next_up(d2+MERLIN_SOURCE_F64_MAX(0,d1))+MERLIN_SOURCE_F64_MAX(0,d0));
  double derivative_lo=merlin_fma_next_down(merlin_fma_next_down(d2+MERLIN_SOURCE_F64_MIN(0,d1))+MERLIN_SOURCE_F64_MIN(0,d0));
  double jump=merlin_fma_next_down(merlin_fma_next_down((double)p->coefficients[0]+p->coefficients[1])+p->coefficients[2]);
  if(!(derivative_hi<1)||!(jump>=0))return out;
  float first=p->cutoff*p->scale,last=upper*p->scale;
  if(!MERLIN_SOURCE_ISFINITE(first)||!MERLIN_SOURCE_ISFINITE(last)||floor((double)first)<-16777216||floor((double)last)>16777215)return out;
  double source_mag=MERLIN_SOURCE_F64_ABS((double)p->coefficients[0]),horner_error=0;
  for(int i=1;i<4;i++){
    double argument_mag=merlin_fma_up_add(source_mag,MERLIN_SOURCE_F64_ABS((double)p->coefficients[i]));
    double error=merlin_polynomial_f32_rounding_error(argument_mag);
    source_mag=merlin_fma_up_add(argument_mag,error);
    horner_error=merlin_fma_up_add(horner_error,error);
    if(!(source_mag<=FLT_MAX))return out;
  }
  double derivative_mag=MERLIN_SOURCE_F64_MAX(MERLIN_SOURCE_F64_ABS(derivative_lo),MERLIN_SOURCE_F64_ABS(derivative_hi));
  double polynomial_error=merlin_fma_up_add(horner_error,merlin_fma_up_mul(derivative_mag,0x1p-24));
  double subtract_mag=merlin_fma_up_add(MERLIN_SOURCE_F64_MAX(MERLIN_SOURCE_F64_ABS((double)first),MERLIN_SOURCE_F64_ABS((double)last)),source_mag);
  double subtract_error=merlin_polynomial_f32_rounding_error(subtract_mag);
  if(!(merlin_fma_up_add(subtract_mag,subtract_error)<=FLT_MAX))return out;
  double encoded_mag=merlin_fma_up_add(merlin_fma_up_mul(p->bit_multiplier,merlin_fma_up_add(subtract_mag,subtract_error)),MERLIN_SOURCE_F64_ABS((double)p->bit_bias));
  double encoded_error=merlin_polynomial_f32_rounding_error(encoded_mag);
  if(!(merlin_fma_up_add(encoded_mag,encoded_error)<=FLT_MAX))return out;
  double total_error=merlin_fma_up_add(merlin_fma_up_mul(p->bit_multiplier,merlin_fma_up_add(polynomial_error,subtract_error)),encoded_error);
  merlin_polynomial_real_interval low=merlin_polynomial_real_e(first,p),high=merlin_polynomial_real_e(last,p);
  double word_lo=merlin_fma_next_down(merlin_fma_next_down((double)p->bit_multiplier*low.lo)+p->bit_bias);
  double word_hi=merlin_fma_next_up(merlin_fma_next_up((double)p->bit_multiplier*high.hi)+p->bit_bias);
  if(!(merlin_fma_next_down(word_lo-total_error)>=0)||!(merlin_fma_next_up(word_hi+total_error)<2139095040.0))return out;
  double budget=ceil(merlin_fma_up_mul(2,total_error))+2;
  if(!MERLIN_SOURCE_ISFINITE(budget)||budget>UINT32_MAX)return out;
  out.rounding_error=total_error;out.word_budget=(uint32_t)budget;out.fast_valid=1;
  for(int i=0;i<26;i++)out.source_error[i]=merlin_fma_up_add(polynomial_error,
      merlin_polynomial_f32_rounding_error(merlin_fma_up_add(ldexp(1.0,i),source_mag)));
  return out;
}

static inline uint32_t merlin_monotone_polynomial_source_word(
    float x,const merlin_bit_polynomial_plan *p) {
  float s=x*p->scale,f=s-MERLIN_MONOTONE_F32_FLOOR(s),poly=p->coefficients[0];
  for(int i=1;i<4;i++)poly=MERLIN_SOURCE_F32_FMA(f,poly,p->coefficients[i]);
  return (uint32_t)(int32_t)MERLIN_SOURCE_F32_FMA(p->bit_multiplier,s-poly,p->bit_bias);
}

static inline float merlin_monotone_polynomial_source_q(
    float s,const merlin_bit_polynomial_plan *p) {
  float f=s-MERLIN_MONOTONE_F32_FLOOR(s),poly=p->coefficients[0];
  for(int i=1;i<4;i++)poly=MERLIN_SOURCE_F32_FMA(f,poly,p->coefficients[i]);
  return s-poly;
}

/* Explicit source-order schedule for two independent admitted endpoints.
 * Each chain retains its own floor/subtract/three rounded FMAs/subtract.
 * The source environment is stable, nontrapping, with flags unobserved.
 * This is a scheduling choice, not reassociation or a new enclosure. */
typedef struct { float low, high; } merlin_polynomial_source_pair;
static inline merlin_polynomial_source_pair merlin_monotone_polynomial_source_pair(
    float low,float high,const merlin_bit_polynomial_plan *p) {
  float lf=low-MERLIN_MONOTONE_F32_FLOOR(low);
  float hf=high-MERLIN_MONOTONE_F32_FLOOR(high);
  float lp=p->coefficients[0],hp=p->coefficients[0];
  for(int i=1;i<4;i++){
    MERLIN_SOURCE_F32_FMA_PAIR(lf,lp,p->coefficients[i],
                             hf,hp,p->coefficients[i],&lp,&hp);
  }
  return (merlin_polynomial_source_pair){low-lp,high-hp};
}

/* Explicit alternative enclosure in positive IEEE word space. Preparation
 * proves monotonic real E(s) and a uniform encoded rounding error. Each source
 * endpoint differs from real E(s)'s encoded value by at most that error plus
 * integer truncation. Twice the error plus two words therefore encloses every
 * interior source word, even across floor boundaries. This can be wider than
 * the default local q enclosure; callers must price any extra exact replay.
 * It requires the same immutable prepared value and stable RNE environment.
 */
static inline merlin_f32_interval merlin_monotone_bit_polynomial_apply_words(
    merlin_f32_interval x, const merlin_monotone_bit_polynomial *p) {
  if (!p) return merlin_interval_bad();
  if (!p->fast_valid || !x.valid || x.hi > p->upper)
    return merlin_prepared_bit_polynomial_apply(x, &p->checked);
  const merlin_bit_polynomial_plan *source = &p->checked.source;
  if (x.hi < source->cutoff) return merlin_interval_point(0);
  int includes_zero = x.lo < source->cutoff;
  float left = MERLIN_SOURCE_F32_MAX(x.lo, source->cutoff);
  uint32_t first = merlin_monotone_polynomial_source_word(left, source);
  if (left == x.hi && !includes_zero)
    return merlin_interval_point(merlin_interval_float(first));
  uint32_t last = merlin_monotone_polynomial_source_word(x.hi, source);
  uint32_t low = includes_zero || first < p->word_budget
      ? 0 : first - p->word_budget;
  uint64_t high = (uint64_t)last + p->word_budget;
  if (high > UINT32_C(2139095039)) high = UINT32_C(2139095039);
  return merlin_interval(merlin_interval_float(low),
                         merlin_interval_float((uint32_t)high));
}

static inline merlin_f32_interval merlin_monotone_bit_polynomial_apply(
    merlin_f32_interval x,const merlin_monotone_bit_polynomial *p) {
  if(!p)return merlin_interval_bad();
  if(!p->fast_valid||!x.valid||x.hi>p->upper)return merlin_prepared_bit_polynomial_apply(x,&p->checked);
  const merlin_bit_polynomial_plan *source=&p->checked.source;
  if(x.hi<source->cutoff)return merlin_interval_point(0);
  int includes_zero=x.lo<source->cutoff;
  float left=MERLIN_SOURCE_F32_MAX(x.lo,source->cutoff);
  if(left==x.hi&&!includes_zero)return merlin_interval_point(
      merlin_interval_float(merlin_monotone_polynomial_source_word(left,source)));
  float first=left*source->scale,last=x.hi*source->scale;
  float magnitude=MERLIN_SOURCE_F32_MAX(MERLIN_SOURCE_F32_ABS(first),MERLIN_SOURCE_F32_ABS(last));
  int index=magnitude<=1?0:(int)((merlin_interval_bits(magnitude)>>23)&255)-126;
  if(index<0||index>=26)return merlin_prepared_bit_polynomial_apply(x,&p->checked);
  double error=merlin_fma_up_mul(2,p->source_error[index]);
  /* Enclose source q directly, retaining the final source FMA's monotonic
   * rounding instead of paying its error twice in integer-word space. */
#ifdef MERLIN_MONOTONE_SOURCE_Q_PAIR
  merlin_polynomial_source_pair endpoints=merlin_monotone_polynomial_source_pair(first,last,source);
  double qlo=merlin_fma_down_add((double)endpoints.low,-error);
  double qhi=merlin_fma_up_add((double)endpoints.high,error);
#else
  double qlo=merlin_fma_down_add((double)merlin_monotone_polynomial_source_q(first,source),-error);
  double qhi=merlin_fma_up_add((double)merlin_monotone_polynomial_source_q(last,source),error);
#endif
  float low=merlin_fma_down_cast_f32(qlo),high=merlin_fma_up_cast_f32(qhi);
  if(!MERLIN_SOURCE_ISFINITE(low)||!MERLIN_SOURCE_ISFINITE(high))return merlin_prepared_bit_polynomial_apply(x,&p->checked);
  float tlo=MERLIN_SOURCE_F32_FMA(source->bit_multiplier,low,source->bit_bias);
  float thi=MERLIN_SOURCE_F32_FMA(source->bit_multiplier,high,source->bit_bias);
  uint32_t lo=includes_zero||tlo<=0?0:(uint32_t)(int32_t)tlo;
  uint32_t hi=thi>=2139095040.0f?UINT32_C(2139095039):(uint32_t)(int32_t)thi;
  return merlin_interval(merlin_interval_float(lo),merlin_interval_float(hi));
}
#endif
