#ifndef MERLIN_SOURCE_RMS_POINT_PRODUCTS_H
#define MERLIN_SOURCE_RMS_POINT_PRODUCTS_H
#include "source_rms_roundoff_estimate.h"
#include "encoded_row_equality.h"
/* Explicit APPROXIMATE replay selection, never a rigorous interval producer.
 * The private canonical product producer proves center[r,n+j] is the exact
 * real dot of the same reconstructed rows. Encoder witnesses prove finite
 * source-point/reconstruction equality and immutable, disjoint current spans.
 * All owners, rounding mode and unobserved nontrapping flags remain stable
 * throughout this synchronous call. No witness is created from an estimate.
 * Unknown/nonpoint/representation-error inputs return 0 for checked fallback.
 * column_max is distinct private scratch with capacity >= columns. Every
 * estimate output is freshly written; on refusal callers discard ALL outputs.
 */
static inline int merlin_source_rms4_point_product_estimates(
 const merlin_fma_bound *environment,
 const merlin_encoded_row_equality *a,const merlin_encoded_row_equality *b,
 const float *source_a,const float *lower_a,const float *upper_a,
 const float *source_b,const double *reconstructed_a,const double *reconstructed_b,
 const double *centers,float *estimated_low,float *estimated_high,
 size_t rows,size_t columns,size_t length,double *column_max,size_t capacity){
 if(!environment||!environment->valid||!a||!b||!source_a||!source_b||
    !reconstructed_a||!reconstructed_b||!centers||!estimated_low||!estimated_high||
    !column_max||!rows||!columns||!length||capacity<columns||
    rows>SIZE_MAX/length||columns>SIZE_MAX/length||rows>SIZE_MAX/columns)return 0;
 merlin_fma_zero_gamma_plan gamma=merlin_fma_zero_gamma_prepare(environment,length);
 merlin_source_rms4_plan policy=merlin_source_rms4_prepare(environment,length,gamma.subnormal_error_upper);
 if(!gamma.valid||!policy.valid)return 0;
 for(size_t r=0;r<rows;r++)if(!merlin_encoded_row_matches(a,r,
     source_a+r*length,reconstructed_a+r*length,
     lower_a?lower_a+r*length:0,upper_a?upper_a+r*length:0,length))return 0;
 for(size_t j=0;j<columns;j++)if(!merlin_encoded_row_matches(b,j,
     source_b+j*length,reconstructed_b+j*length,0,0,length))return 0;
 double global_max=0;
 for(size_t j=0;j<columns;j++){
  double maximum=0;
  for(size_t z=0;z<length;z++){
   double v=MERLIN_SOURCE_F64_ABS((double)source_b[j*length+z]);
   if(v>maximum)maximum=v;
  }
  column_max[j]=maximum;if(maximum>global_max)global_max=maximum;
 }
 for(size_t r=0;r<rows;r++){
  double l1=0;
  for(size_t z=0;z<length;z++)l1=merlin_fma_up_add(l1,
    MERLIN_SOURCE_F64_ABS((double)source_a[r*length+z]));
  /* Retain deterministic source-prefix overflow admission. This safety test
   * is NOT replaced by the RMS model. Every exact/source prefix magnitude is
   * bounded by absolute + gamma*absolute + eta, strictly inside FLT_MAX. */
  double absolute=merlin_fma_up_mul(l1,global_max);
  double rigorous=merlin_fma_up_add(merlin_fma_up_mul(gamma.gamma_upper,absolute),gamma.subnormal_error_upper);
  double prefix=merlin_fma_up_add(absolute,rigorous);
  if(!MERLIN_SOURCE_ISFINITE(prefix)||prefix>=(double)FLT_MAX)return 0;
  double factor=merlin_fma_up_mul(merlin_fma_up_mul(gamma.gamma_upper,l1),policy.ratio);
  for(size_t j=0;j<columns;j++){
   size_t t=r*columns+j;double center=centers[t];
   if(!MERLIN_SOURCE_ISFINITE(center)||center>absolute||center < -absolute)return 0;
   double radius=merlin_fma_up_add(merlin_fma_up_mul(factor,column_max[j]),policy.eta);
   double lo=merlin_fma_down_add(center,-radius),hi=merlin_fma_up_add(center,radius);
#if defined(MERLIN_ENABLE_EXACT_BOUND_CONVERSION)
   estimated_low[t]=MERLIN_F32_EXACT_FLOOR_FROM_F64(lo);
   estimated_high[t]=MERLIN_F32_EXACT_CEIL_FROM_F64(hi);
#else
   float l=(float)lo,h=(float)hi;
   if((double)l>lo)l=merlin_fma_next_down_f32(l);
   if((double)h<hi)h=merlin_fma_next_up_f32(h);
   estimated_low[t]=l;estimated_high[t]=h;
#endif
  }
 }
 return 1;
}
#endif
