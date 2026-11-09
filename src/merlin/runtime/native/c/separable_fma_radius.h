#ifndef MERLIN_SEPARABLE_FMA_RADIUS_H
#define MERLIN_SEPARABLE_FMA_RADIUS_H
#include "prepared_fma_product_bounds.h"
/* Source-zero-seeded ordered binary32 FMA, stable RNE/gradual underflow.
 * Admitted immutable error norms prove exact representation, not approximate
 * equality. The producer's binary64 center must be the exact real dot of those
 * same operands. No arbitrary center/public scalar admission is provided.
 */
typedef struct {
 const merlin_admitted_dot_norms *source;
 size_t columns,length;
 double maximum;
 int valid;
} merlin_fma_exact_columns;
static inline merlin_fma_exact_columns merlin_fma_exact_columns_prepare(
 const merlin_fma_product_columns *producer,
 const merlin_admitted_dot_norms *source,
 const merlin_admitted_dot_norms *error) {
 merlin_fma_exact_columns p={0,0,0,0,0};
 if(!producer||!producer->valid||!source||!error)return p;
 double maximum=0;
 for(size_t j=0;j<producer->columns;j++){
  if(error[j].l1!=0||error[j].maximum!=0||error[j].l2!=0||
     !MERLIN_SOURCE_ISFINITE(source[j].maximum)||source[j].maximum<0)return p;
  maximum=MERLIN_SOURCE_F64_MAX(maximum,source[j].maximum);
 }
 if(!producer->columns||!producer->length)return p;
 return (merlin_fma_exact_columns){source,producer->columns,producer->length,maximum,1};
}
typedef struct {
 const merlin_fma_exact_columns *columns;
 const double *centers;
 double gamma_l1,subnormal;
 int valid;
} merlin_fma_separable_radius;
static inline merlin_fma_separable_radius merlin_fma_separable_radius_prepare(
 const merlin_fma_product_row *producer,
 const merlin_fma_exact_columns *columns,
 const merlin_admitted_dot_norms *source,
 const merlin_admitted_dot_norms *error,size_t uncertain_positions) {
 merlin_fma_separable_radius p={0,0,0,0,0};
 if(!producer||!producer->valid||!columns||!columns->valid||!source||!error||
    uncertain_positions||producer->length!=columns->length||
    producer->columns!=columns->columns||!producer->centers||
    error->l1!=0||error->maximum!=0||error->l2!=0||
    !MERLIN_SOURCE_ISFINITE(source->l1)||source->l1<0)return p;
 double factor=merlin_fma_up_mul(producer->gamma_upper,source->l1);
 double absolute=merlin_fma_up_mul(source->l1,columns->maximum);
 double radius=merlin_fma_up_add(merlin_fma_up_mul(factor,columns->maximum),producer->subnormal_error_upper);
 double envelope=merlin_fma_up_add(absolute,radius);
 if(!MERLIN_SOURCE_ISFINITE(factor)||!MERLIN_SOURCE_ISFINITE(envelope)||envelope>=(double)FLT_MAX)return p;
 return (merlin_fma_separable_radius){columns,producer->centers,factor,producer->subnormal_error_upper,1};
}
/* |source_dot-real_dot| <= gamma_k sum_i |a_i b_i| + eta_k.
 * sum_i |a_i b_ij| <= L1(a) max_i |b_ij|. Both upward products
 * cover the real product regardless of reassociation. The row envelope above
 * also bounds every source prefix and final enclosure below FLT_MAX.
 * Only source-certified private exact products reach this internal consumer.
 */
static inline int merlin_fma_separable_radius_apply(
 const merlin_fma_separable_radius *p,size_t column,float *lower,float *upper) {
 if(!p||!p->valid||column>=p->columns->columns)return 0;
 double radius=merlin_fma_up_add(merlin_fma_up_mul(p->gamma_l1,
   p->columns->source[column].maximum),p->subnormal);
 double lo=merlin_fma_down_add(p->centers[column],-radius);
 double hi=merlin_fma_up_add(p->centers[column],radius);
#if defined(MERLIN_ENABLE_EXACT_BOUND_CONVERSION)
 *lower=MERLIN_F32_EXACT_FLOOR_FROM_F64(lo);
 *upper=MERLIN_F32_EXACT_CEIL_FROM_F64(hi);
#else
 *lower=(float)lo;*upper=(float)hi;
 if((double)*lower>lo)*lower=merlin_fma_next_down_f32(*lower);
 if((double)*upper<hi)*upper=merlin_fma_next_up_f32(*upper);
#endif
 return 1;
}
/* Explicit independent-coordinate schedule. The compiler caller must bind the
 * same immutable admitted row epoch, stable RNE/nontrapping arithmetic and no
 * intermediate exception observations. Outputs are fresh disjoint spans and
 * cannot alias the plan, column metadata or centers. This helper grants no
 * pointer-identity cache or relaxed numerical admission. Scalar caller tails
 * use the original consumer. Unused helper leaves default generated code inert.
 */
static inline int merlin_fma_radius_disjoint(
 const void *a,size_t an,const void *b,size_t bn) {
 uintptr_t x=(uintptr_t)a,y=(uintptr_t)b;
 return a&&b&&an<=UINTPTR_MAX-x&&bn<=UINTPTR_MAX-y&&
        (x+an<=y||y+bn<=x);
}
static inline int merlin_fma_separable_radius_apply_eight(
 const merlin_fma_separable_radius *p,size_t first,float *lower,float *upper) {
 if(!p||!p->valid||!p->columns||!lower||!upper||
    first>p->columns->columns||p->columns->columns-first<8)return 0;
 size_t columns=p->columns->columns;
 if(columns>SIZE_MAX/sizeof(*p->columns->source)||
    columns>SIZE_MAX/sizeof(*p->centers)||
    !merlin_fma_radius_disjoint(lower,8*sizeof(*lower),upper,8*sizeof(*upper)))return 0;
 const void *reads[4]={p,p->columns,p->columns->source,p->centers};
 size_t sizes[4]={sizeof(*p),sizeof(*p->columns),
                 columns*sizeof(*p->columns->source),columns*sizeof(*p->centers)};
 for(size_t i=0;i<4;i++)if(
    !merlin_fma_radius_disjoint(lower,8*sizeof(*lower),reads[i],sizes[i])||
    !merlin_fma_radius_disjoint(upper,8*sizeof(*upper),reads[i],sizes[i]))return 0;
 double maximum[8],center[8],product[8],radius[8],lo[8],hi[8];
 for(size_t i=0;i<8;i++){
  maximum[i]=p->columns->source[first+i].maximum;
  center[i]=p->centers[first+i];
 }
 for(size_t i=0;i<8;i++)product[i]=merlin_fma_up_mul(p->gamma_l1,maximum[i]);
 for(size_t i=0;i<8;i++)radius[i]=merlin_fma_up_add(product[i],p->subnormal);
 for(size_t i=0;i<8;i++)lo[i]=merlin_fma_down_add(center[i],-radius[i]);
 for(size_t i=0;i<8;i++)hi[i]=merlin_fma_up_add(center[i],radius[i]);
#if defined(MERLIN_ENABLE_EXACT_BOUND_CONVERSION)
 for(size_t i=0;i<8;i++)lower[i]=MERLIN_F32_EXACT_FLOOR_FROM_F64(lo[i]);
 for(size_t i=0;i<8;i++)upper[i]=MERLIN_F32_EXACT_CEIL_FROM_F64(hi[i]);
#else
 for(size_t i=0;i<8;i++){
  lower[i]=(float)lo[i];
  if((double)lower[i]>lo[i])lower[i]=merlin_fma_next_down_f32(lower[i]);
 }
 for(size_t i=0;i<8;i++){
  upper[i]=(float)hi[i];
  if((double)upper[i]<hi[i])upper[i]=merlin_fma_next_up_f32(upper[i]);
 }
#endif
 return 1;
}
#endif
