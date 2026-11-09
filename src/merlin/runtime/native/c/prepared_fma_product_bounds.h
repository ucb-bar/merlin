#ifndef MERLIN_PREPARED_FMA_PRODUCT_BOUNDS_H
#define MERLIN_PREPARED_FMA_PRODUCT_BOUNDS_H
#include "representation_error_norms.h"
#include "dot_norm_requirements.h"

/* Optional domain admission for exact reconstructed matrix products. This is
 * not an alternative certificate: caller must prove each center is the exact
 * real dot of the immutable reconstructed rows used by these norms. The actual
 * signed-radix producer supplies that proof; an arbitrary center array does not.
 * All spans remain private/disjoint and unchanged through synchronous use.
 * Stable RNE, gradual underflow and unobserved nontrapping flags are required.
 * Outward capabilities must be monotone for finite nonnegative operands and
 * satisfy down_add(-x,-y) == -up_add(x,y) in value. Default adjacency and the
 * independently qualified directed-rounding implementations meet this rule.
 * No lifetime/alias proof is inferred from addresses or successful arithmetic.
 */
typedef struct {
 size_t columns, length;
 double source_maximum, error_maximum, reconstructed_maximum;
 int valid;
} merlin_fma_product_columns;

static inline merlin_fma_product_columns merlin_fma_product_columns_prepare(
 const merlin_admitted_dot_norms *source,
 const merlin_admitted_dot_norms *error, const double *reconstructed,
 size_t columns,size_t length) {
 merlin_fma_product_columns p={columns,length,0,0,0,0};
 if(!source||!error||!reconstructed||!columns||!length||columns>SIZE_MAX/length)return p;
 for(size_t j=0;j<columns;j++){
  if(!MERLIN_SOURCE_ISFINITE(source[j].maximum)||source[j].maximum<0||
     !MERLIN_SOURCE_ISFINITE(error[j].maximum)||error[j].maximum<0)return p;
  p.source_maximum=MERLIN_SOURCE_F64_MAX(p.source_maximum,source[j].maximum);
  p.error_maximum=MERLIN_SOURCE_F64_MAX(p.error_maximum,error[j].maximum);
  for(size_t z=0;z<length;z++){
   double value=reconstructed[j*length+z];
   if(!MERLIN_SOURCE_ISFINITE(value))return p;
   p.reconstructed_maximum=MERLIN_SOURCE_F64_MAX(p.reconstructed_maximum,MERLIN_SOURCE_F64_ABS(value));
  }
 }
 p.valid=1;return p;
}

typedef struct {
 const double *centers;
 size_t columns,length;
 double gamma_upper,subnormal_error_upper;
 int valid;
} merlin_fma_product_row;

/* Admitted source/reconstruction/error L1 norms cover every row element.
 * Uncertainty values enclose the same distinct source positions used by the
 * checked bound loop. Their count/order is retained in the error envelope.
 * Every column norm uses the same original source and reconstructed matrix.
 * Under these producer facts, monotonicity of each upward operation proves
 * all dynamic center/absolute/error values finite, nonnegative where required,
 * and both source prefixes and final enclosure strictly inside FLT_MAX.
 */
static inline merlin_fma_product_row merlin_fma_product_row_prepare_l1(
 const merlin_fma_zero_gamma_batch *gamma,
 const merlin_fma_product_columns *columns,
 const merlin_admitted_dot_norms *source,
 const merlin_l1_norm *reconstructed,
 const merlin_admitted_dot_norms *error,
 const double *uncertainty,size_t used,const double *centers) {
 merlin_fma_product_row p={0,0,0,0,0,0};
 if(!gamma||!gamma->valid||!columns||!columns->valid||!source||!reconstructed||!error||!centers||
    gamma->length!=columns->length||used>columns->length||(used&&!uncertainty)||
    !MERLIN_SOURCE_ISFINITE(gamma->gamma_upper)||gamma->gamma_upper<=0||
    !MERLIN_SOURCE_ISFINITE(gamma->subnormal_error_upper)||gamma->subnormal_error_upper<=0||
    !MERLIN_SOURCE_ISFINITE(source->l1)||source->l1<0||
    !reconstructed->valid||!MERLIN_SOURCE_ISFINITE(reconstructed->l1)||reconstructed->l1<0||
    !MERLIN_SOURCE_ISFINITE(error->l1)||error->l1<0)return p;
 double center=merlin_fma_up_mul(reconstructed->l1,columns->reconstructed_maximum);
 double absolute=merlin_fma_up_mul(source->l1,columns->source_maximum);
 double repr=merlin_fma_up_add(merlin_fma_up_mul(error->l1,columns->source_maximum),
                             merlin_fma_up_mul(reconstructed->l1,columns->error_maximum));
 for(size_t u=0;u<used;u++){
  if(!MERLIN_SOURCE_ISFINITE(uncertainty[u])||uncertainty[u]<0)return p;
  repr=merlin_fma_up_add(repr,merlin_fma_up_mul(uncertainty[u],columns->source_maximum));
 }
 double magnitude=merlin_fma_up_add(merlin_fma_up_mul(.5,merlin_fma_up_add(center,absolute)),repr);
 double rounding=merlin_fma_up_add(merlin_fma_up_mul(gamma->gamma_upper,magnitude),gamma->subnormal_error_upper);
 double prefix=merlin_fma_up_add(magnitude,rounding);
 double finish=merlin_fma_up_add(center,merlin_fma_up_add(repr,rounding));
 if(!MERLIN_SOURCE_ISFINITE(prefix)||prefix>=(double)FLT_MAX||
    !MERLIN_SOURCE_ISFINITE(finish)||finish>=(double)FLT_MAX)return p;
 return (merlin_fma_product_row){centers,columns->columns,columns->length,
   gamma->gamma_upper,gamma->subnormal_error_upper,1};
}

static inline merlin_fma_product_row merlin_fma_product_row_prepare(
 const merlin_fma_zero_gamma_batch *gamma,
 const merlin_fma_product_columns *columns,
 const merlin_admitted_dot_norms *source,
 const merlin_admitted_dot_norms *reconstructed,
 const merlin_admitted_dot_norms *error,
 const double *uncertainty,size_t used,const double *centers) {
 if(!reconstructed)return (merlin_fma_product_row){0,0,0,0,0,0};
 merlin_l1_norm l1={reconstructed->l1,1};
 return merlin_fma_product_row_prepare_l1(gamma,columns,source,&l1,error,uncertainty,used,centers);
}

/* Internal admitted-row consumer. absolute/repr must be produced by the same
 * unchanged checked norm/error formulas and positions, never public inputs.
 * Caller selects this path only for a successful row admission. The comparison
 * between reconstructed center and original absolute bound remains necessary.
 */
static inline int merlin_fma_product_row_apply(
 const merlin_fma_product_row *row,size_t column,double absolute,double repr,
 float *lower,float *upper) {
 if(!row||!row->valid||column>=row->columns)return 0;
 double center=row->centers[column];
 if(center>absolute||center < -absolute)return 0;
 double magnitude=merlin_fma_up_add(merlin_fma_up_mul(.5,
   merlin_fma_up_add(MERLIN_SOURCE_F64_ABS(center),absolute)),repr);
 double rounding=merlin_fma_up_add(merlin_fma_up_mul(row->gamma_upper,magnitude),row->subnormal_error_upper);
 double radius=merlin_fma_up_add(repr,rounding);
 double lo=merlin_fma_down_add(center,-radius),hi=merlin_fma_up_add(center,radius);
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
#endif
