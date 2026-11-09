#ifndef MERLIN_PREPARED_DOT_NORMS_H
#define MERLIN_PREPARED_DOT_NORMS_H
#include "ordered_fma_bounds.h"
/* Prepared conservative row norms. Stable RNE, gradual underflow and
 * nontrapping/unobserved flags are the existing bound environment contract.
 * Each magnitude must itself enclose the absolute source value. */
typedef struct { double l1,maximum,squares,l2; int valid; } merlin_dot_norms;
static inline merlin_dot_norms merlin_dot_norms_begin(const merlin_fma_bound *environment){
 return (merlin_dot_norms){0,0,0,INFINITY,environment&&environment->valid};
}
static inline void merlin_dot_norms_add(merlin_dot_norms *p,double magnitude){
 if(!p||!p->valid)return;
 if(!MERLIN_SOURCE_ISFINITE(magnitude)||magnitude<0){p->valid=0;return;}
 p->l1=merlin_fma_up_add(p->l1,magnitude);
 p->maximum=MERLIN_SOURCE_F64_MAX(p->maximum,magnitude);
 p->squares=merlin_fma_up_add(p->squares,merlin_fma_up_mul(magnitude,magnitude));
 if(!MERLIN_SOURCE_ISFINITE(p->l1))p->valid=0;
}
static inline void merlin_dot_norms_finish(merlin_dot_norms *p){
 if(!p||!p->valid)return;
 p->l2=INFINITY;
 if(!MERLIN_SOURCE_ISFINITE(p->squares)||p->squares<0)return;
 double candidate=merlin_fma_next_up(sqrt(p->squares));
 /* Verify the square-root upper enclosure independently of libm accuracy.
  * Failed/overflowing L2 preparation retains the valid L1/Linf bound. */
 double square_lower=merlin_fma_next_down(candidate*candidate);
 if(MERLIN_SOURCE_ISFINITE(candidate)&&candidate>=0&&square_lower>=p->squares)p->l2=candidate;
}
static inline double merlin_dot_norms_product_upper(const merlin_dot_norms *a,const merlin_dot_norms *b){
 if(!a||!b||!a->valid||!b->valid)return INFINITY;
 double holder=merlin_fma_up_mul(a->l1,b->maximum);
 double cauchy=merlin_fma_up_mul(a->l2,b->l2);
 return MERLIN_SOURCE_F64_MIN(holder,cauchy);
}
/* Explicit successful preparation is the admission for immutable read-only
 * norm products. This representation omits accumulation state and revalidating
 * the same metadata in every output cell. A failed preparation must select the
 * caller's source fallback; it never publishes a usable row object. */
typedef struct { double l1, maximum, l2; } merlin_admitted_dot_norms;
static inline int merlin_dot_norms_admit(const merlin_dot_norms *source,
 merlin_admitted_dot_norms *destination) {
 if(!source||!source->valid||!MERLIN_SOURCE_ISFINITE(source->l1)||source->l1<0||
    !MERLIN_SOURCE_ISFINITE(source->maximum)||source->maximum<0||
    isnan(source->l2)||source->l2<0)return 0;
 *destination=(merlin_admitted_dot_norms){source->l1,source->maximum,source->l2};
 return 1;
}
static inline double merlin_dot_norms_admitted_product_upper(
 const merlin_admitted_dot_norms *a,const merlin_admitted_dot_norms *b) {
 return MERLIN_SOURCE_F64_MIN(merlin_fma_up_mul(a->l1,b->maximum),
             merlin_fma_up_mul(a->l2,b->l2));
}
#endif
