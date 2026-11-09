#ifndef MERLIN_REPRESENTATION_ERROR_NORMS_H
#define MERLIN_REPRESENTATION_ERROR_NORMS_H
#include "prepared_dot_norms.h"
/* Optional preparation for finite original/reconstructed real operands under
 * the existing stable-RNE, gradual-underflow, unobserved-flags contract.
 * Equality is a runtime proof of zero representation error, including signed
 * zero. It does not eliminate the independent source-FMA rounding bound.
 * No source/golden samples or type-wide representability assumptions enter. */
static inline void merlin_representation_error_add(merlin_dot_norms *norm,
 double original, double reconstructed) {
 if(!norm||!norm->valid)return;
 if(!MERLIN_SOURCE_ISFINITE(original)||!MERLIN_SOURCE_ISFINITE(reconstructed)){norm->valid=0;return;}
 if(original==reconstructed)return;
 merlin_dot_norms_add(norm,
     merlin_fma_next_up(MERLIN_SOURCE_F64_ABS(original-reconstructed)));
}
static inline void merlin_representation_error_finish(merlin_dot_norms *norm) {
 if(!norm||!norm->valid)return;
 /* The object must originate from begin/add above, remain unmodified, and
  * contain every element of its immutable row. Zero l1 then proves all errors
  * zero; the exact real norm is zero, independent of libm sqrt accuracy. */
 if(norm->l1==0){norm->l2=0;return;}
 merlin_dot_norms_finish(norm);
}
static inline double merlin_representation_error_product_upper(
 const merlin_admitted_dot_norms *error,const merlin_admitted_dot_norms *other) {
 /* Both arguments are successful immutable admissions. Unlike infinity*0,
  * a proved zero error vector has exactly zero product error. */
 if(error->l1==0)return 0;
 return merlin_dot_norms_admitted_product_upper(error,other);
}
#endif
