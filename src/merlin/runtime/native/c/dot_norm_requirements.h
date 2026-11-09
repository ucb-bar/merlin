#ifndef MERLIN_DOT_NORM_REQUIREMENTS_H
#define MERLIN_DOT_NORM_REQUIREMENTS_H
#include "prepared_dot_norms.h"
/* An immutable admitted error vector with L1 exactly zero contributes no
 * representation-product term. If every RHS error vector has this property,
 * reconstructed LHS norms need only L1/Linf, including producer-domain checks.
 * The caller retains the complete zero-error branch at every consuming output.
 * Unknown/invalid metadata keeps the full norm path. Skipped floating exception
 * flags are unobserved under the existing nontrapping norm contract.
 */
typedef struct { int require_l2; } merlin_dot_norm_requirements;
static inline merlin_dot_norm_requirements merlin_reconstruction_norm_requirements(
 const merlin_admitted_dot_norms *errors,size_t columns) {
 merlin_dot_norm_requirements p={1};
 if(!errors||!columns)return p;
 for(size_t j=0;j<columns;j++)
  if(errors[j].l1!=0 || errors[j].maximum!=0 || errors[j].l2!=0)return p;
 p.require_l2=0;return p;
}
/* A distinct L1-only object cannot be passed to any L2/min consumer. */
typedef struct { double l1; int valid; } merlin_l1_norm;
static inline merlin_l1_norm merlin_l1_norm_begin(const merlin_fma_bound *env) {
 return (merlin_l1_norm){0,env&&env->valid};
}
static inline void merlin_l1_norm_add(merlin_l1_norm *p,double magnitude) {
 if(!p||!p->valid)return;
 if(!MERLIN_SOURCE_ISFINITE(magnitude)||magnitude<0){p->valid=0;return;}
 p->l1=merlin_fma_up_add(p->l1,magnitude);
 if(!MERLIN_SOURCE_ISFINITE(p->l1))p->valid=0;
}
#endif
