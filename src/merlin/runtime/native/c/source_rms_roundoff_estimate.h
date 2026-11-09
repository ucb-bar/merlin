#ifndef MERLIN_SOURCE_RMS_ROUNDOFF_ESTIMATE_H
#define MERLIN_SOURCE_RMS_ROUNDOFF_ESTIMATE_H
#include "ordered_fma_bounds.h"
/* EXPLICIT APPROXIMATE POLICY, NEVER A PROVED SOURCE ENCLOSURE.
 * For independent, zero-mean RNE errors with uniform variance u^2/3,
 * and a common admitted absolute prefix envelope T, the modeled standard
 * deviation is u*T*sqrt(k/3). This fixed four-sigma experiment scales the
 * worst-case k*u*T radius by min(1,4/sqrt(3*k)); deterministic eta is retained.
 * Independence/uniformity need not hold for real source inputs. There is no
 * claimed probability guarantee, deterministic bound or whole-output budget.
 * Only an explicitly authorized approximate route may use this type. Source
 * representation and input-interval errors MUST NOT be scaled by this policy.
 * Original source overflow admission, masks, nonlinear DAG, checked fallback
 * and the independently declared whole-output gate remain obligations.
 */
typedef struct {size_t length;double ratio,eta;int valid;} merlin_source_rms4_plan;
typedef struct {float low,high;int available;} merlin_source_rms4_estimate;
static inline merlin_source_rms4_plan merlin_source_rms4_prepare(
 const merlin_fma_bound *env,size_t length,double deterministic_eta){
 merlin_source_rms4_plan p={0};
 if(!env||!env->valid||!length||length>=((size_t)1<<24)||
    !MERLIN_SOURCE_ISFINITE(deterministic_eta)||deterministic_eta<0)return p;
 double ratio=4.0/sqrt(3.0*(double)length);if(ratio>1.0)ratio=1.0;
 return (merlin_source_rms4_plan){length,ratio,deterministic_eta,1};
}
static inline merlin_source_rms4_estimate merlin_source_rms4_apply(
 const merlin_source_rms4_plan *p,double exact_center,float rigorous_low,float rigorous_high){
 merlin_source_rms4_estimate failed={0,0,0};
 if(!p||!p->valid||!MERLIN_SOURCE_ISFINITE(exact_center)||
    !MERLIN_SOURCE_ISFINITE(rigorous_low)||!MERLIN_SOURCE_ISFINITE(rigorous_high)||
    rigorous_low>exact_center||rigorous_high<exact_center)return failed;
 if(p->ratio==1)return (merlin_source_rms4_estimate){rigorous_low,rigorous_high,1};
 double left=exact_center-(double)rigorous_low,right=(double)rigorous_high-exact_center;
 double radius=left>right?left:right;
 if(radius<p->eta)return (merlin_source_rms4_estimate){rigorous_low,rigorous_high,1};
 double estimated=merlin_fma_up_add(merlin_fma_up_mul(radius-p->eta,p->ratio),p->eta);
 double l=merlin_fma_down_add(exact_center,-estimated),h=merlin_fma_up_add(exact_center,estimated);
 float lo=(float)l,hi=(float)h;
 if((double)lo>l)lo=merlin_fma_next_down_f32(lo);
 if((double)hi<h)hi=merlin_fma_next_up_f32(hi);
 if(lo<rigorous_low)lo=rigorous_low;if(hi>rigorous_high)hi=rigorous_high;
 return (merlin_source_rms4_estimate){lo,hi,1};
}
#endif
