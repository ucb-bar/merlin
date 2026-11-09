#ifndef MERLIN_SOURCE_F32_PAIR_H
#define MERLIN_SOURCE_F32_PAIR_H
#include "source_f32_math.h"

/* Two independent source FMAs. The caller supplies distinct output locations,
 * a stable rounding environment and nontrapping/unobserved exception flags.
 * A selected implementation must retain each operand order, one rounding per
 * result, gradual underflow, and the source nonfinite behavior. */
static inline void merlin_source_f32_fma_pair(
    float al,float bl,float cl,float ah,float bh,float ch,float *low,float *high) {
  float l=MERLIN_SOURCE_F32_FMA(al,bl,cl);
  float h=MERLIN_SOURCE_F32_FMA(ah,bh,ch);
  *low=l;*high=h;
}
#ifndef MERLIN_SOURCE_F32_FMA_PAIR
#define MERLIN_SOURCE_F32_FMA_PAIR merlin_source_f32_fma_pair
#endif
#endif
