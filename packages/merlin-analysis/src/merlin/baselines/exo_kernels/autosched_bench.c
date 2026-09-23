/* Standalone f32 dot GEMM micro-benchmark for on-board autotuning of the EXO-AUTOSCHEDULED kernel
 * (K1, rv64gcv).
 *
 * Times fdot_nk_ref (from the exocc-emitted autosched_dot.c, linked per candidate nblock) on a
 * fixed representative shape and prints the rdtime ticks. The autotuner (merlin.baselines.exo)
 * compiles one binary per nblock candidate, runs each once under board_lock, and keeps the fastest
 * — a bounded empirical search (EXO has no cost model), not an automatic schedule search.
 *
 * Shape via -DBM= -DBN= -DBK= (default the lm_head-ish 8 x 4096 x 2048). Correctness is checked by
 * the whole-model gate; this only reports ticks + a checksum (guards dead-code elimination).
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "autosched_dot.h"   /* void fdot_nk_ref(void*, M,N,K, float* Y, const float* X, const float* Wf) */

#ifndef BM
#define BM 8
#endif
#ifndef BN
#define BN 4096
#endif
#ifndef BK
#define BK 2048
#endif
#ifndef REPS
#define REPS 4
#endif

static inline uint64_t rd_time(void){ uint64_t t; __asm__ volatile("rdtime %0":"=r"(t)); return t; }

int main(void){
  static float X[BM*BK];
  static float Wf[BN*BK];
  static float Y[BM*BN];
  uint32_t s=12345u;
  for(long i=0;i<(long)BM*BK;i++){ s=s*1103515245u+12345u; X[i]=(float)((int)((s>>16)&0xFF)-128)/64.0f; }
  for(long i=0;i<(long)BN*BK;i++){ s=s*1103515245u+12345u; Wf[i]=(float)((int)((s>>16)&0xFF)-128)/64.0f; }
  fdot_nk_ref(0, BM, BN, BK, Y, X, Wf);   /* warm */
  uint64_t t0=rd_time();
  for(int r=0;r<REPS;r++) fdot_nk_ref(0, BM, BN, BK, Y, X, Wf);
  uint64_t t1=rd_time();
  double cs=0; for(long i=0;i<(long)BM*BN;i++) cs+=Y[i];
  printf("BENCH_TICKS %llu reps %d shape %dx%dx%d checksum %g\n",
         (unsigned long long)(t1-t0), REPS, BM, BN, BK, cs);
  return 0;
}
